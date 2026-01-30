#!/bin/bash
set -e

# GCP Parallel Index Build
# Works within 12 CPU quota by running in batches
# Estimated cost: ~$5-10 total (spot instances)

PROJECT="argo-app-23158"
ZONE="us-central1-a"
REPO="https://github.com/Jaso1024/Fastgram.git"
BRANCH="gram-decoding-rl"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"

create_instance() {
    local name="$1"
    local machine="$2"
    local disk="$3"
    local dataset="$4"
    local limit="$5"
    local ram_cap="$6"

    local limit_arg=""
    [ -n "$limit" ] && limit_arg="--limit $limit"

    echo ">>> Creating $name ($machine, ${disk}GB disk, dataset=$dataset)"

    # Create startup script
    local startup=$(cat <<SCRIPT
#!/bin/bash
exec > >(tee /var/log/gram-build.log) 2>&1
set -ex

echo "=== START: \$(date) ==="

# Install deps
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv cmake build-essential git

# Clone repo
cd /root
git clone --branch $BRANCH $REPO gram
cd gram

# Python setup
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch transformers datasets huggingface_hub hf_transfer

# HF login with fast transfer
export HF_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli login --token "\$HF_TOKEN" --add-to-git-credential

# Build C++ tools
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tg_build_index -j\$(nproc)

# Ingest
cd index_factory
echo "=== INGEST START: \$(date) ==="
python3 ingest_single.py --dataset $dataset $limit_arg --output-dir /root/index --proc \$(nproc)

# Build table
echo "=== TABLE BUILD START: \$(date) ==="
RAM_CAP=$ram_cap ./build_table.sh /root/index

echo "=== COMPLETE: \$(date) ==="
ls -lh /root/index/
touch /root/DONE
SCRIPT
)

    gcloud compute instances create "$name" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$machine" \
        --provisioning-model=SPOT \
        --instance-termination-action=DELETE \
        --boot-disk-size="${disk}GB" \
        --boot-disk-type=pd-ssd \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --metadata=startup-script="$startup" \
        --scopes=cloud-platform \
        --tags=gram-build
}

cmd_batch1() {
    echo "=== BATCH 1: FineWeb (largest, uses 8 vCPUs) ==="
    # e2-highmem-8: 8 vCPU, 64GB RAM - $0.27/hr spot
    create_instance "gram-fineweb" "e2-highmem-8" "250" "fineweb" "8000000" "58000000000"
    echo ""
    echo "Batch 1 started. Monitor with: $0 status"
    echo "When complete, run: $0 batch2"
}

cmd_batch2() {
    echo "=== BATCH 2: OpenThoughts, Tulu, Magpie (4 vCPUs each = 12 total) ==="

    # e2-highmem-4: 4 vCPU, 32GB RAM - $0.13/hr spot each
    create_instance "gram-openthoughts" "e2-highmem-4" "100" "openthoughts" "" "28000000000" &
    create_instance "gram-tulu" "e2-highmem-4" "100" "tulu" "" "28000000000" &
    create_instance "gram-magpie" "e2-highmem-4" "80" "magpie" "" "28000000000" &

    wait
    echo ""
    echo "Batch 2 started. Monitor with: $0 status"
}

cmd_all() {
    echo "Running all batches (will wait for quota between batches)..."
    echo ""

    # Start batch 1
    cmd_batch1

    echo ""
    echo "Waiting for FineWeb to complete before starting batch 2..."
    echo "(This is the largest build, ~3-4 hours)"
    echo ""

    # Wait for batch1 to complete
    while true; do
        if gcloud compute ssh gram-fineweb --zone="$ZONE" --command="test -f /root/DONE" 2>/dev/null; then
            echo "FineWeb complete!"
            break
        fi
        echo -n "."
        sleep 60
    done

    # Delete fineweb to free up quota
    echo "Deleting gram-fineweb to free quota..."
    # First download
    mkdir -p indices/fineweb
    gcloud compute scp --recurse --compress --zone="$ZONE" \
        "gram-fineweb:/root/index/*" "indices/fineweb/"
    gcloud compute instances delete gram-fineweb --zone="$ZONE" --quiet

    # Start batch 2
    cmd_batch2
}

cmd_status() {
    echo "Instance status:"
    echo "================"

    for name in gram-fineweb gram-openthoughts gram-tulu gram-magpie; do
        echo -n "$name: "

        # Check if instance exists
        state=$(gcloud compute instances describe "$name" --zone="$ZONE" --format="value(status)" 2>/dev/null) || {
            echo "NOT FOUND"
            continue
        }

        if [ "$state" != "RUNNING" ]; then
            echo "$state"
            continue
        fi

        # Try to get build status
        status=$(gcloud compute ssh "$name" --zone="$ZONE" --command="
            if [ -f /root/DONE ]; then
                echo 'COMPLETE'
            elif [ -f /var/log/gram-build.log ]; then
                tail -1 /var/log/gram-build.log 2>/dev/null | head -c 80
            else
                echo 'Initializing...'
            fi
        " 2>/dev/null) || status="SSH not ready"

        echo "$status"
    done
}

cmd_logs() {
    local name="${1:?Usage: $0 logs <instance-name>}"
    gcloud compute ssh "$name" --zone="$ZONE" --command="tail -50 /var/log/gram-build.log"
}

cmd_stream() {
    local name="${1:?Usage: $0 stream <instance-name>}"
    gcloud compute ssh "$name" --zone="$ZONE" --command="tail -f /var/log/gram-build.log"
}

cmd_download() {
    local dest="${1:-indices}"
    mkdir -p "$dest"

    for name in gram-fineweb gram-openthoughts gram-tulu gram-magpie; do
        local dataset="${name#gram-}"
        echo ">>> Downloading $name -> $dest/$dataset/"

        # Check if instance exists and is complete
        if ! gcloud compute instances describe "$name" --zone="$ZONE" &>/dev/null; then
            # Check if already downloaded
            if [ -d "$dest/$dataset" ] && [ -n "$(ls -A "$dest/$dataset" 2>/dev/null)" ]; then
                echo "    Already downloaded"
            else
                echo "    SKIPPED (instance not found)"
            fi
            continue
        fi

        if ! gcloud compute ssh "$name" --zone="$ZONE" --command="test -f /root/DONE" 2>/dev/null; then
            echo "    SKIPPED (not complete)"
            continue
        fi

        mkdir -p "$dest/$dataset"
        gcloud compute scp --recurse --compress --zone="$ZONE" \
            "$name:/root/index/*" "$dest/$dataset/"
        echo "    DONE"
    done

    echo ""
    echo "Downloaded indices:"
    ls -la "$dest"/*/ 2>/dev/null || echo "(none yet)"
}

cmd_delete() {
    echo "Deleting all gram-* instances..."
    gcloud compute instances delete \
        gram-fineweb gram-openthoughts gram-tulu gram-magpie \
        --zone="$ZONE" --quiet 2>/dev/null || true
    echo "Done"
}

cmd_cost() {
    echo "Estimated costs (spot pricing, us-central1):"
    echo "============================================"
    echo "gram-fineweb:      e2-highmem-8  @ \$0.27/hr x ~4hr = ~\$1.08"
    echo "gram-openthoughts: e2-highmem-4  @ \$0.13/hr x ~2hr = ~\$0.26"
    echo "gram-tulu:         e2-highmem-4  @ \$0.13/hr x ~1.5hr = ~\$0.20"
    echo "gram-magpie:       e2-highmem-4  @ \$0.13/hr x ~1hr = ~\$0.13"
    echo "--------------------------------------------"
    echo "TOTAL ESTIMATE: ~\$1.67 (well under \$100 budget)"
    echo ""
    echo "Disk: ~530GB @ \$0.10/GB-mo = ~\$1.70/day"
}

case "${1:-help}" in
    batch1) cmd_batch1 ;;
    batch2) cmd_batch2 ;;
    all) cmd_all ;;
    status) cmd_status ;;
    logs) cmd_logs "$2" ;;
    stream) cmd_stream "$2" ;;
    download) cmd_download "$2" ;;
    delete) cmd_delete ;;
    cost) cmd_cost ;;
    help|*)
        echo "GCP Parallel Index Builder"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands (within 12 CPU quota):"
        echo "  batch1    - Start FineWeb build (8 vCPUs, ~4hr)"
        echo "  batch2    - Start OpenThoughts+Tulu+Magpie (12 vCPUs total, ~2hr)"
        echo "  all       - Run both batches sequentially"
        echo ""
        echo "Monitoring:"
        echo "  status    - Check build progress"
        echo "  logs NAME - View recent logs"
        echo "  stream NAME - Stream logs (tail -f)"
        echo ""
        echo "Results:"
        echo "  download [DIR] - Download completed indices"
        echo "  delete    - Delete all instances"
        echo "  cost      - Show cost estimate"
        echo ""
        echo "Recommended workflow:"
        echo "  1. $0 batch1          # Start FineWeb (largest)"
        echo "  2. $0 status          # Monitor progress"
        echo "  3. $0 download        # Download when batch1 complete"
        echo "  4. $0 delete          # Free quota"
        echo "  5. $0 batch2          # Start remaining 3"
        echo "  6. $0 download        # Download all"
        echo "  7. $0 delete          # Cleanup"
        ;;
esac
