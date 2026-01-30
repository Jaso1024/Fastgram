#!/bin/bash
# RunPod: All datasets on 128 vCPU / 1Gbps
# Est. time: 30-45 minutes total
set -ex

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
export HF_HUB_ENABLE_HF_TRANSFER=1
PROC=120  # Leave some cores for system

cd ~
apt-get update && apt-get install -y cmake build-essential git

git clone --branch gram-decoding-rl https://github.com/Jaso1024/Fastgram.git gram
cd gram

pip install torch transformers datasets huggingface_hub hf_transfer
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# Build C++ tools with all cores
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
cmake --build build --target tg_build_index -j128

cd index_factory

# Run all ingests in parallel (they're I/O bound on download)
echo "=== PARALLEL INGEST START: $(date) ==="
python3 ingest_single.py --dataset openthoughts --output-dir ~/index/openthoughts --proc $PROC &
python3 ingest_single.py --dataset tulu --output-dir ~/index/tulu --proc $PROC &
python3 ingest_single.py --dataset magpie --output-dir ~/index/magpie --proc $PROC &
python3 ingest_single.py --dataset fineweb --limit 8000000 --output-dir ~/index/fineweb --proc $PROC &
wait
echo "=== PARALLEL INGEST DONE: $(date) ==="

# Build tables in parallel (1TB RAM = ~250GB each)
echo "=== PARALLEL TABLE BUILDS START: $(date) ==="
RAM_CAP=250000000000 ./build_table.sh ~/index/magpie &
RAM_CAP=250000000000 ./build_table.sh ~/index/tulu &
RAM_CAP=250000000000 ./build_table.sh ~/index/openthoughts &
RAM_CAP=250000000000 ./build_table.sh ~/index/fineweb &
wait
echo "=== PARALLEL TABLE BUILDS DONE: $(date) ==="

echo ""
echo "============================================"
echo "ALL COMPLETE: $(date)"
echo "============================================"
ls -lh ~/index/*/
du -sh ~/index/*

echo ""
echo "To copy to your machine:"
echo "  rsync -avz ~/index/ user@your-machine:~/gram-indices/"
