#!/bin/bash
set -e

SERVER="fastgram-build-node"
ZONE="us-central1-a"

echo ">>> Packaging code..."
tar -czf fastgram_deploy.tar.gz \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='index_factory/indices' \
    --exclude='bench/build_inputs' \
    --exclude='bench/build_outputs' \
    --exclude='bench/build_refs' \
    --exclude='dist' \
    --exclude='index' \
    --exclude='fastgram_deploy.tar.gz' \
    .

echo ">>> Uploading to $SERVER..."
gcloud compute scp --zone="$ZONE" fastgram_deploy.tar.gz $SERVER:~/

echo ">>> Extracting and setting up on $SERVER..."
gcloud compute ssh $SERVER --zone="$ZONE" --command="
    mkdir -p Fastgram
    tar -xzf fastgram_deploy.tar.gz -C Fastgram
    rm fastgram_deploy.tar.gz
    cd Fastgram
    bash setup_node.sh
"

echo ">>> Starting Build on $SERVER..."
gcloud compute ssh $SERVER --zone="$ZONE" --command="
    cd Fastgram
    source venv/bin/activate
    cd index_factory
    # Run in background
    nohup ./build_indices.sh > main_build.log 2>&1 &
    echo 'Build started in background. Tailing log...'
    tail -f main_build.log
"

# Cleanup local tar
rm fastgram_deploy.tar.gz
