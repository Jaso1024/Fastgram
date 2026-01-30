#!/bin/bash
# RunPod Instance 1: FineWeb (8M docs)
# 32 vCPU / 64GB RAM
set -ex

cd ~
apt-get update && apt-get install -y cmake build-essential git

# Clone repo
git clone --branch gram-decoding-rl https://github.com/Jaso1024/Fastgram.git gram
cd gram

# Python deps
pip install torch transformers datasets huggingface_hub hf_transfer

# HF login
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# Build C++ tools
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tg_build_index -j32

# Build FineWeb index
cd index_factory
echo "=== FINEWEB START: $(date) ==="
python3 ingest_single.py --dataset fineweb --limit 8000000 --output-dir ~/index/fineweb --proc 32
RAM_CAP=58000000000 ./build_table.sh ~/index/fineweb
echo "=== FINEWEB DONE: $(date) ==="

ls -lh ~/index/fineweb/
echo "COMPLETE - copy ~/index/fineweb/ to your machine"
