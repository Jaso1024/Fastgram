#!/bin/bash
# RunPod Instance 2: OpenThoughts + Tulu + Magpie (2.4M docs)
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

cd index_factory

# OpenThoughts (1.2M docs)
echo "=== OPENTHOUGHTS START: $(date) ==="
python3 ingest_single.py --dataset openthoughts --output-dir ~/index/openthoughts --proc 32
RAM_CAP=58000000000 ./build_table.sh ~/index/openthoughts
echo "=== OPENTHOUGHTS DONE: $(date) ==="

# Tulu (939K docs)
echo "=== TULU START: $(date) ==="
python3 ingest_single.py --dataset tulu --output-dir ~/index/tulu --proc 32
RAM_CAP=58000000000 ./build_table.sh ~/index/tulu
echo "=== TULU DONE: $(date) ==="

# Magpie (300K docs)
echo "=== MAGPIE START: $(date) ==="
python3 ingest_single.py --dataset magpie --output-dir ~/index/magpie --proc 32
RAM_CAP=58000000000 ./build_table.sh ~/index/magpie
echo "=== MAGPIE DONE: $(date) ==="

ls -lh ~/index/*/
echo "COMPLETE - copy ~/index/* to your machine"
