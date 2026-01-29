#!/bin/bash
set -e

# Configuration
TOKENIZER="Qwen/Qwen2.5-72B-Instruct"
SHARD_SIZE=1000000000 # 1B tokens per shard (~4GB file size)
BUILD_TOOL="../build/tg_build_index"
# PARALLEL_JOBS=8  <-- Too much for 64GB RAM.
# We need ~40GB per build job. With 64GB total, we can safely run 1 job.
PARALLEL_JOBS=1 
LOG_FILE="build_progress.log"

# Setup Logging
exec > >(tee -a "$LOG_FILE") 2>&1

echo ">>> Starting Build Process at $(date)"
echo ">>> Config: Shard Size=$SHARD_SIZE, Jobs=$PARALLEL_JOBS"

# Ensure build tool exists
if [ ! -f "$BUILD_TOOL" ]; then
    echo "Building tg_build_index..."
    cd ..
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target tg_build_index -j
    cd index_factory
fi

build_single_shard() {
    f=
    DIR=$(dirname "$f")
    SHARD_ID="${f##*.}" # Extract extension
    FINAL_TABLE="$DIR/table.$SHARD_ID"
    
    # Checkpoint: Skip if table already exists
    if [ -f "$FINAL_TABLE" ]; then
        echo "[$(date +%T)] SKIPPING shard $SHARD_ID in $DIR (Already built)"
        return 0
    fi

    OUT_DIR="$DIR/shard_$SHARD_ID"
    mkdir -p "$OUT_DIR"
    
    # Link/Move tokenized file to shard dir as tokenized.0
    ln -f "$f" "$OUT_DIR/tokenized.0" || cp "$f" "$OUT_DIR/tokenized.0"
    
    echo "[$(date +%T)] Building table and LCP for shard $SHARD_ID in $DIR..."
    # Usage: tg_build_index <in_dir> <out_dir> <token_width> <version> <mode> [ram_cap]
    # token_width=4 (u32), version=4, mode=full, ram_cap=60GB (approx)
    "$BUILD_TOOL" "$OUT_DIR" "$OUT_DIR" 4 4 full 60000000000
    
    if [ -f "$OUT_DIR/table.0" ]; then
        mv "$OUT_DIR/table.0" "$FINAL_TABLE"
        rm -rf "$OUT_DIR"
        echo "[$(date +%T)] FINISHED shard $SHARD_ID in $DIR"
    else
        echo "[$(date +%T)] ERROR: Failed to build table for shard $SHARD_ID"
        exit 1
    fi
}

export -f build_single_shard
export BUILD_TOOL

# Function to build index from tokenized shards
build_shards() {
    DIR="$1"
    echo "Building indices for $DIR with $PARALLEL_JOBS parallel jobs..."
    find "$DIR" -name "tokenized.*" -print0 | xargs -0 -n 1 -P "$PARALLEL_JOBS" -I {} bash -c 'build_single_shard "$@"' _ {}
}

# 1. Reasoning Index
echo ">>> Ingesting Reasoning Index..."
python3 ingest.py \
    --dataset "a-m-team/AM-DeepSeek-R1-Distilled-1.4M" \
    --subset "am_0.9M" \
    --output-dir "indices/reasoning" \
    --tokenizer "$TOKENIZER" \
    --shard-size "$SHARD_SIZE"

build_shards "indices/reasoning"

echo ">>> Reasoning Index Done!"
# Commenting out the huge ones for now to focus on the request
# ... (Knowledge / Skill sections commented out for speed)