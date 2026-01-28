#!/bin/bash
set -e

# Configuration
TOKENIZER="Qwen/Qwen2.5-72B-Instruct"
SHARD_SIZE=1000000000 # 1B tokens per shard (~4GB file size)
BUILD_TOOL="../build/tg_build_index"
PARALLEL_JOBS=8 # Number of parallel builds (Adjust based on RAM: ~40GB RAM per job)

# Ensure build tool exists
if [ ! -f "$BUILD_TOOL" ]; then
    echo "Building tg_build_index..."
    cd ..
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target tg_build_index -j
    cd index_factory
fi

build_single_shard() {
    f=$1
    DIR=$(dirname "$f")
    SHARD_ID="${f##*.}" # Extract extension
    OUT_DIR="$DIR/shard_$SHARD_ID"
    mkdir -p "$OUT_DIR"
    
    # Link/Move tokenized file to shard dir as tokenized.0
    ln -f "$f" "$OUT_DIR/tokenized.0" || cp "$f" "$OUT_DIR/tokenized.0"
    
    echo "[$(date +%T)] Building table for shard $SHARD_ID in $DIR..."
    # Usage: tg_build_index <in_dir> <out_dir> <token_width> <version> <mode> [ram_cap]
    # token_width=4 (u32), version=4, mode=table_only
    "$BUILD_TOOL" "$OUT_DIR" "$OUT_DIR" 4 4 table_only > /dev/null
    
    mv "$OUT_DIR/table.0" "$DIR/table.$SHARD_ID"
    rm -rf "$OUT_DIR"
    echo "[$(date +%T)] Finished shard $SHARD_ID in $DIR"
}

export -f build_single_shard
export BUILD_TOOL

# Function to build index from tokenized shards
build_shards() {
    DIR=$1
    echo "Building indices for $DIR with $PARALLEL_JOBS parallel jobs..."
    find "$DIR" -name "tokenized.*" -print0 | xargs -0 -n 1 -P "$PARALLEL_JOBS" -I {} bash -c 'build_single_shard "$@"' _ {}
}

# 1. Reasoning Index
echo ">>> Ingesting Reasoning Index..."
python3 ingest.py \
    --dataset "a-m-team/AM-DeepSeek-R1-Distilled-1.4M" \
    --output-dir "indices/reasoning" \
    --tokenizer "$TOKENIZER" \
    --shard-size "$SHARD_SIZE"

build_shards "indices/reasoning"

# 2. Knowledge Index (FineWeb-Edu)
echo ">>> Ingesting Knowledge Index..."
python3 ingest.py \
    --dataset "HuggingFaceFW/fineweb-edu" \
    --subset "sample-10BT" \
    --output-dir "indices/knowledge" \
    --tokenizer "$TOKENIZER" \
    --shard-size "$SHARD_SIZE"

build_shards "indices/knowledge"

# 3. Skill Index (The Stack - filtering for Code)
echo ">>> Ingesting Skill Index..."
python3 ingest.py \
    --dataset "bigcode/the-stack-v2-dedup" \
    --subset "Python" \
    --output-dir "indices/skill" \
    --tokenizer "$TOKENIZER" \
    --shard-size "$SHARD_SIZE" \
    --limit 100000

build_shards "indices/skill"

echo "All indices built in index_factory/indices/"