#!/bin/bash
set -e

# Configuration
TOKENIZER="Qwen/Qwen2.5-72B-Instruct"
SHARD_SIZE=2000000000 # 2B tokens per shard (~8GB file size)
BUILD_TOOL="../build/tg_build_index"

# Ensure build tool exists
if [ ! -f "$BUILD_TOOL" ]; then
    echo "Building tg_build_index..."
    cd ..
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target tg_build_index -j
    cd index_factory
fi

# Function to build index from tokenized shards
build_shards() {
    DIR=$1
    echo "Building indices for $DIR..."
    for f in "$DIR"/tokenized.*; do
        if [ -f "$f" ]; then
            SHARD_ID="${f##*.}" # Extract extension (0, 1, 2...)
            OUT_DIR="$DIR/shard_$SHARD_ID"
            mkdir -p "$OUT_DIR"
            
            # Link/Move tokenized file to shard dir as tokenized.0 (tg_build_index expects .0)
            # We use hardlink if possible to save space, or cp
            ln -f "$f" "$OUT_DIR/tokenized.0" || cp "$f" "$OUT_DIR/tokenized.0"
            
            echo "Building table for shard $SHARD_ID..."
            # Usage: tg_build_index <in_dir> <out_dir> <token_width> <version> <mode> [ram_cap]
            # token_width=4 (u32), version=4, mode=table_only
            "$BUILD_TOOL" "$OUT_DIR" "$OUT_DIR" 4 4 table_only
            
            # Clean up the temporary link if we want, but fastgram needs tokenized.0 in the directory
            # So we actually leave it there. The original large directory is just a staging area.
            
            # Optional: Move the generated table.0 back to main dir if we supported flat structure,
            # but fastgram works best with "index_dir" containing shards.
            # Actually, fastgram treats a directory as a collection of shards if it finds tokenized.0, tokenized.1 etc?
            # No, `Index::LoadIndexDir` expects `tokenized.*`, `table.*` in ONE directory.
            # So we should move `table.0` -> `table.$SHARD_ID` in the main directory.
            
            mv "$OUT_DIR/table.0" "$DIR/table.$SHARD_ID"
            
            # We don't need offset/metadata for pure n-gram counting usually, but if we want document retrieval:
            # We need to build "full" mode. However, "full" requires offset.0 etc which ingest.py DOES NOT produce yet.
            # ingest.py only produces tokenized sequence. 
            # To get offsets/metadata, ingest.py needs to write them. 
            # For now, we are building a "table_only" index (good for counting/next-token-dist).
            
            rm -rf "$OUT_DIR"
        fi
    done
}

# 1. Reasoning Index
echo ">>> Ingesting Reasoning Index..."
python3 ingest.py 
    --dataset "a-m-team/AM-DeepSeek-R1-Distilled-1.4M" 
    --output-dir "indices/reasoning" 
    --tokenizer "$TOKENIZER" 
    --shard-size "$SHARD_SIZE"

build_shards "indices/reasoning"

# 2. Knowledge Index (FineWeb-Edu)
echo ">>> Ingesting Knowledge Index..."
python3 ingest.py 
    --dataset "HuggingFaceFW/fineweb-edu" 
    --subset "sample-10BT" 
    --output-dir "indices/knowledge" 
    --tokenizer "$TOKENIZER" 
    --shard-size "$SHARD_SIZE"

build_shards "indices/knowledge"

# 3. Skill Index (The Stack - filtering for Code)
echo ">>> Ingesting Skill Index..."
python3 ingest.py 
    --dataset "bigcode/the-stack-v2-dedup" 
    --subset "Python" 
    --output-dir "indices/skill" 
    --tokenizer "$TOKENIZER" 
    --shard-size "$SHARD_SIZE" 
    --limit 100000 # Limit to avoid fetching petabytes for this demo script

build_shards "indices/skill"

echo "All indices built in index_factory/indices/"
