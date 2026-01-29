#!/bin/bash
set -e

# Configuration
BUILD_TOOL="../build/tg_build_index"
PARALLEL_JOBS=1
LOG_FILE="build_progress.log"

exec > >(tee -a "$LOG_FILE") 2>&1

build_single_shard() {
    f="$1"
    DIR=$(dirname "$f")
    SHARD_ID="${f##*.}" 
    FINAL_TABLE="$DIR/table.$SHARD_ID"
    
    if [ -f "$FINAL_TABLE" ]; then
        echo "[$(date +%T)] SKIPPING shard $SHARD_ID in $DIR (Already built)"
        return 0
    fi

    OUT_DIR="$DIR/shard_$SHARD_ID"
    mkdir -p "$OUT_DIR"
    
    # Link tokenized file
    ln -f "$f" "$OUT_DIR/tokenized.0" || cp "$f" "$OUT_DIR/tokenized.0"
    
    echo "[$(date +%T)] Building table and LCP for shard $SHARD_ID in $DIR..."
    # token_width=4, version=4, mode=full (for LCP), ram_cap=60GB
    "$BUILD_TOOL" "$OUT_DIR" "$OUT_DIR" 4 4 full 60000000000
    
    if [ -f "$OUT_DIR/table.0" ]; then
        mv "$OUT_DIR/table.0" "$FINAL_TABLE"
        # Also move auxiliary files if they exist (LCP, etc)
        [ -f "$OUT_DIR/lcp.0" ] && mv "$OUT_DIR/lcp.0" "$DIR/lcp.$SHARD_ID"
        # Remove the temp dir
        rm -rf "$OUT_DIR"
        echo "[$(date +%T)] FINISHED shard $SHARD_ID in $DIR"
    else
        echo "[$(date +%T)] ERROR: Failed to build table for shard $SHARD_ID"
        exit 1
    fi
}

export -f build_single_shard
export BUILD_TOOL

DIR="indices/reasoning"
echo "Removing duplicate shards (4-7)..."
rm -f "$DIR/tokenized.4" "$DIR/tokenized.5" "$DIR/tokenized.6" "$DIR/tokenized.7"

echo "Building indices for $DIR..."
find "$DIR" -name "tokenized.*" -print0 | xargs -0 -n 1 -P "$PARALLEL_JOBS" -I {} bash -c 'build_single_shard "$@"' _ {}

echo ">>> Build Complete!"
