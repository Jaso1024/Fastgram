#!/bin/bash
# Comprehensive overnight index build
# For 255-core vast.ai server with /dev/shm storage
set -e

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
export HF_HUB_ENABLE_HF_TRANSFER=1

PROC=64  # Conservative to avoid overload
INDEX_DIR=/root/index
RAM_CAP=400000000000  # 400GB per table build (plenty of room in 2TB)

cd ~/gram
git pull origin gram-decoding-rl

# Build C++ tools if needed
if [ ! -f build/tg_build_index ]; then
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
    cmake --build build --target tg_build_index -j64
fi

cd index_factory

# ============================================================================
# Skip already-completed datasets
# ============================================================================
skip_if_done() {
    local name="$1"
    if [ -f "$INDEX_DIR/$name/table.0" ]; then
        echo ">>> SKIPPING $name (already complete)"
        return 0
    fi
    return 1
}

# ============================================================================
# Build function - ingest + table
# ============================================================================
build_dataset() {
    local name="$1"
    local limit="$2"

    skip_if_done "$name" && return 0

    echo ""
    echo "============================================"
    echo "=== $name START: $(date) ==="
    echo "============================================"

    local limit_arg=""
    [ -n "$limit" ] && limit_arg="--limit $limit"

    # Ingest
    python3 ingest_single.py --dataset "$name" --output-dir "$INDEX_DIR/$name" --proc $PROC $limit_arg

    # Build table
    RAM_CAP=$RAM_CAP ./build_table.sh "$INDEX_DIR/$name"

    echo "=== $name DONE: $(date) ==="
    ls -lh "$INDEX_DIR/$name/"
}

# ============================================================================
# Build order: fastest first, then larger ones
# ============================================================================

echo "Starting comprehensive overnight build: $(date)"
echo "Index dir: $INDEX_DIR"
echo "Processes: $PROC"
echo ""

# Already done: magpie, tulu
# Partial: openthoughts (will rebuild from scratch)

# Reasoning datasets (~2.2M total)
build_dataset "openthoughts" ""
build_dataset "openthoughts2" ""

# Code datasets (~7M total)
build_dataset "github-code-2025" ""
build_dataset "github-code" ""

# General knowledge (20M - the big one, run last)
build_dataset "fineweb" ""

echo ""
echo "============================================"
echo "ALL COMPLETE: $(date)"
echo "============================================"
echo ""
du -sh $INDEX_DIR/*
echo ""
ls -lh $INDEX_DIR/*/table.*
