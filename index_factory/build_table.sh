#!/bin/bash
set -e

# Build suffix array table for an ingested index
#
# Usage:
#   ./build_table.sh indices/openthoughts
#   ./build_table.sh indices/tulu
#   RAM_CAP=64000000000 ./build_table.sh indices/fineweb

INDEX_DIR="${1:?Usage: build_table.sh <index_dir>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_TOOL="$ROOT_DIR/build/tg_build_index"
RAM_CAP="${RAM_CAP:-32000000000}"  # 32GB default

echo "=============================================="
echo "Building tables for: $INDEX_DIR"
echo "RAM cap: $((RAM_CAP / 1000000000))GB"
echo "=============================================="

# Build C++ tools if needed
if [ ! -f "$BUILD_TOOL" ]; then
    echo ">>> Building C++ tools..."
    cd "$ROOT_DIR"
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target tg_build_index -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cd "$SCRIPT_DIR"
fi

# Build each shard
for f in "$INDEX_DIR"/tokenized.*; do
    [ -f "$f" ] || continue

    shard_id="${f##*.}"
    final_table="$INDEX_DIR/table.$shard_id"

    if [ -f "$final_table" ]; then
        echo "[SKIP] table.$shard_id exists"
        continue
    fi

    out_dir="$INDEX_DIR/shard_$shard_id"
    mkdir -p "$out_dir"

    ln -f "$f" "$out_dir/tokenized.0" 2>/dev/null || cp "$f" "$out_dir/tokenized.0"

    echo "[BUILD] table.$shard_id ..."
    "$BUILD_TOOL" "$out_dir" "$out_dir" 4 4 full "$RAM_CAP"

    if [ -f "$out_dir/table.0" ]; then
        mv "$out_dir/table.0" "$final_table"
        rm -rf "$out_dir"
        echo "[DONE] table.$shard_id"
    else
        echo "[ERROR] Failed: table.$shard_id"
        exit 1
    fi
done

echo ""
echo "=============================================="
echo "BUILD COMPLETE: $INDEX_DIR"
echo "=============================================="
ls -lh "$INDEX_DIR"
