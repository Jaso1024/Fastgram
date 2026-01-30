#!/bin/bash
set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/indices/medium}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-1.5B-Instruct}"
BUILD_TOOL="$ROOT_DIR/build/tg_build_index"

# Limits for medium index (~10M docs)
OPENTHOUGHTS_LIMIT="${OPENTHOUGHTS_LIMIT:-}"  # Full 1.2M
TULU_LIMIT="${TULU_LIMIT:-}"                   # Full 939K
MAGPIE_LIMIT="${MAGPIE_LIMIT:-}"               # Full 300K
FINEWEB_LIMIT="${FINEWEB_LIMIT:-8000000}"      # 8M docs

echo "=============================================="
echo "Building Medium Gram Index"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo "Tokenizer: $TOKENIZER"
echo "Limits: OpenThoughts=${OPENTHOUGHTS_LIMIT:-full}, Tulu=${TULU_LIMIT:-full}, Magpie=${MAGPIE_LIMIT:-full}, FineWeb=${FINEWEB_LIMIT}"
echo ""

# Step 1: Build C++ tools if needed
if [ ! -f "$BUILD_TOOL" ]; then
    echo ">>> Building C++ tools..."
    cd "$ROOT_DIR"
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target tg_build_index -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
    cd "$SCRIPT_DIR"
    echo ">>> C++ tools built!"
fi

# Step 2: Run ingest
echo ""
echo ">>> Starting ingestion..."
mkdir -p "$OUTPUT_DIR"

INGEST_ARGS=(
    --output-dir "$OUTPUT_DIR"
    --tokenizer "$TOKENIZER"
    --fineweb-limit "$FINEWEB_LIMIT"
)

[ -n "$OPENTHOUGHTS_LIMIT" ] && INGEST_ARGS+=(--openthoughts-limit "$OPENTHOUGHTS_LIMIT")
[ -n "$TULU_LIMIT" ] && INGEST_ARGS+=(--tulu-limit "$TULU_LIMIT")
[ -n "$MAGPIE_LIMIT" ] && INGEST_ARGS+=(--magpie-limit "$MAGPIE_LIMIT")

python3 "$SCRIPT_DIR/ingest_medium.py" "${INGEST_ARGS[@]}"

echo ""
echo ">>> Ingestion complete!"

# Step 3: Build suffix array tables
echo ""
echo ">>> Building suffix array tables..."

build_shard() {
    local f="$1"
    local dir=$(dirname "$f")
    local shard_id="${f##*.}"
    local final_table="$dir/table.$shard_id"

    if [ -f "$final_table" ]; then
        echo "[SKIP] table.$shard_id already exists"
        return 0
    fi

    local out_dir="$dir/shard_$shard_id"
    mkdir -p "$out_dir"

    # Link tokenized file
    ln -f "$f" "$out_dir/tokenized.0" 2>/dev/null || cp "$f" "$out_dir/tokenized.0"

    echo "[BUILD] Building table.$shard_id..."
    # token_width=4 (u32), version=4, mode=full
    # RAM cap depends on available memory - use 32GB default
    local ram_cap="${RAM_CAP:-32000000000}"
    "$BUILD_TOOL" "$out_dir" "$out_dir" 4 4 full "$ram_cap"

    if [ -f "$out_dir/table.0" ]; then
        mv "$out_dir/table.0" "$final_table"
        rm -rf "$out_dir"
        echo "[DONE] table.$shard_id"
    else
        echo "[ERROR] Failed to build table.$shard_id"
        return 1
    fi
}

# Process all shards
for f in "$OUTPUT_DIR"/tokenized.*; do
    [ -f "$f" ] || continue
    build_shard "$f"
done

echo ""
echo "=============================================="
echo "BUILD COMPLETE!"
echo "=============================================="
echo "Index location: $OUTPUT_DIR"
echo ""
echo "Files:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "To use this index:"
echo "  from fastgram import GramEngine"
echo "  engine = GramEngine("
echo "      index_dir='$OUTPUT_DIR',"
echo "      eos_token_id=151643,"
echo "      vocab_size=151936,"
echo "      version=4,"
echo "      token_dtype='u32',"
echo "  )"
