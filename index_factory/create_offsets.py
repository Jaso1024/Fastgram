#!/usr/bin/env python
"""
Create offset files for indices that don't have them.
Each shard gets one document spanning the entire shard.
"""
import struct
import sys
from pathlib import Path


def create_offset_file(tokenized_path: Path):
    """Create offset.X file for tokenized.X file."""
    shard_id = tokenized_path.name.split('.')[-1]
    offset_path = tokenized_path.parent / f"offset.{shard_id}"

    if offset_path.exists():
        print(f"  {offset_path.name} already exists, skipping")
        return

    # Get size of tokenized file (in bytes)
    token_bytes = tokenized_path.stat().st_size

    # Create offset file with single document starting at 0
    # offset.X contains uint64 values, each marking a document start byte position
    # For a single document, we just need [0]
    with open(offset_path, 'wb') as f:
        f.write(struct.pack('<Q', 0))  # Document starts at byte 0

    print(f"  Created {offset_path.name} (1 doc, {token_bytes} bytes)")


def process_index(index_dir: Path):
    """Create offset files for all shards in an index."""
    print(f"Processing {index_dir}")

    tokenized_files = sorted(index_dir.glob("tokenized.*"))
    if not tokenized_files:
        print(f"  No tokenized files found")
        return

    for tf in tokenized_files:
        create_offset_file(tf)


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_offsets.py <index_dir> [index_dir2] ...")
        sys.exit(1)

    for arg in sys.argv[1:]:
        process_index(Path(arg))


if __name__ == "__main__":
    main()
