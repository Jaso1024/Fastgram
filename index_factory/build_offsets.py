import numpy as np
import struct
from pathlib import Path

index_dir = Path("indices/reasoning")
eos_token = 151643

for f in sorted(index_dir.glob("tokenized.*")):
    shard_id = f.suffix.split(".")[-1]
    offset_file = index_dir / f"offset.{shard_id}"
    
    print(f"Building offsets for {f} -> {offset_file}")
    
    # Read entire file at once if it fits in RAM (each shard is 4GB, we have 64GB)
    tokens = np.fromfile(f, dtype='<u4')
    
    # Find indices where token == eos_token
    eos_indices = np.where(tokens == eos_token)[0]
    
    # Convert token indices to byte offsets
    # document start is 0, and every index after an EOS
    doc_start_indices = np.concatenate(([0], eos_indices + 1))
    
    # Filter out start index if it's past the end of the array
    doc_start_indices = doc_start_indices[doc_start_indices < len(tokens)]
    
    byte_offsets = doc_start_indices * 4
    
    print(f"  Found {len(byte_offsets)} documents.")
    
    # Save as u64
    byte_offsets.astype('<u8').tofile(offset_file)

print("Done.")