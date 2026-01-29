from huggingface_hub import HfFileSystem
import sys

dataset = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
print(f"Listing files in {dataset}...")

fs = HfFileSystem()
try:
    files = fs.ls(f"hf://datasets/{dataset}", detail=False)
    for f in files:
        print(f)
        
    print("Recursive glob for parquet...")
    files = fs.glob(f"hf://datasets/{dataset}/**/*.parquet")
    for f in files:
        print(f)

except Exception as e:
    print(f"Error: {e}")