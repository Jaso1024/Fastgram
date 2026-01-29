from datasets import load_dataset
import sys

dataset_name = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
subset = "am_0.9M"

print(f"Inspecting {dataset_name} ({subset})...")
try:
    ds = load_dataset(dataset_name, subset, split="train", streaming=True)
    print("Features:", ds.features)
    
    print("First item keys:", next(iter(ds)).keys())
except Exception as e:
    print("Error:", e)
