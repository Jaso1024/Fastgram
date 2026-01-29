# gram-rl

RL finetuning scaffolding for training with gram-assisted decoding.

Target setup:
- policy: `Qwen/Qwen2.5-1.5B-Instruct`
- task: GSM8K train
- draft: `fastgram` index built from DeepSeek reasoning traces
- decoding: loose gram decoding (not distribution-correct)

GCS index:
- `gs://fastgram-indices-jaso1024/reasoning`

## Train (single GPU)

```bash
python gram-rl/scripts/train_grpo_gsm8k.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --index-gcs gs://fastgram-indices-jaso1024/reasoning \
  --index-dir index/reasoning \
  --sync-index \
  --steps 1000
```

