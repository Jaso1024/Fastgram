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
  --accept-bonus 0.2 \
  --steps 1000
```

## Train (4 GPUs, DDP)

```bash
torchrun --standalone --nproc_per_node=4 gram-rl/scripts/train_grpo_gsm8k.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --index-dir index/reasoning \
  --wandb \
  --accept-bonus 0.2 \
  --steps 1000
```
