# gram-decoding

Draft decoding with `fastgram` as the draft model and a HuggingFace causal LM as the verifier.

This is a pragmatic, "gram decoding" variant inspired by speculative decoding:
- `fastgram` proposes up to `k` draft tokens using next-token distributions (`infgram` by default, i.e. suffix backoff).
- the target model verifies the draft in a single forward pass
- optional attention-mass gating can force an early stop in the draft based on verifier attention

## Setup

From repo root:

```bash
pip install -e .
pip install torch transformers
```

## Run

```bash
python gram-decoding/scripts/gram_decode.py \
  --model gpt2 \
  --index-dir index/v4_pileval_gpt2 \
  --prompt "Write a short paragraph about fast text search." \
  --max-new-tokens 64 \
  --draft-k 8 \
  --max-support 200
```

With attention gating:

```bash
python gram-decoding/scripts/gram_decode.py \
  --model gpt2 \
  --index-dir index/v4_pileval_gpt2 \
  --prompt "Write a short paragraph about fast text search." \
  --gate attn \
  --attn-threshold 0.02
```

Cheaper gates (no attentions):

```bash
python gram-decoding/scripts/gram_decode.py \
  --model gpt2 \
  --index-dir index/v4_pileval_gpt2 \
  --prompt "Write a short paragraph about fast text search." \
  --gate margin \
  --gate-margin 2.0
```
