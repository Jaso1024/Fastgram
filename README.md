# gram

Single-machine mmap n-gram engine compatible with InfiniGram-style shard directories (`tokenized.*`, `table.*`, `offset.*`).

Build:

`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`

`cmake --build build -j`

Test:

`ctest --test-dir build --output-on-failure`

Tools:

- `tg_rpc`: stdin/stdout RPC for benchmarking + integration
- `tg_query`: quick CLI query helper
- `tg_build_unigram_ranges`: build `unigram_ranges.bin` for faster unigram range lookup

Python:

`python -m pip install -e .`

`python -c "from gram import GramEngine; print(GramEngine)"`

Download indices:

Requires AWS CLI (`aws`).

`gram`  # interactive

`gram list`

`gram download v4_pileval_gpt2 --to index/v4_pileval_gpt2`

Run:

`gram run --index index/v4_pileval_gpt2 --prompt "natural language processing"`

Interactive run:

`gram` -> `1` (run)

Settings in run mode:

`/settings`

`/set topk 50`

`/set temperature 0.8`

`/gen 20 hello world`

Notes:

- Uses the tokenizer specified for the index in the catalog.
- Some tokenizers require `HF_TOKEN` (for gated models like Llama-2).
