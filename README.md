# Fastgram

High-performance memory-mapped n-gram engine compatible with InfiniGram-style shard directories (`tokenized.*`, `table.*`, `offset.*`).

## Installation

Install from PyPI:

```bash
pip install fast-gram
```

Or install from source:

```bash
git clone https://github.com/Jaso1024/Fastgram.git
cd Fastgram
pip install -e .
```

## Usage

Import the engine:

```python
from fastgram import gram

engine = gram(
    index_dir="path/to/index",
    eos_token_id=50256,
    vocab_size=50257,
    version=4,
    token_width="u16"
)

result = engine.count([15496, 3303])
print(result)
```

CLI tool:

```bash
gram list                     # list available indices
gram download <index>         # download an index
gram run --index <path>       # run queries interactively
```

## Building from Source

`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`

`cmake --build build -j`

Test:

`ctest --test-dir build --output-on-failure`

Tools:

- `tg_rpc`: stdin/stdout RPC for benchmarking + integration
- `tg_query`: quick CLI query helper
- `tg_build_unigram_ranges`: build `unigram_ranges.bin` for faster unigram range lookup
- `tools/run_bench.py`: benchmark runner (uses `bench/bench_config.json`)
- `tools/gen_bench_queries.py`: build deterministic query suite for coverage
- `tools/run_bench_suite.py`: suite runner (uses `bench/bench_suite_config.json`)
- `tg_slice_index`: build deterministic slices for build benchmarks
- `tg_build_index`: build table/full index (benchmark target)
- `tools/run_build_bench.py`: index build benchmark runner (uses `bench/build_bench_config.json`)
- `tools/verify_built_index.py`: correctness check for build outputs

Benchmarking:

Query benchmarks measure `find` and `ntd` operation performance:
- `python tools/run_bench.py` - runs find/ntd benchmarks using `bench/bench_config.json`
- `python tools/run_bench_suite.py` - runs comprehensive suite using `bench/bench_suite_config.json`

Build benchmarks measure index construction performance:

1. Create test slices from an existing index:
```bash
# Small slice: 2000 docs, token_width=2 (u16)
./build/tg_slice_index <source_index_dir> bench/build_inputs/small 2000 2

# Medium slice: 20000 docs, token_width=2 (u16)
./build/tg_slice_index <source_index_dir> bench/build_inputs/medium 20000 2
```

2. Build reference indices:
```bash
# token_width=2, version=4, mode=table_only, ram_cap=8GB
./build/tg_build_index bench/build_inputs/small bench/build_refs/small 2 4 table_only 8589934592
./build/tg_build_index bench/build_inputs/medium bench/build_refs/medium 2 4 table_only 8589934592
```

3. Run build benchmarks:
```bash
python tools/run_build_bench.py
```

## CLI Features

Interactive mode:

```bash
gram                          # start interactive mode
```

Download indices (requires AWS CLI):

```bash
gram list
gram download v4_pileval_gpt2 --to index/v4_pileval_gpt2
```

Run queries:

```bash
gram run --index index/v4_pileval_gpt2 --prompt "natural language processing"
```

Interactive run mode:

```bash
gram                          # main menu
# Select: 1 (run)
# Then:
/settings                     # show current settings
/set topk 50                  # set top-k results
/set temperature 0.8          # set sampling temperature
/gen 20 hello world           # generate 20 tokens
```

## Notes

- Build scripts auto-detect build directory or use `GRAM_BUILD_DIR` environment variable
- `ram_cap_bytes` in configs is 8589934592 (8GB) to limit memory during benchmarking
- Generate query suites with `python tools/gen_bench_queries.py --index-dir <path> --eos <eos_id> --vocab <vocab_size>`
- Uses the tokenizer specified for the index in the catalog
- Some tokenizers require `HF_TOKEN` environment variable (for gated models like Llama-2)
