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

`gram list`

`gram download v4_pileval_gpt2 --to index/v4_pileval_gpt2`
