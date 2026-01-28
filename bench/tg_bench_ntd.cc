#include "gram/engine.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename Token>
int Run(const std::string& index_dir,
        std::uint64_t eos_token_id,
        std::uint64_t vocab_size,
        std::uint64_t version,
        std::size_t iters,
        std::uint64_t max_support) {
  gram::IndexConfig cfg;
  cfg.index_dirs = {index_dir};
  cfg.version = version;
  cfg.token_width = sizeof(Token);
  cfg.eos_token_id = eos_token_id;
  cfg.vocab_size = vocab_size;

  gram::Index index(cfg);
  auto st = index.Load();
  if (!st.ok) {
    std::cerr << st.message << "\n";
    return 2;
  }

  gram::EngineOptions opts;
  opts.thread_count = 1;
  gram::Engine<Token> engine(std::move(index), opts);

  std::vector<Token> q = {static_cast<Token>(3158), static_cast<Token>(8516)};

  auto start = std::chrono::steady_clock::now();
  std::uint64_t sink = 0;
  for (std::size_t i = 0; i < iters; i++) {
    sink += engine.Ntd(q, max_support).prompt_cnt;
  }
  auto end = std::chrono::steady_clock::now();
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "iters\t" << iters << "\n";
  std::cout << "total_ns\t" << ns << "\n";
  std::cout << "per_call_ns\t" << (ns / static_cast<double>(iters)) << "\n";
  std::cout << "sink\t" << sink << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) {
    std::cerr << "usage: tg_bench_ntd <index_dir> <dtype:u8|u16|u32> <eos_token_id> <vocab_size> <version> <max_support> [iters]\n";
    return 2;
  }
  const std::string index_dir = argv[1];
  const std::string dtype = argv[2];
  const std::uint64_t eos = std::strtoull(argv[3], nullptr, 10);
  const std::uint64_t vocab = std::strtoull(argv[4], nullptr, 10);
  const std::uint64_t version = std::strtoull(argv[5], nullptr, 10);
  const std::uint64_t max_support = std::strtoull(argv[6], nullptr, 10);
  const std::size_t iters = (argc >= 8) ? static_cast<std::size_t>(std::strtoull(argv[7], nullptr, 10)) : 10000;

  if (dtype == "u8") return Run<gram::u8>(index_dir, eos, vocab, version, iters, max_support);
  if (dtype == "u16") return Run<gram::u16>(index_dir, eos, vocab, version, iters, max_support);
  if (dtype == "u32") return Run<gram::u32>(index_dir, eos, vocab, version, iters, max_support);
  std::cerr << "unknown dtype\n";
  return 2;
}
