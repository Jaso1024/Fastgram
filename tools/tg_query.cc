#include "gram/engine.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename Token>
int Run(const std::string& index_dir, std::uint64_t eos_token_id, std::uint64_t vocab_size, std::uint64_t version, const std::vector<Token>& query) {
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
  gram::Engine<Token> engine(std::move(index), opts);

  auto fr = engine.Find(query);
  std::cout << "count\t" << fr.cnt << "\n";
  for (std::size_t s = 0; s < fr.segment_by_shard.size(); s++) {
    std::cout << "shard\t" << s << "\t" << fr.segment_by_shard[s].first << "\t" << fr.segment_by_shard[s].second << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: tg_query <index_dir> <dtype:u8|u16|u32> <eos_token_id> <vocab_size> <version> [token_ids...]\n";
    return 2;
  }

  const std::string index_dir = argv[1];
  const std::string dtype = argv[2];
  const std::uint64_t eos = std::strtoull(argv[3], nullptr, 10);
  const std::uint64_t vocab = std::strtoull(argv[4], nullptr, 10);
  const std::uint64_t version = std::strtoull(argv[5], nullptr, 10);

  if (dtype == "u8") {
    std::vector<gram::u8> q;
    for (int i = 6; i < argc; i++) q.push_back(static_cast<gram::u8>(std::strtoul(argv[i], nullptr, 10)));
    return Run(index_dir, eos, vocab, version, q);
  }
  if (dtype == "u16") {
    std::vector<gram::u16> q;
    for (int i = 6; i < argc; i++) q.push_back(static_cast<gram::u16>(std::strtoul(argv[i], nullptr, 10)));
    return Run(index_dir, eos, vocab, version, q);
  }
  if (dtype == "u32") {
    std::vector<gram::u32> q;
    for (int i = 6; i < argc; i++) q.push_back(static_cast<gram::u32>(std::strtoul(argv[i], nullptr, 10)));
    return Run(index_dir, eos, vocab, version, q);
  }

  std::cerr << "unknown dtype\n";
  return 2;
}
