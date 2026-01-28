#include "gram/engine.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Query {
  std::string bucket;
  std::size_t len = 0;
  std::vector<std::uint64_t> tokens;
};

static bool StartsWith(const std::string& s, const std::string& pref) { return s.rfind(pref, 0) == 0; }

static std::vector<Query> LoadQueries(const std::string& path, const std::string& bucket, std::size_t length) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open query file");
  }
  std::vector<Query> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || StartsWith(line, "#")) continue;
    std::istringstream iss(line);
    Query q;
    if (!(iss >> q.bucket)) continue;
    if (!(iss >> q.len)) continue;
    std::uint64_t cnt = 0;
    if (!(iss >> cnt)) continue;
    std::uint64_t tok = 0;
    while (iss >> tok) q.tokens.push_back(tok);
    if (q.tokens.size() != q.len) continue;
    if (!bucket.empty() && q.bucket != bucket) continue;
    if (length > 0 && q.len != length) continue;
    out.push_back(std::move(q));
  }
  return out;
}

template <typename Token>
int Run(const std::string& index_dir,
        std::uint64_t eos_token_id,
        std::uint64_t vocab_size,
        std::uint64_t version,
        std::uint64_t max_support,
        const std::string& query_file,
        const std::string& op,
        const std::string& bucket,
        std::size_t length,
        std::size_t iters) {
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

  const auto queries = LoadQueries(query_file, bucket, length);
  if (queries.empty()) {
    std::cerr << "no queries matched\n";
    return 2;
  }

  auto start = std::chrono::steady_clock::now();
  std::uint64_t sink = 0;
  for (std::size_t i = 0; i < iters; i++) {
    for (const auto& q : queries) {
      std::vector<Token> toks;
      toks.reserve(q.tokens.size());
      for (auto t : q.tokens) toks.push_back(static_cast<Token>(t));
      if (op == "find") {
        sink += engine.Find(toks).cnt;
      } else {
        sink += engine.Ntd(toks, max_support).prompt_cnt;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  const std::size_t total_calls = iters * queries.size();
  std::cout << "iters\t" << iters << "\n";
  std::cout << "queries\t" << queries.size() << "\n";
  std::cout << "total_calls\t" << total_calls << "\n";
  std::cout << "total_ns\t" << ns << "\n";
  std::cout << "per_call_ns\t" << (ns / static_cast<double>(total_calls)) << "\n";
  std::cout << "sink\t" << sink << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 9) {
    std::cerr << "usage: tg_bench_suite <index_dir> <dtype:u8|u16|u32> <eos_token_id> <vocab_size> <version> <max_support> <query_file> <op:find|ntd> [iters] [--bucket B] [--length N]\n";
    return 2;
  }
  const std::string index_dir = argv[1];
  const std::string dtype = argv[2];
  const std::uint64_t eos = std::strtoull(argv[3], nullptr, 10);
  const std::uint64_t vocab = std::strtoull(argv[4], nullptr, 10);
  const std::uint64_t version = std::strtoull(argv[5], nullptr, 10);
  const std::uint64_t max_support = std::strtoull(argv[6], nullptr, 10);
  const std::string query_file = argv[7];
  const std::string op = argv[8];
  std::size_t iters = 1;
  std::string bucket;
  std::size_t length = 0;

  for (int i = 9; i < argc; i++) {
    std::string a = argv[i];
    if (StartsWith(a, "--bucket=")) {
      bucket = a.substr(std::string("--bucket=").size());
    } else if (StartsWith(a, "--length=")) {
      length = static_cast<std::size_t>(std::strtoull(a.substr(std::string("--length=").size()).c_str(), nullptr, 10));
    } else if (StartsWith(a, "--")) {
      std::cerr << "unknown flag: " << a << "\n";
      return 2;
    } else {
      char* endptr = nullptr;
      iters = static_cast<std::size_t>(std::strtoull(a.c_str(), &endptr, 10));
      if (endptr == a.c_str() || *endptr != '\0') {
        std::cerr << "invalid iters value: " << a << "\n";
        return 2;
      }
    }
  }

  if (op != "find" && op != "ntd") {
    std::cerr << "unknown op\n";
    return 2;
  }
  if (dtype == "u8") return Run<gram::u8>(index_dir, eos, vocab, version, max_support, query_file, op, bucket, length, iters);
  if (dtype == "u16") return Run<gram::u16>(index_dir, eos, vocab, version, max_support, query_file, op, bucket, length, iters);
  if (dtype == "u32") return Run<gram::u32>(index_dir, eos, vocab, version, max_support, query_file, op, bucket, length, iters);
  std::cerr << "unknown dtype\n";
  return 2;
}
