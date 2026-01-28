#include "gram/engine.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct Args {
  std::string index_dir;
  std::string dtype;
  std::uint64_t eos = 0;
  std::uint64_t vocab = 0;
  std::uint64_t version = 4;
  std::size_t threads = 0;
  bool precompute_unigram_logprobs = false;
};

bool ParseBool(std::string_view s) { return s == "1" || s == "true" || s == "True"; }

template <typename UInt>
UInt ParseUInt(std::string_view s) {
  UInt out = 0;
  const char* begin = s.data();
  const char* end = s.data() + s.size();
  auto [ptr, ec] = std::from_chars(begin, end, out);
  if (ec != std::errc{} || ptr != end) {
    throw std::invalid_argument("invalid integer");
  }
  return out;
}

Args ParseArgs(int argc, char** argv) {
  if (argc < 6) {
    throw std::invalid_argument(
        "usage: tg_rpc <index_dir> <dtype:u8|u16|u32> <eos_token_id> <vocab_size> <version> [--threads N] "
        "[--precompute-unigram-logprobs 0|1]");
  }
  Args a;
  a.index_dir = argv[1];
  a.dtype = argv[2];
  a.eos = ParseUInt<std::uint64_t>(argv[3]);
  a.vocab = ParseUInt<std::uint64_t>(argv[4]);
  a.version = ParseUInt<std::uint64_t>(argv[5]);
  for (int i = 6; i < argc; i++) {
    std::string_view key(argv[i]);
    if (key == "--threads") {
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing --threads value");
      }
      a.threads = static_cast<std::size_t>(ParseUInt<std::uint64_t>(argv[++i]));
      continue;
    }
    if (key == "--precompute-unigram-logprobs") {
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing --precompute-unigram-logprobs value");
      }
      a.precompute_unigram_logprobs = ParseBool(argv[++i]);
      continue;
    }
    throw std::invalid_argument("unknown arg");
  }
  return a;
}

void SplitFields(const std::string& line, std::vector<std::string_view>* out) {
  out->clear();
  std::size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r')) {
      i++;
    }
    if (i >= line.size()) {
      break;
    }
    std::size_t j = i;
    while (j < line.size() && line[j] != ' ' && line[j] != '\t' && line[j] != '\r') {
      j++;
    }
    out->emplace_back(line.data() + i, j - i);
    i = j;
  }
}

template <typename Token>
void ParseTokenVec(const std::vector<std::string_view>& f, std::size_t start, std::size_t n, std::vector<Token>* out) {
  out->clear();
  out->reserve(n);
  for (std::size_t i = 0; i < n; i++) {
    out->push_back(static_cast<Token>(ParseUInt<std::uint64_t>(f[start + i])));
  }
}

template <typename Token>
int Run(const Args& args) {
  gram::IndexConfig cfg;
  cfg.index_dirs = {args.index_dir};
  cfg.version = args.version;
  cfg.token_width = sizeof(Token);
  cfg.eos_token_id = args.eos;
  cfg.vocab_size = args.vocab;

  gram::Index index(cfg);
  auto st = index.Load();
  if (!st.ok) {
    std::cerr << st.message << "\n";
    return 2;
  }

  gram::EngineOptions opts;
  opts.thread_count = args.threads;
  opts.precompute_unigram_logprobs = args.precompute_unigram_logprobs;

  gram::Engine<Token> engine(std::move(index), opts);

  std::string line;
  std::vector<std::string_view> fields;
  fields.reserve(256);
  std::vector<Token> ids;
  ids.reserve(256);
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    SplitFields(line, &fields);
    if (fields.empty()) {
      continue;
    }
    const std::string_view op = fields[0];
    if (op == "quit") {
      break;
    }
    if (op == "ping") {
      std::cout << "pong\n";
      continue;
    }
    try {
      if (op == "count") {
        if (fields.size() < 2) {
          throw std::invalid_argument("count needs n");
        }
        const std::size_t n = static_cast<std::size_t>(ParseUInt<std::uint64_t>(fields[1]));
        if (2 + n > fields.size()) {
          throw std::invalid_argument("not enough token ids");
        }
        ParseTokenVec<Token>(fields, 2, n, &ids);
        const auto r = engine.Count(ids);
        std::cout << "count " << r.count << " " << (r.approx ? 1 : 0) << "\n";
        continue;
      }
      if (op == "prob") {
        if (fields.size() < 3) {
          throw std::invalid_argument("prob needs cont_id and n");
        }
        const Token cont = static_cast<Token>(ParseUInt<std::uint64_t>(fields[1]));
        const std::size_t n = static_cast<std::size_t>(ParseUInt<std::uint64_t>(fields[2]));
        if (3 + n > fields.size()) {
          throw std::invalid_argument("not enough token ids");
        }
        ParseTokenVec<Token>(fields, 3, n, &ids);
        const auto r = engine.Prob(ids, cont);
        std::cout << "prob " << r.prompt_cnt << " " << r.cont_cnt << " " << r.prob << "\n";
        continue;
      }
      if (op == "ntd") {
        if (fields.size() < 3) {
          throw std::invalid_argument("ntd needs max_support and n");
        }
        const std::uint64_t max_support = ParseUInt<std::uint64_t>(fields[1]);
        const std::size_t n = static_cast<std::size_t>(ParseUInt<std::uint64_t>(fields[2]));
        if (3 + n > fields.size()) {
          throw std::invalid_argument("not enough token ids");
        }
        ParseTokenVec<Token>(fields, 3, n, &ids);
        const auto r = engine.Ntd(ids, max_support);
        std::cout << "ntd " << r.prompt_cnt << " " << (r.approx ? 1 : 0) << " " << r.result_by_token_id.size() << "\n";
        continue;
      }
      std::cout << "error unknown_op\n";
    } catch (const std::exception& e) {
      std::cout << "error " << e.what() << "\n";
    }
  }

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(&std::cout);
  try {
    const Args args = ParseArgs(argc, argv);
    if (args.dtype == "u8") {
      return Run<gram::u8>(args);
    }
    if (args.dtype == "u16") {
      return Run<gram::u16>(args);
    }
    if (args.dtype == "u32") {
      return Run<gram::u32>(args);
    }
    std::cerr << "unknown dtype\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 2;
  }
}
