#include "gram/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Query {
  std::string bucket;
  std::size_t len = 0;
  std::uint64_t cnt = 0;
  std::vector<std::uint64_t> tokens;
};

struct Stats {
  double mean = 0;
  double stddev = 0;
  double min = 0;
  double max = 0;
  double p50 = 0;
  double p95 = 0;
  double p99 = 0;
};

static bool StartsWith(const std::string& s, const std::string& pref) {
  return s.rfind(pref, 0) == 0;
}

static std::vector<Query> LoadQueries(const std::string& path,
                                       const std::string& bucket,
                                       std::size_t min_length,
                                       std::size_t max_length,
                                       std::uint64_t min_cnt) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open query file: " + path);
  }
  std::vector<Query> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || StartsWith(line, "#")) continue;
    std::istringstream iss(line);
    Query q;
    if (!(iss >> q.bucket)) continue;
    if (!(iss >> q.len)) continue;
    if (!(iss >> q.cnt)) continue;
    std::uint64_t tok = 0;
    while (iss >> tok) q.tokens.push_back(tok);
    if (q.tokens.size() != q.len) continue;
    if (!bucket.empty() && q.bucket != bucket) continue;
    if (q.len < min_length || q.len > max_length) continue;
    if (q.cnt < min_cnt) continue;
    out.push_back(std::move(q));
  }
  return out;
}

static Stats ComputeStats(std::vector<double>& samples) {
  if (samples.empty()) return {};
  std::sort(samples.begin(), samples.end());

  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  const double mean = sum / static_cast<double>(samples.size());

  double sq_sum = 0;
  for (const auto v : samples) {
    sq_sum += (v - mean) * (v - mean);
  }
  const double stddev = std::sqrt(sq_sum / static_cast<double>(samples.size()));

  auto percentile = [&](double p) -> double {
    const double idx = p * static_cast<double>(samples.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, samples.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return samples[lo] * (1.0 - frac) + samples[hi] * frac;
  };

  return Stats{
    .mean = mean,
    .stddev = stddev,
    .min = samples.front(),
    .max = samples.back(),
    .p50 = percentile(0.50),
    .p95 = percentile(0.95),
    .p99 = percentile(0.99),
  };
}

static void PrintStats(const std::string& name, const Stats& s) {
  std::cout << std::fixed << std::setprecision(1);
  std::cout << name << "\t"
            << "mean=" << s.mean << "ns\t"
            << "stddev=" << s.stddev << "\t"
            << "min=" << s.min << "\t"
            << "p50=" << s.p50 << "\t"
            << "p95=" << s.p95 << "\t"
            << "p99=" << s.p99 << "\t"
            << "max=" << s.max << "\n";
}

template <typename Token>
class DraftBenchmark {
 public:
  DraftBenchmark(gram::Engine<Token>* engine, std::uint64_t seed)
      : engine_(engine), rng_(seed) {}

  void BenchSampleOne(const std::vector<Query>& queries,
                      std::size_t samples_per_query,
                      std::size_t warmup_iters) {
    std::cout << "\n=== SampleOne Benchmark ===\n";
    std::cout << "queries: " << queries.size() << "\n";
    std::cout << "samples_per_query: " << samples_per_query << "\n";
    std::cout << "warmup_iters: " << warmup_iters << "\n\n";

    std::vector<double> all_times;
    all_times.reserve(queries.size() * samples_per_query);

    for (const auto& q : queries) {
      std::vector<Token> toks;
      toks.reserve(q.tokens.size());
      for (auto t : q.tokens) toks.push_back(static_cast<Token>(t));

      auto cursor = engine_->MakeCursor();
      for (const auto& tok : toks) {
        cursor.Advance(tok);
      }
      if (cursor.cnt() == 0) continue;

      // Warmup
      Token sink = 0;
      for (std::size_t i = 0; i < warmup_iters; i++) {
        sink += cursor.SampleOne(rng_);
      }
      volatile Token v = sink;
      (void)v;

      // Timed runs
      for (std::size_t i = 0; i < samples_per_query; i++) {
        auto start = std::chrono::steady_clock::now();
        Token tok = cursor.SampleOne(rng_);
        auto end = std::chrono::steady_clock::now();
        volatile Token s = tok;
        (void)s;
        const double ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        all_times.push_back(ns);
      }
    }

    auto stats = ComputeStats(all_times);
    PrintStats("SampleOne", stats);
  }

  void BenchGenerateDraft(const std::vector<Query>& queries,
                          const std::vector<std::size_t>& draft_lengths,
                          std::size_t samples_per_query,
                          std::size_t warmup_iters) {
    std::cout << "\n=== GenerateDraft Benchmark ===\n";
    std::cout << "queries: " << queries.size() << "\n";
    std::cout << "samples_per_query: " << samples_per_query << "\n";
    std::cout << "warmup_iters: " << warmup_iters << "\n";
    std::cout << "draft_lengths: ";
    for (auto n : draft_lengths) std::cout << n << " ";
    std::cout << "\n\n";

    for (const auto draft_len : draft_lengths) {
      std::vector<double> all_times;
      std::vector<double> per_token_times;
      all_times.reserve(queries.size() * samples_per_query);
      per_token_times.reserve(queries.size() * samples_per_query);

      for (const auto& q : queries) {
        std::vector<Token> toks;
        toks.reserve(q.tokens.size());
        for (auto t : q.tokens) toks.push_back(static_cast<Token>(t));

        // Warmup
        std::size_t sink = 0;
        for (std::size_t i = 0; i < warmup_iters; i++) {
          auto cursor = engine_->MakeCursor();
          for (const auto& tok : toks) {
            cursor.Advance(tok);
          }
          if (cursor.cnt() == 0) break;
          sink += cursor.GenerateDraft(draft_len, rng_).size();
        }
        volatile std::size_t v = sink;
        (void)v;

        // Timed runs
        for (std::size_t i = 0; i < samples_per_query; i++) {
          auto cursor = engine_->MakeCursor();
          for (const auto& tok : toks) {
            cursor.Advance(tok);
          }
          if (cursor.cnt() == 0) continue;

          auto start = std::chrono::steady_clock::now();
          auto draft = cursor.GenerateDraft(draft_len, rng_);
          auto end = std::chrono::steady_clock::now();
          volatile std::size_t s = draft.size();
          (void)s;

          const double ns = static_cast<double>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
          all_times.push_back(ns);
          if (!draft.empty()) {
            per_token_times.push_back(ns / static_cast<double>(draft.size()));
          }
        }
      }

      auto stats = ComputeStats(all_times);
      auto per_tok_stats = ComputeStats(per_token_times);
      std::cout << "draft_len=" << draft_len << ":\n";
      PrintStats("  total", stats);
      PrintStats("  per_token", per_tok_stats);
    }
  }

  void BenchNtdBaseline(const std::vector<Query>& queries,
                        const std::vector<std::size_t>& draft_lengths,
                        std::size_t samples_per_query,
                        std::size_t warmup_iters,
                        std::uint64_t max_support) {
    std::cout << "\n=== NTD Baseline (for comparison) ===\n";
    std::cout << "queries: " << queries.size() << "\n";
    std::cout << "samples_per_query: " << samples_per_query << "\n";
    std::cout << "max_support: " << max_support << "\n\n";

    for (const auto draft_len : draft_lengths) {
      std::vector<double> all_times;
      std::vector<double> per_token_times;
      all_times.reserve(queries.size() * samples_per_query);
      per_token_times.reserve(queries.size() * samples_per_query);

      for (const auto& q : queries) {
        std::vector<Token> toks;
        toks.reserve(q.tokens.size());
        for (auto t : q.tokens) toks.push_back(static_cast<Token>(t));

        // Warmup
        std::size_t sink = 0;
        for (std::size_t i = 0; i < warmup_iters; i++) {
          auto cursor = engine_->MakeCursor();
          for (const auto& tok : toks) {
            cursor.Advance(tok);
          }
          if (cursor.cnt() == 0) break;
          for (std::size_t j = 0; j < draft_len && cursor.cnt() > 0; j++) {
            auto ntd = cursor.Ntd(max_support);
            if (ntd.tokens.empty()) break;
            sink += ntd.tokens.size();
            // Sample from distribution
            std::uniform_int_distribution<std::size_t> dist(0, ntd.tokens.size() - 1);
            cursor.Advance(ntd.tokens[dist(rng_)]);
          }
        }
        volatile std::size_t v = sink;
        (void)v;

        // Timed runs
        for (std::size_t i = 0; i < samples_per_query; i++) {
          auto cursor = engine_->MakeCursor();
          for (const auto& tok : toks) {
            cursor.Advance(tok);
          }
          if (cursor.cnt() == 0) continue;

          auto start = std::chrono::steady_clock::now();
          std::size_t generated = 0;
          for (std::size_t j = 0; j < draft_len && cursor.cnt() > 0; j++) {
            auto ntd = cursor.Ntd(max_support);
            if (ntd.tokens.empty()) break;
            std::uniform_int_distribution<std::size_t> dist(0, ntd.tokens.size() - 1);
            cursor.Advance(ntd.tokens[dist(rng_)]);
            generated++;
          }
          auto end = std::chrono::steady_clock::now();

          const double ns = static_cast<double>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
          all_times.push_back(ns);
          if (generated > 0) {
            per_token_times.push_back(ns / static_cast<double>(generated));
          }
        }
      }

      auto stats = ComputeStats(all_times);
      auto per_tok_stats = ComputeStats(per_token_times);
      std::cout << "draft_len=" << draft_len << ":\n";
      PrintStats("  total", stats);
      PrintStats("  per_token", per_tok_stats);
    }
  }

  void BenchByPromptLength(const std::vector<Query>& queries,
                           std::size_t draft_len,
                           std::size_t samples_per_query) {
    std::cout << "\n=== GenerateDraft by Prompt Length (n=" << draft_len << ") ===\n";

    std::map<std::size_t, std::vector<double>> times_by_len;
    for (const auto& q : queries) {
      std::vector<Token> toks;
      toks.reserve(q.tokens.size());
      for (auto t : q.tokens) toks.push_back(static_cast<Token>(t));

      for (std::size_t i = 0; i < samples_per_query; i++) {
        auto cursor = engine_->MakeCursor();
        for (const auto& tok : toks) {
          cursor.Advance(tok);
        }
        if (cursor.cnt() == 0) continue;

        auto start = std::chrono::steady_clock::now();
        auto draft = cursor.GenerateDraft(draft_len, rng_);
        auto end = std::chrono::steady_clock::now();
        volatile std::size_t s = draft.size();
        (void)s;

        const double ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        times_by_len[q.len].push_back(ns);
      }
    }

    std::cout << "prompt_len\tmean_ns\t\tstddev\t\tsamples\n";
    for (auto& [len, times] : times_by_len) {
      auto stats = ComputeStats(times);
      std::cout << len << "\t\t" << std::fixed << std::setprecision(1)
                << stats.mean << "\t\t" << stats.stddev << "\t\t" << times.size() << "\n";
    }
  }

  void BenchByCountBucket(const std::vector<Query>& queries,
                          std::size_t draft_len,
                          std::size_t samples_per_query) {
    std::cout << "\n=== GenerateDraft by Count Bucket (n=" << draft_len << ") ===\n";

    std::map<std::string, std::vector<double>> times_by_bucket;
    for (const auto& q : queries) {
      std::vector<Token> toks;
      toks.reserve(q.tokens.size());
      for (auto t : q.tokens) toks.push_back(static_cast<Token>(t));

      for (std::size_t i = 0; i < samples_per_query; i++) {
        auto cursor = engine_->MakeCursor();
        for (const auto& tok : toks) {
          cursor.Advance(tok);
        }
        if (cursor.cnt() == 0) continue;

        auto start = std::chrono::steady_clock::now();
        auto draft = cursor.GenerateDraft(draft_len, rng_);
        auto end = std::chrono::steady_clock::now();
        volatile std::size_t s = draft.size();
        (void)s;

        const double ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        times_by_bucket[q.bucket].push_back(ns);
      }
    }

    std::cout << "bucket\t\tmean_ns\t\tstddev\t\tsamples\n";
    const std::vector<std::string> bucket_order = {
        "ultra_rare", "rare", "low", "mid", "high", "very_high"};
    for (const auto& bucket : bucket_order) {
      auto it = times_by_bucket.find(bucket);
      if (it == times_by_bucket.end()) continue;
      auto stats = ComputeStats(it->second);
      std::cout << bucket << "\t\t" << std::fixed << std::setprecision(1)
                << stats.mean << "\t\t" << stats.stddev << "\t\t" << it->second.size() << "\n";
    }
  }

 private:
  gram::Engine<Token>* engine_;
  std::mt19937_64 rng_;
};

template <typename Token>
int Run(const std::string& index_dir,
        std::uint64_t eos_token_id,
        std::uint64_t vocab_size,
        std::uint64_t version,
        const std::string& query_file,
        std::size_t samples_per_query,
        std::size_t warmup_iters,
        std::uint64_t max_support,
        std::uint64_t seed) {
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

  // Load queries with cnt > 0 (need matches to sample from)
  const auto queries = LoadQueries(query_file, "", 1, 16, 1);
  if (queries.empty()) {
    std::cerr << "no queries found\n";
    return 2;
  }

  std::cout << "========================================\n";
  std::cout << "  Draft Generation Benchmark Suite\n";
  std::cout << "========================================\n";
  std::cout << "index: " << index_dir << "\n";
  std::cout << "dtype: u" << (sizeof(Token) * 8) << "\n";
  std::cout << "total_queries: " << queries.size() << "\n";
  std::cout << "samples_per_query: " << samples_per_query << "\n";
  std::cout << "warmup_iters: " << warmup_iters << "\n";
  std::cout << "seed: " << seed << "\n";

  DraftBenchmark<Token> bench(&engine, seed);

  // 1. SampleOne benchmark
  bench.BenchSampleOne(queries, samples_per_query, warmup_iters);

  // 2. GenerateDraft with various draft lengths
  const std::vector<std::size_t> draft_lengths = {1, 5, 10, 20, 50};
  bench.BenchGenerateDraft(queries, draft_lengths, samples_per_query, warmup_iters);

  // 3. NTD baseline for comparison
  bench.BenchNtdBaseline(queries, {5, 10}, samples_per_query / 10, warmup_iters / 10, max_support);

  // 4. Breakdown by prompt length
  bench.BenchByPromptLength(queries, 10, samples_per_query);

  // 5. Breakdown by count bucket
  bench.BenchByCountBucket(queries, 10, samples_per_query);

  std::cout << "\n========================================\n";
  std::cout << "  Benchmark Complete\n";
  std::cout << "========================================\n";

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) {
    std::cerr << "usage: tg_bench_draft <index_dir> <dtype:u8|u16|u32> <eos_token_id> "
              << "<vocab_size> <version> <query_file> [samples_per_query] [warmup_iters] "
              << "[max_support] [seed]\n";
    std::cerr << "\nExample:\n";
    std::cerr << "  ./tg_bench_draft index/v4_pileval_gpt2 u16 50256 50257 4 "
              << "bench/queries_v4_pileval_gpt2.txt 1000 100 1000 42\n";
    return 2;
  }

  const std::string index_dir = argv[1];
  const std::string dtype = argv[2];
  const std::uint64_t eos = std::strtoull(argv[3], nullptr, 10);
  const std::uint64_t vocab = std::strtoull(argv[4], nullptr, 10);
  const std::uint64_t version = std::strtoull(argv[5], nullptr, 10);
  const std::string query_file = argv[6];
  const std::size_t samples_per_query = (argc >= 8) ?
      static_cast<std::size_t>(std::strtoull(argv[7], nullptr, 10)) : 1000;
  const std::size_t warmup_iters = (argc >= 9) ?
      static_cast<std::size_t>(std::strtoull(argv[8], nullptr, 10)) : 100;
  const std::uint64_t max_support = (argc >= 10) ?
      std::strtoull(argv[9], nullptr, 10) : 1000;
  const std::uint64_t seed = (argc >= 11) ?
      std::strtoull(argv[10], nullptr, 10) : 42;

  if (dtype == "u8") return Run<gram::u8>(index_dir, eos, vocab, version, query_file,
                                          samples_per_query, warmup_iters, max_support, seed);
  if (dtype == "u16") return Run<gram::u16>(index_dir, eos, vocab, version, query_file,
                                            samples_per_query, warmup_iters, max_support, seed);
  if (dtype == "u32") return Run<gram::u32>(index_dir, eos, vocab, version, query_file,
                                            samples_per_query, warmup_iters, max_support, seed);
  std::cerr << "unknown dtype\n";
  return 2;
}
