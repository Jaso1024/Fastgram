#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include <unordered_map>

#include "gram/common.h"
#include "gram/index.h"
#include "gram/thread_pool.h"

namespace gram {

struct FindResult {
  u64 cnt = 0;
  std::vector<std::pair<u64, u64>> segment_by_shard;
};

struct FindCnfResult {
  u64 cnt = 0;
  bool approx = false;
  std::vector<std::vector<u64>> ptrs_by_shard;
};

struct CountResult {
  u64 count = 0;
  bool approx = false;
};

struct ProbResult {
  u64 prompt_cnt = 0;
  u64 cont_cnt = 0;
  double prob = -1.0;
};

struct DistTokenResult {
  u64 cont_cnt = 0;
  double prob = 0.0;
};

template <typename Token>
struct DistResult {
  u64 prompt_cnt = 0;
  std::map<Token, DistTokenResult> result_by_token_id;
  bool approx = false;
};

struct InfgramProbResult {
  u64 prompt_cnt = 0;
  u64 cont_cnt = 0;
  double prob = -1.0;
  u64 suffix_len = 0;
};

template <typename Token>
struct InfgramDistResult {
  u64 prompt_cnt = 0;
  std::map<Token, DistTokenResult> result_by_token_id;
  bool approx = false;
  u64 suffix_len = 0;
};

template <typename Token>
struct DocResult {
  u64 doc_ix = 0;
  u64 doc_len = 0;
  u64 disp_len = 0;
  u64 needle_offset = 0;
  std::string metadata;
  std::vector<Token> token_ids;
  bool blocked = false;
};

template <typename Token>
struct SearchDocsResult {
  u64 cnt = 0;
  bool approx = false;
  std::vector<u64> idxs;
  std::vector<DocResult<Token>> docs;
};

struct CreativityResult {
  std::vector<std::size_t> rs;
};

struct AttributionDoc {
  std::size_t s = 0;
  u64 ptr = 0;
};

struct AttributionSpan {
  std::size_t l = 0;
  std::size_t r = 0;
  std::size_t length = 0;
  u64 count = 0;
  double unigram_logprob_sum = 0.0;
  std::vector<AttributionDoc> docs;
};

struct AttributionResult {
  std::vector<AttributionSpan> spans;
};

template <typename Token>
class Engine;

template <typename Token>
class Cursor {
 public:
  explicit Cursor(const Engine<Token>* engine);
  void Reset();
  [[nodiscard]] u64 cnt() const { return cnt_; }
  [[nodiscard]] std::size_t num_bytes() const { return num_bytes_; }

  DistResult<Token> Ntd(u64 max_support) const;
  FindResult Advance(Token next_token_id);
  DistResult<Token> AdvanceNtd(Token next_token_id, u64 max_support);

 private:
  const Engine<Token>* engine_ = nullptr;
  std::vector<std::pair<u64, u64>> segment_by_shard_;
  u64 cnt_ = 0;
  std::size_t num_bytes_ = 0;
};

struct EngineOptions {
  std::size_t thread_count = 0;
  std::size_t ds_prefetch_depth = 0;
  std::size_t sa_prefetch_depth = 0;
  std::size_t od_prefetch_depth = 0;
  std::set<u32> bow_ids;
  std::size_t attribution_block_size = 512;
  bool precompute_unigram_logprobs = false;
};

template <typename Token>
class Engine {
 public:
  Engine(Index index, EngineOptions opts);
  virtual ~Engine() = default;

  [[nodiscard]] const Index& index() const { return index_; }
  [[nodiscard]] std::size_t num_shards() const { return index_.num_shards(); }
  [[nodiscard]] u64 tok_cnt(std::size_t s) const;
  [[nodiscard]] u64 ds_size(std::size_t s) const;
  [[nodiscard]] u64 total_tok_cnt() const;

  Cursor<Token> MakeCursor() const { return Cursor<Token>(this); }

  FindResult Find(const std::vector<Token>& input_ids) const;
  FindResult FindWithHint(const std::vector<Token>& input_ids,
                          const std::vector<std::pair<u64, u64>>& hint_segment_by_shard) const;
  CountResult Count(const std::vector<Token>& input_ids) const;
  ProbResult PrimitiveProb(const std::vector<Token>& prompt_ids, Token cont_id) const;
  DistResult<Token> PrimitiveNtd(const std::vector<Token>& prompt_ids, u64 max_support) const;
  DistResult<Token> NtdFromSegment(std::size_t num_bytes,
                                   const std::vector<std::pair<u64, u64>>& segment_by_shard,
                                   u64 max_support) const;
  InfgramProbResult Prob(const std::vector<Token>& prompt_ids, Token cont_id) const;
  InfgramDistResult<Token> Ntd(const std::vector<Token>& prompt_ids, u64 max_support) const;

  FindCnfResult FindCnf(const std::vector<std::vector<std::vector<Token>>>& cnf,
                        u64 max_clause_freq,
                        u64 max_diff_tokens) const;
  CountResult CountCnf(const std::vector<std::vector<std::vector<Token>>>& cnf,
                       u64 max_clause_freq,
                       u64 max_diff_tokens) const;

  SearchDocsResult<Token> SearchDocs(const std::vector<Token>& input_ids, std::size_t maxnum, u64 max_disp_len) const;
  SearchDocsResult<Token> SearchDocsCnf(const std::vector<std::vector<std::vector<Token>>>& cnf,
                                        std::size_t maxnum,
                                        u64 max_disp_len,
                                        u64 max_clause_freq,
                                        u64 max_diff_tokens) const;

  DocResult<Token> GetDocByRank(std::size_t s, u64 rank, u64 max_disp_len) const;
  std::vector<DocResult<Token>> GetDocsByRanks(const std::vector<std::pair<std::size_t, u64>>& list_of_s_and_rank,
                                               u64 max_disp_len) const;
  DocResult<Token> GetDocByPtr(std::size_t s, u64 ptr, u64 max_disp_len) const;
  std::vector<DocResult<Token>> GetDocsByPtrs(const std::vector<std::pair<std::size_t, u64>>& list_of_s_and_ptr,
                                              u64 max_disp_len) const;
  DocResult<Token> GetDocByIx(u64 doc_ix, u64 max_disp_len) const;
  std::vector<DocResult<Token>> GetDocsByIxs(const std::vector<u64>& list_of_doc_ix, u64 max_disp_len) const;

  DocResult<Token> GetDocByRank2(std::size_t s, u64 rank, u64 needle_len, u64 max_ctx_len) const;
  std::vector<DocResult<Token>> GetDocsByRanks2(const std::vector<std::tuple<std::size_t, u64, u64, u64>>& requests) const;
  DocResult<Token> GetDocByPtr2(std::size_t s, u64 ptr, u64 needle_len, u64 max_ctx_len) const;
  std::vector<DocResult<Token>> GetDocsByPtrs2(const std::vector<std::tuple<std::size_t, u64, u64, u64>>& requests) const;
  DocResult<Token> GetDocByIx2(u64 doc_ix, u64 max_ctx_len) const;
  std::vector<DocResult<Token>> GetDocsByIxs2(const std::vector<std::tuple<u64, u64>>& requests) const;

  u64 TotalDocCnt() const;
  std::map<Token, u64> ComputeUnigramCounts(std::size_t s) const;

  CreativityResult Creativity(const std::vector<Token>& input_ids) const;
  AttributionResult Attribute(const std::vector<Token>& input_ids,
                              const std::vector<Token>& delim_ids,
                              std::size_t min_len,
                              std::size_t max_cnt,
                              bool enforce_bow) const;

 private:
  friend class Cursor<Token>;

  static void CheckLittleEndian();

  [[nodiscard]] const Shard& ShardAt(std::size_t s) const { return index_.shards()[s]; }

  [[nodiscard]] Token DocSepId() const;
  [[nodiscard]] std::size_t TokenWidth() const { return index_.cfg().token_width; }
  [[nodiscard]] u64 Version() const { return index_.cfg().version; }
  [[nodiscard]] Token EosTokenId() const { return static_cast<Token>(index_.cfg().eos_token_id); }

  [[nodiscard]] u64 ConvertRankToPtr(const Shard& shard, u64 rank) const;
  [[nodiscard]] Token ConvertPtrToTokenId(const Shard& shard, u64 ptr) const;
  [[nodiscard]] Token ConvertPtrToRawTokenId(const Shard& shard, u64 ptr) const;

  [[nodiscard]] int ComparePrefix(const Shard& shard, u64 ptr, const u8* input_buf, u64 num_bytes) const;
  void FindThread(std::size_t s,
                  const u8* input_buf,
                  u64 num_bytes,
                  std::pair<u64, u64> hint_segment,
                  std::pair<u64, u64>* out_segment) const;
  FindResult FindBytes(const u8* input_buf,
                       u64 num_bytes,
                       const std::vector<std::pair<u64, u64>>& hint_segment_by_shard) const;

  [[nodiscard]] std::pair<u64, u64> RefineSegmentByNextToken(const Shard& shard,
                                                             u64 num_bytes,
                                                             std::pair<u64, u64> segment,
                                                             Token next_token_id) const;
  FindResult AdvanceSegmentByToken(const std::vector<std::pair<u64, u64>>& segment_by_shard,
                                   std::size_t num_bytes,
                                   Token next_token_id) const;

  void GetFreqByTokenIdApprox(std::size_t s,
                              u64 num_bytes,
                              std::pair<u64, u64> segment,
                              u64 unit,
                              const Token* token_start,
                              const Token* token_end,
                              std::unordered_map<Token, u64>* out_freq_by_token_id) const;

  [[nodiscard]] u64 ConvertDocIxToPtr(const Shard& shard, u64 doc_ix) const;
  [[nodiscard]] u64 ConvertDocIxToMetaPtr(const Shard& shard, u64 doc_ix) const;
  [[nodiscard]] std::pair<u64, u64> BinSearch(const std::vector<u64>& arr, u64 val) const;

  std::size_t GetLcpLen(const u8* a, std::size_t len_a, const u8* b, std::size_t len_b) const;
  void ComputeLongestPrefixLenThread(const std::vector<Token>* input_ids,
                                     std::size_t s,
                                     std::size_t* out_longest_prefix_len) const;
 protected:
  virtual std::size_t ComputeLongestPrefixLen(const std::vector<Token>& input_ids,
                                              const std::vector<Token>& delim_ids,
                                              bool enforce_bow) const;

 private:
  void CreativityThread(const std::vector<Token>* input_ids, std::size_t l, std::size_t* out_r) const;
  void ComputeInterestingSpansThread(const std::vector<Token>* input_ids,
                                     std::size_t l,
                                     const std::vector<Token>* delim_ids,
                                     std::size_t min_len,
                                     std::size_t max_cnt,
                                     bool enforce_bow,
                                     std::vector<std::pair<std::pair<std::size_t, std::size_t>, FindResult>>* out_span_find_pairs) const;
  std::vector<std::pair<std::pair<std::size_t, std::size_t>, FindResult>> ComputeInterestingSpans(
      const std::vector<Token>& input_ids,
      const std::vector<Token>& delim_ids,
      std::size_t min_len,
      std::size_t max_cnt,
      bool enforce_bow) const;
  void GetAttributionSpanThread(
      const std::pair<std::pair<std::size_t, std::size_t>, FindResult>* span_find_pair,
      AttributionSpan* out_attribution_span) const;

  struct FindDisjResult {
    u64 cnt = 0;
    std::vector<u64> cnt_by_shard;
    std::vector<std::vector<std::pair<u64, u64>>> segment_by_term_by_shard;
    std::vector<double> subsampling_factor_by_shard;
  };

  FindDisjResult FindDisj(const std::vector<std::vector<Token>>& disj_clause, u64 max_clause_freq) const;
  void FindDisjThread(std::size_t s,
                      const std::vector<FindResult>* find_result_by_term,
                      u64 max_clause_freq_per_shard,
                      u64* out_cnt,
                      std::vector<std::pair<u64, u64>>* out_segment_by_term,
                      double* out_subsampling_factor) const;
  void FindCnfThread(std::size_t s,
                     const std::vector<FindDisjResult>* find_disj_result_by_clause,
                     u64 max_diff_tokens,
                     u64* out_cnt,
                     std::vector<std::pair<u64, u64>>* out_valid_ptr_ranges,
                     double* out_subsampling_factor) const;

  Index index_;
  EngineOptions opts_;
  mutable std::optional<ThreadPool> pool_;

  std::map<Token, double> unigram_logprobs_;
};

template <typename Token>
class EngineDiff : public Engine<Token> {
 public:
  EngineDiff(Index main_index, Index diff_index, EngineOptions opts);

  std::vector<std::vector<DocResult<Token>>> GetDocsByPtrs2Grouped(
      const std::vector<std::tuple<std::vector<std::pair<std::size_t, u64>>, std::vector<Token>, u64, u64>>& requests) const;

 private:
  std::size_t ComputeLongestPrefixLen(const std::vector<Token>& input_ids,
                                      const std::vector<Token>& delim_ids,
                                      bool enforce_bow) const override;

  std::unique_ptr<Engine<Token>> diff_;
};

}  // namespace gram
