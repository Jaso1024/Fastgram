#include "gram/engine.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace gram {

Index::Index(IndexConfig cfg) : cfg_(std::move(cfg)) {}

Status Index::LoadIndexDir(const std::filesystem::path& dir) {
  if (!std::filesystem::exists(dir)) {
    return Status::Error("index dir does not exist: " + dir.string());
  }

  std::vector<std::filesystem::path> ds_paths;
  std::vector<std::filesystem::path> sa_paths;
  std::vector<std::filesystem::path> od_paths;
  std::vector<std::filesystem::path> mt_paths;
  std::vector<std::filesystem::path> om_paths;
  std::vector<std::filesystem::path> ug_paths;

  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    const auto p = entry.path();
    const auto name = p.filename().string();
    if (name.find("tokenized") != std::string::npos) {
      ds_paths.push_back(p);
    } else if (name.find("table") != std::string::npos) {
      sa_paths.push_back(p);
    } else if (name.find("offset") != std::string::npos) {
      od_paths.push_back(p);
    } else if (name.find("metadata") != std::string::npos) {
      mt_paths.push_back(p);
    } else if (name.find("metaoff") != std::string::npos) {
      om_paths.push_back(p);
    } else if (name.find("unigram") != std::string::npos) {
      ug_paths.push_back(p);
    }
  }

  std::sort(ds_paths.begin(), ds_paths.end());
  std::sort(sa_paths.begin(), sa_paths.end());
  std::sort(od_paths.begin(), od_paths.end());
  std::sort(mt_paths.begin(), mt_paths.end());
  std::sort(om_paths.begin(), om_paths.end());
  std::sort(ug_paths.begin(), ug_paths.end());

  if (ds_paths.empty()) {
    return Status::Error("no shards found in: " + dir.string());
  }
  if (sa_paths.size() != ds_paths.size() || od_paths.size() != ds_paths.size()) {
    return Status::Error("mismatched shard file counts in: " + dir.string());
  }
  if (!mt_paths.empty() && mt_paths.size() != ds_paths.size()) {
    return Status::Error("mismatched metadata shard file counts in: " + dir.string());
  }
  if (om_paths.size() != mt_paths.size()) {
    return Status::Error("mismatched metaoff shard file counts in: " + dir.string());
  }
  if (!ug_paths.empty() && ug_paths.size() != ds_paths.size()) {
    return Status::Error("mismatched unigram shard file counts in: " + dir.string());
  }

  for (std::size_t s = 0; s < ds_paths.size(); s++) {
    Shard shard;
    shard.ds = MmapFile(ds_paths[s].string());
    shard.sa = MmapFile(sa_paths[s].string());
    shard.od = MmapFile(od_paths[s].string());
    if (!mt_paths.empty()) {
      shard.mt = MmapFile(mt_paths[s].string());
      shard.om = MmapFile(om_paths[s].string());
    }
    if (!ug_paths.empty()) {
      shard.unigram = MmapFile(ug_paths[s].string());
    }

    shard.ds_size = shard.ds.size();
    if (cfg_.token_width == 0 || shard.ds_size % cfg_.token_width != 0) {
      return Status::Error("tokenized shard size not divisible by token_width");
    }
    shard.tok_cnt = shard.ds_size / cfg_.token_width;

    if (shard.sa.size() % shard.tok_cnt != 0) {
      return Status::Error("table shard size not divisible by tok_cnt");
    }
    shard.ptr_size = static_cast<u8>(shard.sa.size() / shard.tok_cnt);
    if (shard.ptr_size == 0 || shard.ptr_size > 8) {
      return Status::Error("unsupported ptr_size");
    }

    if (shard.od.size() % sizeof(u64) != 0) {
      return Status::Error("offset shard size not divisible by 8");
    }
    shard.doc_cnt = shard.od.size() / sizeof(u64);

    if (shard.mt.has_value()) {
      if (!shard.om.has_value()) {
        return Status::Error("metadata present without metaoff");
      }
      if (shard.om->size() != shard.doc_cnt * sizeof(u64)) {
        return Status::Error("metaoff size mismatch");
      }
    }

    shards_.push_back(std::move(shard));
  }

  return Status::Ok();
}

Status Index::Load() {
  shards_.clear();
  unigram_ranges_.reset();
  for (const auto& d : cfg_.index_dirs) {
    auto st = LoadIndexDir(d);
    if (!st.ok) {
      return st;
    }
  }

  if (cfg_.token_width <= 2 && cfg_.index_dirs.size() == 1 && !shards_.empty()) {
    const std::filesystem::path path = std::filesystem::path(cfg_.index_dirs[0]) / "unigram_ranges.bin";
    if (std::filesystem::exists(path)) {
      const u64 token_space = static_cast<u64>(1) << (8 * cfg_.token_width);
      const u64 expected_size = token_space * shards_.size() * 2 * sizeof(u64);
      try {
        MmapFile f(path.string());
        if (f.size() == expected_size) {
          unigram_ranges_ = std::move(f);
        }
      } catch (const std::exception&) {
      }
    }
  }

  return Status::Ok();
}

template <typename Token>
Cursor<Token>::Cursor(const Engine<Token>* engine) : engine_(engine) {
  if (!engine_) {
    throw std::invalid_argument("engine is null");
  }
  Reset();
}

template <typename Token>
void Cursor<Token>::Reset() {
  segment_by_shard_.clear();
  segment_by_shard_.reserve(engine_->num_shards());
  cnt_ = 0;
  for (const auto& shard : engine_->index().shards()) {
    segment_by_shard_.push_back({0, shard.tok_cnt});
    cnt_ += shard.tok_cnt;
  }
  num_bytes_ = 0;
}

template <typename Token>
DistResult<Token> Cursor<Token>::Ntd(u64 max_support) const {
  return engine_->NtdFromSegment(num_bytes_, segment_by_shard_, max_support);
}

template <typename Token>
FindResult Cursor<Token>::Advance(Token next_token_id) {
  auto r = engine_->AdvanceSegmentByToken(segment_by_shard_, num_bytes_, next_token_id);
  segment_by_shard_ = r.segment_by_shard;
  cnt_ = r.cnt;
  num_bytes_ += sizeof(Token);
  return r;
}

template <typename Token>
DistResult<Token> Cursor<Token>::AdvanceNtd(Token next_token_id, u64 max_support) {
  Advance(next_token_id);
  return engine_->NtdFromSegment(num_bytes_, segment_by_shard_, max_support);
}

template <typename Token>
Engine<Token>::Engine(Index index, EngineOptions opts) : index_(std::move(index)), opts_(std::move(opts)) {
  CheckLittleEndian();
  if (index_.cfg().token_width != sizeof(Token)) {
    throw std::invalid_argument("token_width does not match Token size");
  }
  if (opts_.thread_count == 0) {
    const std::size_t hw = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t shard_limit = std::max<std::size_t>(1, num_shards());
    opts_.thread_count = std::min(hw, shard_limit);
  }
  if (opts_.thread_count > 1) {
    pool_.emplace(opts_.thread_count);
  }

  if (opts_.precompute_unigram_logprobs) {
    if constexpr (sizeof(Token) <= 2) {
      const std::size_t token_space = static_cast<std::size_t>(1) << (8 * sizeof(Token));
      std::vector<u64> counts(token_space, 0);
      u64 total = 0;
      for (const auto& shard : index_.shards()) {
        const Token* toks = reinterpret_cast<const Token*>(shard.ds.data());
        for (u64 i = 0; i < shard.tok_cnt; i++) {
          counts[static_cast<std::size_t>(toks[i])] += 1;
        }
        total += shard.tok_cnt;
      }
      const double log_total = std::log(static_cast<double>(total));
      for (std::size_t i = 0; i < counts.size(); i++) {
        if (counts[i] == 0) {
          continue;
        }
        unigram_logprobs_[static_cast<Token>(i)] = std::log(static_cast<double>(counts[i])) - log_total;
      }
    } else {
      std::unordered_map<Token, u64> counts;
      u64 total = 0;
      for (const auto& shard : index_.shards()) {
        const Token* toks = reinterpret_cast<const Token*>(shard.ds.data());
        for (u64 i = 0; i < shard.tok_cnt; i++) {
          counts[toks[i]] += 1;
        }
        total += shard.tok_cnt;
      }
      const double log_total = std::log(static_cast<double>(total));
      for (const auto& [tok, cnt] : counts) {
        unigram_logprobs_[tok] = std::log(static_cast<double>(cnt)) - log_total;
      }
    }
  }
}

template <typename Token>
void Engine<Token>::CheckLittleEndian() {
  const u32 x = 1;
  const u8* p = reinterpret_cast<const u8*>(&x);
  if (p[0] != 1) {
    throw std::runtime_error("gram requires little-endian");
  }
}

template <typename Token>
u64 Engine<Token>::tok_cnt(std::size_t s) const {
  if (s >= num_shards()) {
    throw std::out_of_range("shard index out of range");
  }
  return ShardAt(s).tok_cnt;
}

template <typename Token>
u64 Engine<Token>::ds_size(std::size_t s) const {
  if (s >= num_shards()) {
    throw std::out_of_range("shard index out of range");
  }
  return ShardAt(s).ds_size;
}

template <typename Token>
u64 Engine<Token>::total_tok_cnt() const {
  u64 total = 0;
  for (const auto& shard : index_.shards()) {
    total += shard.tok_cnt;
  }
  return total;
}

template <typename Token>
std::map<Token, u64> Engine<Token>::ComputeUnigramCounts(std::size_t s) const {
  if (s >= num_shards()) {
    throw std::out_of_range("shard index out of range");
  }
  const auto& shard = ShardAt(s);
  if constexpr (sizeof(Token) <= 2) {
    const std::size_t token_space = static_cast<std::size_t>(1) << (8 * sizeof(Token));
    std::vector<u64> counts(token_space, 0);
    const Token* toks = reinterpret_cast<const Token*>(shard.ds.data());
    for (u64 i = 0; i < shard.tok_cnt; i++) {
      counts[static_cast<std::size_t>(toks[i])] += 1;
    }
    std::map<Token, u64> out;
    for (std::size_t i = 0; i < counts.size(); i++) {
      if (counts[i] != 0) {
        out[static_cast<Token>(i)] = counts[i];
      }
    }
    return out;
  } else {
    std::unordered_map<Token, u64> counts;
    const Token* toks = reinterpret_cast<const Token*>(shard.ds.data());
    for (u64 i = 0; i < shard.tok_cnt; i++) {
      counts[toks[i]] += 1;
    }
    std::map<Token, u64> out;
    for (const auto& [tok, cnt] : counts) {
      out[tok] = cnt;
    }
    return out;
  }
}

template <typename Token>
Token Engine<Token>::DocSepId() const {
  if constexpr (sizeof(Token) == 1) {
    return static_cast<Token>(0xFF);
  } else if constexpr (sizeof(Token) == 2) {
    return static_cast<Token>(0xFFFF);
  } else {
    return static_cast<Token>(0xFFFFFFFFu);
  }
}

template <typename Token>
u64 Engine<Token>::ConvertRankToPtr(const Shard& shard, u64 rank) const {
  const u8* src = shard.sa.data() + rank * shard.ptr_size;
  switch (shard.ptr_size) {
    case 1:
      return src[0];
    case 2: {
      u16 v;
      std::memcpy(&v, src, sizeof(v));
      return v;
    }
    case 3:
      return static_cast<u64>(src[0]) | (static_cast<u64>(src[1]) << 8) | (static_cast<u64>(src[2]) << 16);
    case 4: {
      u32 v;
      std::memcpy(&v, src, sizeof(v));
      return v;
    }
    case 5: {
      u32 lo;
      std::memcpy(&lo, src, sizeof(lo));
      return static_cast<u64>(lo) | (static_cast<u64>(src[4]) << 32);
    }
    case 6: {
      u32 lo;
      std::memcpy(&lo, src, sizeof(lo));
      u16 hi;
      std::memcpy(&hi, src + 4, sizeof(hi));
      return static_cast<u64>(lo) | (static_cast<u64>(hi) << 32);
    }
    case 7: {
      u32 lo;
      std::memcpy(&lo, src, sizeof(lo));
      u16 hi;
      std::memcpy(&hi, src + 4, sizeof(hi));
      return static_cast<u64>(lo) | (static_cast<u64>(hi) << 32) | (static_cast<u64>(src[6]) << 48);
    }
    case 8: {
      u64 v;
      std::memcpy(&v, src, sizeof(v));
      return v;
    }
    default:
      throw std::runtime_error("unsupported ptr_size");
  }
}

template <typename Token>
Token Engine<Token>::ConvertPtrToRawTokenId(const Shard& shard, u64 ptr) const {
  if (ptr == shard.ds_size) {
    return DocSepId();
  }
  Token v;
  std::memcpy(&v, shard.ds.data() + ptr, sizeof(Token));
  return v;
}

template <typename Token>
Token Engine<Token>::ConvertPtrToTokenId(const Shard& shard, u64 ptr) const {
  if (ptr == shard.ds_size) {
    return EosTokenId();
  }
  Token v;
  std::memcpy(&v, shard.ds.data() + ptr, sizeof(Token));
  if (v == DocSepId()) {
    return EosTokenId();
  }
  return v;
}

template <typename Token>
int Engine<Token>::ComparePrefix(const Shard& shard, u64 ptr, const u8* input_buf, u64 num_bytes) const {
  const u64 avail = shard.ds_size - ptr;
  const u64 len = (avail < num_bytes) ? avail : num_bytes;
  const int cmp = std::memcmp(shard.ds.data() + ptr, input_buf, static_cast<std::size_t>(len));
  if (cmp != 0) {
    return cmp;
  }
  if (avail < num_bytes) {
    return -1;
  }
  return 0;
}

template <typename Token>
void Engine<Token>::FindThread(std::size_t s,
                               const u8* input_buf,
                               u64 num_bytes,
                               std::pair<u64, u64> hint_segment,
                               std::pair<u64, u64>* out_segment) const {
  const auto& shard = ShardAt(s);
  if (num_bytes == 0) {
    *out_segment = {0, shard.tok_cnt};
    return;
  }

  const u64 start = hint_segment.first;
  const u64 end = hint_segment.second;

  u64 lo = start;
  u64 hi = end;
  while (lo < hi) {
    const u64 mid = (lo + hi) >> 1;
    const u64 ptr = ConvertRankToPtr(shard, mid);
    const int cmp = ComparePrefix(shard, ptr, input_buf, num_bytes);
    if (cmp < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const u64 left = lo;

  lo = left;
  hi = end;
  while (lo < hi) {
    const u64 mid = (lo + hi) >> 1;
    const u64 ptr = ConvertRankToPtr(shard, mid);
    const int cmp = ComparePrefix(shard, ptr, input_buf, num_bytes);
    if (cmp > 0) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  const u64 right = lo;

  *out_segment = {left, right};
}

template <typename Token>
FindResult Engine<Token>::FindBytes(const u8* input_buf,
                                    u64 num_bytes,
                                    const std::vector<std::pair<u64, u64>>& hint_segment_by_shard) const {
  if (hint_segment_by_shard.size() != num_shards()) {
    throw std::invalid_argument("hint shard count mismatch");
  }

  std::vector<std::pair<u64, u64>> segment_by_shard(num_shards());
  if (!pool_.has_value() || num_shards() == 1) {
    for (std::size_t s = 0; s < num_shards(); s++) {
      FindThread(s, input_buf, num_bytes, hint_segment_by_shard[s], &segment_by_shard[s]);
    }
  } else {
    for (std::size_t s = 0; s < num_shards(); s++) {
      pool_->Enqueue([this, s, input_buf, num_bytes, &hint_segment_by_shard, &segment_by_shard]() {
        FindThread(s, input_buf, num_bytes, hint_segment_by_shard[s], &segment_by_shard[s]);
      });
    }
    pool_->WaitIdle();
  }

  u64 cnt = 0;
  for (const auto& seg : segment_by_shard) {
    cnt += seg.second - seg.first;
  }
  return FindResult{.cnt = cnt, .segment_by_shard = std::move(segment_by_shard)};
}

template <typename Token>
FindResult Engine<Token>::Find(const std::vector<Token>& input_ids) const {
  std::vector<std::pair<u64, u64>> hints;
  if constexpr (sizeof(Token) <= 2) {
    if (!input_ids.empty() && index_.unigram_ranges().has_value()) {
      const u64 token_ix = static_cast<u64>((Version() == 5) ? input_ids.back() : input_ids.front());
      const u64* ranges = reinterpret_cast<const u64*>(index_.unigram_ranges()->data());
      hints.reserve(num_shards());
      u64 cnt = 0;
      for (std::size_t s = 0; s < num_shards(); s++) {
        const u64 base = (token_ix * num_shards() + s) * 2;
        const u64 start = ranges[base];
        const u64 end = ranges[base + 1];
        hints.push_back({start, end});
        cnt += end - start;
      }
      if (input_ids.size() == 1) {
        return FindResult{.cnt = cnt, .segment_by_shard = std::move(hints)};
      }
    }
  }
  if (hints.empty()) {
    hints.reserve(num_shards());
    for (const auto& shard : index_.shards()) {
      hints.push_back({0, shard.tok_cnt});
    }
  }

  std::vector<Token> reversed;
  const u8* input_buf = nullptr;
  if (Version() == 4) {
    input_buf = reinterpret_cast<const u8*>(input_ids.data());
  } else if (Version() == 5) {
    reversed = input_ids;
    std::reverse(reversed.begin(), reversed.end());
    input_buf = reinterpret_cast<const u8*>(reversed.data());
  } else {
    throw std::invalid_argument("unsupported version");
  }
  const u64 num_bytes = static_cast<u64>(input_ids.size() * sizeof(Token));
  return FindBytes(input_buf, num_bytes, hints);
}

template <typename Token>
FindResult Engine<Token>::FindWithHint(const std::vector<Token>& input_ids,
                                       const std::vector<std::pair<u64, u64>>& hint_segment_by_shard) const {
  std::vector<Token> reversed;
  const u8* input_buf = nullptr;
  if (Version() == 4) {
    input_buf = reinterpret_cast<const u8*>(input_ids.data());
  } else if (Version() == 5) {
    reversed = input_ids;
    std::reverse(reversed.begin(), reversed.end());
    input_buf = reinterpret_cast<const u8*>(reversed.data());
  } else {
    throw std::invalid_argument("unsupported version");
  }
  const u64 num_bytes = static_cast<u64>(input_ids.size() * sizeof(Token));
  return FindBytes(input_buf, num_bytes, hint_segment_by_shard);
}

template <typename Token>
CountResult Engine<Token>::Count(const std::vector<Token>& input_ids) const {
  auto fr = Find(input_ids);
  return CountResult{.count = fr.cnt, .approx = false};
}

template <typename Token>
ProbResult Engine<Token>::PrimitiveProb(const std::vector<Token>& prompt_ids, Token cont_id) const {
  const auto prompt = Find(prompt_ids);
  if (prompt.cnt == 0) {
    return ProbResult{.prompt_cnt = 0, .cont_cnt = 0, .prob = -1.0};
  }
  std::vector<Token> joined = prompt_ids;
  joined.push_back(cont_id);
  FindResult cont;
  if (Version() == 4) {
    cont = FindWithHint(joined, prompt.segment_by_shard);
  } else {
    cont = Find(joined);
  }
  return ProbResult{.prompt_cnt = prompt.cnt, .cont_cnt = cont.cnt, .prob = static_cast<double>(cont.cnt) / prompt.cnt};
}

template <typename Token>
DistResult<Token> Engine<Token>::PrimitiveNtd(const std::vector<Token>& prompt_ids, u64 max_support) const {
  auto fr = Find(prompt_ids);
  if (fr.cnt == 0) {
    return DistResult<Token>{.prompt_cnt = 0, .result_by_token_id = {}, .approx = false};
  }
  return NtdFromSegment(prompt_ids.size() * sizeof(Token), fr.segment_by_shard, max_support);
}

template <typename Token>
DistResult<Token> Engine<Token>::NtdFromSegment(std::size_t num_bytes,
                                                const std::vector<std::pair<u64, u64>>& segment_by_shard,
                                                u64 max_support) const {
  if (segment_by_shard.size() != num_shards()) {
    throw std::invalid_argument("segment shard count mismatch");
  }
  u64 prompt_cnt = 0;
  for (const auto& seg : segment_by_shard) {
    prompt_cnt += seg.second - seg.first;
  }
  if (prompt_cnt == 0) {
    return DistResult<Token>{.prompt_cnt = 0, .result_by_token_id = {}, .approx = false};
  }
  u64 unit = 1;
  while (prompt_cnt > unit * max_support) {
    unit <<= 1;
  }
  const bool approx = (unit > 1);

  std::vector<std::unordered_map<Token, u64>> freq_by_shard(num_shards());
  for (auto& m : freq_by_shard) {
    m.reserve(static_cast<std::size_t>(max_support * 2));
  }
  if (!pool_.has_value() || num_shards() == 1) {
    for (std::size_t s = 0; s < num_shards(); s++) {
      GetFreqByTokenIdApprox(s, num_bytes, segment_by_shard[s], unit, nullptr, nullptr, &freq_by_shard[s]);
    }
  } else {
    for (std::size_t s = 0; s < num_shards(); s++) {
      pool_->Enqueue([this, s, num_bytes, &segment_by_shard, unit, &freq_by_shard]() {
        GetFreqByTokenIdApprox(s, num_bytes, segment_by_shard[s], unit, nullptr, nullptr, &freq_by_shard[s]);
      });
    }
    pool_->WaitIdle();
  }

  std::unordered_map<Token, u64> merged;
  merged.reserve(static_cast<std::size_t>(max_support * 4));
  for (const auto& m : freq_by_shard) {
    for (const auto& [tok, cnt] : m) {
      merged[tok] += cnt;
    }
  }
  u64 total = 0;
  for (const auto& [tok, cnt] : merged) {
    (void)tok;
    total += cnt;
  }
  std::map<Token, DistTokenResult> result;
  for (const auto& [tok, cnt] : merged) {
    result[tok] = DistTokenResult{.cont_cnt = cnt, .prob = static_cast<double>(cnt) / total};
  }
  return DistResult<Token>{.prompt_cnt = total, .result_by_token_id = std::move(result), .approx = approx};
}

template <typename Token>
static u32 LexKey(Token token_id) {
  if constexpr (sizeof(Token) == 1) {
    return static_cast<u32>(static_cast<u8>(token_id));
  } else if constexpr (sizeof(Token) == 2) {
    return static_cast<u32>(__builtin_bswap16(static_cast<u16>(token_id)));
  } else {
    return __builtin_bswap32(static_cast<u32>(token_id));
  }
}

template <typename Token>
std::pair<u64, u64> Engine<Token>::RefineSegmentByNextToken(const Shard& shard,
                                                            u64 num_bytes,
                                                            std::pair<u64, u64> segment,
                                                            Token next_token_id) const {
  u64 start = segment.first;
  u64 end = segment.second;
  if (start >= end) {
    return {start, start};
  }
  const Token raw_target = (next_token_id == EosTokenId()) ? DocSepId() : next_token_id;
  const u32 target_key = LexKey(raw_target);

  auto token_after_prefix_raw = [&](u64 rank) -> Token {
    const u64 ptr = ConvertRankToPtr(shard, rank);
    const u64 next_ptr = (Version() == 4) ? (ptr + num_bytes) : (ptr - sizeof(Token));
    return ConvertPtrToRawTokenId(shard, next_ptr);
  };

  u64 lo = start;
  u64 hi = end;
  while (lo < hi) {
    const u64 mid = (lo + hi) >> 1;
    const u32 mid_key = LexKey(token_after_prefix_raw(mid));
    if (mid_key < target_key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const u64 left = lo;

  lo = left;
  hi = end;
  while (lo < hi) {
    const u64 mid = (lo + hi) >> 1;
    const u32 mid_key = LexKey(token_after_prefix_raw(mid));
    if (mid_key <= target_key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const u64 right = lo;
  return {left, right};
}

template <typename Token>
FindResult Engine<Token>::AdvanceSegmentByToken(const std::vector<std::pair<u64, u64>>& segment_by_shard,
                                                std::size_t num_bytes,
                                                Token next_token_id) const {
  if (segment_by_shard.size() != num_shards()) {
    throw std::invalid_argument("segment shard count mismatch");
  }
  std::vector<std::pair<u64, u64>> new_seg;
  new_seg.reserve(num_shards());
  u64 cnt = 0;
  for (std::size_t s = 0; s < num_shards(); s++) {
    const auto seg = RefineSegmentByNextToken(ShardAt(s), static_cast<u64>(num_bytes), segment_by_shard[s], next_token_id);
    new_seg.push_back(seg);
    cnt += seg.second - seg.first;
  }
  return FindResult{.cnt = cnt, .segment_by_shard = std::move(new_seg)};
}

template <typename Token>
void Engine<Token>::GetFreqByTokenIdApprox(std::size_t s,
                                           u64 num_bytes,
                                           std::pair<u64, u64> segment,
                                           u64 unit,
                                           const Token* token_start,
                                           const Token* token_end,
                                           std::unordered_map<Token, u64>* out_freq_by_token_id) const {
  const auto& shard = ShardAt(s);
  u64 start = segment.first;
  u64 end = segment.second;
  if (end - start < 4 * unit) {
    for (u64 rank = start; rank < end; rank += unit) {
      const u64 rank_mid = (rank + unit <= end) ? (rank + (unit >> 1)) : ((rank + end) >> 1);
      const u64 ptr = ConvertRankToPtr(shard, rank_mid);
      const u64 next_ptr = (Version() == 4) ? (ptr + num_bytes) : (ptr - sizeof(Token));
      const Token tok = ConvertPtrToTokenId(shard, next_ptr);
      (*out_freq_by_token_id)[tok] += (rank + unit <= end) ? unit : (end - rank);
    }
    return;
  }

  Token t_start = token_start ? *token_start : ConvertPtrToTokenId(shard, (Version() == 4)
                                                                             ? (ConvertRankToPtr(shard, start) + num_bytes)
                                                                             : (ConvertRankToPtr(shard, start) - sizeof(Token)));
  Token t_end = token_end ? *token_end : ConvertPtrToTokenId(shard, (Version() == 4)
                                                                       ? (ConvertRankToPtr(shard, end - 1) + num_bytes)
                                                                       : (ConvertRankToPtr(shard, end - 1) - sizeof(Token)));
  if (t_start == t_end) {
    (*out_freq_by_token_id)[t_start] += end - start;
    return;
  }

  const u64 mi = (start + end) >> 1;
  GetFreqByTokenIdApprox(s, num_bytes, {start, mi}, unit, &t_start, nullptr, out_freq_by_token_id);
  GetFreqByTokenIdApprox(s, num_bytes, {mi, end}, unit, nullptr, &t_end, out_freq_by_token_id);
}

template <typename Token>
InfgramProbResult Engine<Token>::Prob(const std::vector<Token>& prompt_ids, Token cont_id) const {
  const std::size_t L = prompt_ids.size();
  std::size_t l_lo = 0;
  std::size_t l_hi = 1;
  for (;;) {
    if (l_hi > L) {
      l_hi = L + 1;
      break;
    }
    std::vector<Token> suffix(prompt_ids.end() - static_cast<std::ptrdiff_t>(l_hi), prompt_ids.end());
    if (Find(suffix).cnt == 0) {
      break;
    }
    l_lo = l_hi;
    l_hi <<= 1;
  }
  while (l_hi - l_lo > 1) {
    const std::size_t l_mid = (l_lo + l_hi) >> 1;
    std::vector<Token> suffix(prompt_ids.end() - static_cast<std::ptrdiff_t>(l_mid), prompt_ids.end());
    if (Find(suffix).cnt == 0) {
      l_hi = l_mid;
    } else {
      l_lo = l_mid;
    }
  }
  const u64 suffix_len = static_cast<u64>(l_lo);
  std::vector<Token> suffix(prompt_ids.end() - static_cast<std::ptrdiff_t>(suffix_len), prompt_ids.end());
  auto r = PrimitiveProb(suffix, cont_id);
  return InfgramProbResult{.prompt_cnt = r.prompt_cnt, .cont_cnt = r.cont_cnt, .prob = r.prob, .suffix_len = suffix_len};
}

template <typename Token>
InfgramDistResult<Token> Engine<Token>::Ntd(const std::vector<Token>& prompt_ids, u64 max_support) const {
  const std::size_t L = prompt_ids.size();
  std::size_t l_lo = 0;
  std::size_t l_hi = 1;
  for (;;) {
    if (l_hi > L) {
      l_hi = L + 1;
      break;
    }
    std::vector<Token> suffix(prompt_ids.end() - static_cast<std::ptrdiff_t>(l_hi), prompt_ids.end());
    if (Find(suffix).cnt == 0) {
      break;
    }
    l_lo = l_hi;
    l_hi <<= 1;
  }
  while (l_hi - l_lo > 1) {
    const std::size_t l_mid = (l_lo + l_hi) >> 1;
    std::vector<Token> suffix(prompt_ids.end() - static_cast<std::ptrdiff_t>(l_mid), prompt_ids.end());
    if (Find(suffix).cnt == 0) {
      l_hi = l_mid;
    } else {
      l_lo = l_mid;
    }
  }
  const u64 suffix_len = static_cast<u64>(l_lo);
  std::vector<Token> suffix(prompt_ids.end() - static_cast<std::ptrdiff_t>(suffix_len), prompt_ids.end());
  auto r = PrimitiveNtd(suffix, max_support);
  return InfgramDistResult<Token>{
      .prompt_cnt = r.prompt_cnt,
      .result_by_token_id = r.result_by_token_id,
      .approx = r.approx,
      .suffix_len = suffix_len,
  };
}

template <typename Token>
typename Engine<Token>::FindDisjResult Engine<Token>::FindDisj(const std::vector<std::vector<Token>>& disj_clause,
                                                               u64 max_clause_freq) const {
  std::vector<FindResult> find_result_by_term(disj_clause.size());
  for (std::size_t t = 0; t < disj_clause.size(); t++) {
    find_result_by_term[t] = Find(disj_clause[t]);
  }

  FindDisjResult out;
  out.cnt_by_shard.resize(num_shards());
  out.segment_by_term_by_shard.resize(num_shards());
  out.subsampling_factor_by_shard.resize(num_shards());

  const u64 max_per_shard = (max_clause_freq + (num_shards() - 1)) / num_shards();
  for (std::size_t s = 0; s < num_shards(); s++) {
    FindDisjThread(s, &find_result_by_term, max_per_shard, &out.cnt_by_shard[s], &out.segment_by_term_by_shard[s],
                   &out.subsampling_factor_by_shard[s]);
  }
  out.cnt = std::accumulate(out.cnt_by_shard.begin(), out.cnt_by_shard.end(), static_cast<u64>(0));
  return out;
}

template <typename Token>
void Engine<Token>::FindDisjThread(std::size_t s,
                                   const std::vector<FindResult>* find_result_by_term,
                                   u64 max_clause_freq_per_shard,
                                   u64* out_cnt,
                                   std::vector<std::pair<u64, u64>>* out_segment_by_term,
                                   double* out_subsampling_factor) const {
  std::mt19937 gen(19260817);

  u64 cnt = 0;
  std::vector<std::pair<u64, u64>> segs;
  segs.reserve(find_result_by_term->size());
  for (const auto& fr : *find_result_by_term) {
    const auto seg = fr.segment_by_shard[s];
    segs.push_back(seg);
    cnt += seg.second - seg.first;
  }
  double subsampling_factor = 1.0;
  if (cnt > max_clause_freq_per_shard) {
    u64 new_cnt = 0;
    std::vector<std::pair<u64, u64>> new_segs;
    new_segs.reserve(segs.size());
    for (const auto& [start, end] : segs) {
      const u64 length = end - start;
      const u64 new_length = (length * max_clause_freq_per_shard + (cnt - 1)) / cnt;
      std::uniform_int_distribution<u64> dis(0, length - new_length);
      const u64 new_start = start + dis(gen);
      const u64 new_end = new_start + new_length;
      new_cnt += new_length;
      new_segs.push_back({new_start, new_end});
    }
    subsampling_factor = static_cast<double>(cnt) / new_cnt;
    segs = std::move(new_segs);
  }

  *out_cnt = cnt;
  *out_segment_by_term = std::move(segs);
  *out_subsampling_factor = subsampling_factor;
}

template <typename Token>
void Engine<Token>::FindCnfThread(std::size_t s,
                                  const std::vector<FindDisjResult>* find_disj_result_by_clause,
                                  u64 max_diff_tokens,
                                  u64* out_cnt,
                                  std::vector<std::pair<u64, u64>>* out_valid_ptr_ranges,
                                  double* out_subsampling_factor) const {
  std::vector<FindDisjResult> clauses = *find_disj_result_by_clause;
  std::sort(clauses.begin(), clauses.end(),
            [s](const FindDisjResult& a, const FindDisjResult& b) { return a.cnt_by_shard[s] < b.cnt_by_shard[s]; });

  const auto& shard = ShardAt(s);
  const auto& first = clauses[0];

  std::vector<std::pair<u64, u64>> valid;
  for (const auto& [start, end] : first.segment_by_term_by_shard[s]) {
    std::vector<u64> ptrs;
    ptrs.reserve(end - start);
    for (u64 rank = start; rank < end; rank++) {
      ptrs.push_back(ConvertRankToPtr(shard, rank));
    }
    for (const auto ptr : ptrs) {
      valid.push_back({ptr, ptr});
    }
  }
  double subsampling_factor = first.subsampling_factor_by_shard[s];

  for (std::size_t d = 1; d < clauses.size(); d++) {
    const auto& clause = clauses[d];
    std::vector<u64> ptrs;
    for (const auto& [start, end] : clause.segment_by_term_by_shard[s]) {
      std::vector<u64> new_ptrs;
      new_ptrs.reserve(end - start);
      for (u64 rank = start; rank < end; rank++) {
        new_ptrs.push_back(ConvertRankToPtr(shard, rank));
      }
      ptrs.insert(ptrs.end(), new_ptrs.begin(), new_ptrs.end());
    }
    std::sort(ptrs.begin(), ptrs.end());

    std::vector<std::pair<u64, u64>> next_valid;
    for (const auto& [l, r] : valid) {
      const auto lo = BinSearch(ptrs, r).first;
      const u64 new_l = (lo == static_cast<u64>(-1)) ? static_cast<u64>(-1) : std::min(l, ptrs[lo]);
      const auto hi = BinSearch(ptrs, l).second;
      const u64 new_r = (hi == ptrs.size()) ? static_cast<u64>(-1) : std::max(r, ptrs[hi]);
      const u64 max_diff = max_diff_tokens * sizeof(Token);
      if (new_l != static_cast<u64>(-1) && new_l + max_diff >= l && new_r != static_cast<u64>(-1) && new_r <= r + max_diff) {
        next_valid.push_back({new_l, new_r});
      } else {
        if (new_l != static_cast<u64>(-1) && new_l + max_diff >= l) {
          next_valid.push_back({new_l, r});
        }
        if (new_r != static_cast<u64>(-1) && new_r <= r + max_diff) {
          next_valid.push_back({l, new_r});
        }
      }
    }
    valid = std::move(next_valid);
    subsampling_factor *= clause.subsampling_factor_by_shard[s];
  }

  const std::vector<u8> doc_sep_bytes(sizeof(Token), 0xFF);
  std::vector<std::pair<u64, u64>> filtered;
  for (const auto& [l, r] : valid) {
    const u8* begin = shard.ds.data() + l;
    const u8* end = shard.ds.data() + r;
    auto it = std::search(begin, end, doc_sep_bytes.begin(), doc_sep_bytes.end());
    if (it == end) {
      filtered.push_back({l, r});
    }
  }
  valid = std::move(filtered);

  *out_cnt = static_cast<u64>(valid.size() * subsampling_factor);
  *out_valid_ptr_ranges = std::move(valid);
  *out_subsampling_factor = subsampling_factor;
}

template <typename Token>
FindCnfResult Engine<Token>::FindCnf(const std::vector<std::vector<std::vector<Token>>>& cnf,
                                     u64 max_clause_freq,
                                     u64 max_diff_tokens) const {
  if (cnf.empty()) {
    throw std::invalid_argument("cnf must be non-empty");
  }
  std::vector<FindDisjResult> disj_results(cnf.size());
  for (std::size_t c = 0; c < cnf.size(); c++) {
    disj_results[c] = FindDisj(cnf[c], max_clause_freq);
    if (disj_results[c].cnt == 0) {
      return FindCnfResult{.cnt = 0, .approx = false, .ptrs_by_shard = {}};
    }
  }

  std::vector<u64> cnt_by_shard(num_shards());
  std::vector<std::vector<std::pair<u64, u64>>> valid_ranges(num_shards());
  std::vector<double> subsampling_factor(num_shards());

  for (std::size_t s = 0; s < num_shards(); s++) {
    FindCnfThread(s, &disj_results, max_diff_tokens, &cnt_by_shard[s], &valid_ranges[s], &subsampling_factor[s]);
  }

  const u64 cnt = std::accumulate(cnt_by_shard.begin(), cnt_by_shard.end(), static_cast<u64>(0));
  const bool approx = std::any_of(subsampling_factor.begin(), subsampling_factor.end(), [](double f) { return f != 1.0; });

  std::vector<std::vector<u64>> ptrs_by_shard(num_shards());
  for (std::size_t s = 0; s < num_shards(); s++) {
    for (const auto& [l, r] : valid_ranges[s]) {
      (void)r;
      ptrs_by_shard[s].push_back(l);
    }
  }

  return FindCnfResult{.cnt = cnt, .approx = approx, .ptrs_by_shard = std::move(ptrs_by_shard)};
}

template <typename Token>
CountResult Engine<Token>::CountCnf(const std::vector<std::vector<std::vector<Token>>>& cnf,
                                    u64 max_clause_freq,
                                    u64 max_diff_tokens) const {
  auto r = FindCnf(cnf, max_clause_freq, max_diff_tokens);
  return CountResult{.count = r.cnt, .approx = r.approx};
}

template <typename Token>
u64 Engine<Token>::ConvertDocIxToPtr(const Shard& shard, u64 doc_ix) const {
  if (doc_ix == shard.doc_cnt) {
    return shard.ds_size;
  }
  u64 ptr;
  std::memcpy(&ptr, shard.od.data() + doc_ix * sizeof(u64), sizeof(ptr));
  return ptr;
}

template <typename Token>
u64 Engine<Token>::ConvertDocIxToMetaPtr(const Shard& shard, u64 doc_ix) const {
  if (!shard.mt.has_value() || !shard.om.has_value()) {
    return 0;
  }
  if (doc_ix == shard.doc_cnt) {
    return shard.mt->size();
  }
  u64 ptr;
  std::memcpy(&ptr, shard.om->data() + doc_ix * sizeof(u64), sizeof(ptr));
  return ptr;
}

template <typename Token>
std::pair<u64, u64> Engine<Token>::BinSearch(const std::vector<u64>& arr, u64 val) const {
  const auto it = std::lower_bound(arr.begin(), arr.end(), val);
  const u64 hi = static_cast<u64>(std::distance(arr.begin(), it));
  const u64 lo = (hi == 0) ? static_cast<u64>(-1) : (hi - 1);
  return {lo, hi};
}

template <typename Token>
DocResult<Token> Engine<Token>::GetDocByPtr(std::size_t s, u64 ptr, u64 max_disp_len) const {
  const auto& shard = ShardAt(s);
  const u64 max_pre = max_disp_len / 2;
  const u64 max_app = (max_disp_len + 1) / 2;

  u64 lo = 0;
  u64 hi = shard.doc_cnt;
  while (hi - lo > 1) {
    const u64 mi = (lo + hi) >> 1;
    const u64 p = ConvertDocIxToPtr(shard, mi);
    if (p <= ptr) {
      lo = mi;
    } else {
      hi = mi;
    }
  }

  const u64 local_doc_ix = lo;
  u64 doc_ix = 0;
  for (std::size_t i = 0; i < s; i++) {
    doc_ix += ShardAt(i).doc_cnt;
  }
  doc_ix += local_doc_ix;

  const u64 doc_start_ptr = ConvertDocIxToPtr(shard, local_doc_ix) + sizeof(Token);
  const u64 doc_end_ptr = ConvertDocIxToPtr(shard, local_doc_ix + 1);
  const u64 doc_len = (doc_end_ptr - doc_start_ptr) / sizeof(Token);

  const u64 disp_start_ptr =
      std::max(doc_start_ptr, ptr < sizeof(Token) * max_pre ? static_cast<u64>(0) : (ptr - sizeof(Token) * max_pre));
  const u64 disp_end_ptr = std::min(doc_end_ptr, ptr + sizeof(Token) * max_app);
  const u64 disp_len = (disp_end_ptr - disp_start_ptr) / sizeof(Token);
  const u64 needle_offset = (ptr - disp_start_ptr) / sizeof(Token);

  std::string metadata;
  if (shard.mt.has_value()) {
    const u64 meta_start = ConvertDocIxToMetaPtr(shard, local_doc_ix);
    const u64 meta_end = ConvertDocIxToMetaPtr(shard, local_doc_ix + 1);
    metadata.assign(reinterpret_cast<const char*>(shard.mt->data() + meta_start),
                    reinterpret_cast<const char*>(shard.mt->data() + meta_end));
  }

  std::vector<Token> token_ids;
  const Token* begin = reinterpret_cast<const Token*>(shard.ds.data() + disp_start_ptr);
  const Token* end = reinterpret_cast<const Token*>(shard.ds.data() + disp_end_ptr);
  token_ids.assign(begin, end);
  if (Version() == 5) {
    std::reverse(token_ids.begin(), token_ids.end());
  }

  return DocResult<Token>{
      .doc_ix = doc_ix,
      .doc_len = doc_len,
      .disp_len = disp_len,
      .needle_offset = needle_offset,
      .metadata = std::move(metadata),
      .token_ids = std::move(token_ids),
  };
}

template <typename Token>
DocResult<Token> Engine<Token>::GetDocByRank(std::size_t s, u64 rank, u64 max_disp_len) const {
  const auto& shard = ShardAt(s);
  const u64 ptr = ConvertRankToPtr(shard, rank);
  return GetDocByPtr(s, ptr, max_disp_len);
}

template <typename Token>
std::vector<DocResult<Token>> Engine<Token>::GetDocsByPtrs(const std::vector<std::pair<std::size_t, u64>>& list_of_s_and_ptr,
                                                           u64 max_disp_len) const {
  std::vector<DocResult<Token>> out;
  out.reserve(list_of_s_and_ptr.size());
  for (const auto& [s, ptr] : list_of_s_and_ptr) {
    out.push_back(GetDocByPtr(s, ptr, max_disp_len));
  }
  return out;
}

template <typename Token>
std::vector<DocResult<Token>> Engine<Token>::GetDocsByRanks(const std::vector<std::pair<std::size_t, u64>>& list_of_s_and_rank,
                                                            u64 max_disp_len) const {
  std::vector<DocResult<Token>> out;
  out.reserve(list_of_s_and_rank.size());
  for (const auto& [s, rank] : list_of_s_and_rank) {
    out.push_back(GetDocByRank(s, rank, max_disp_len));
  }
  return out;
}

template <typename Token>
u64 Engine<Token>::TotalDocCnt() const {
  u64 total = 0;
  for (const auto& shard : index_.shards()) {
    total += shard.doc_cnt;
  }
  return total;
}

template <typename Token>
DocResult<Token> Engine<Token>::GetDocByIx(u64 doc_ix, u64 max_disp_len) const {
  u64 local = doc_ix;
  std::size_t s = 0;
  while (s < num_shards() && local >= ShardAt(s).doc_cnt) {
    local -= ShardAt(s).doc_cnt;
    s++;
  }
  if (s >= num_shards()) {
    throw std::out_of_range("doc_ix out of range");
  }

  const auto& shard = ShardAt(s);
  const u64 doc_start_ptr = ConvertDocIxToPtr(shard, local) + sizeof(Token);
  const u64 doc_end_ptr = ConvertDocIxToPtr(shard, local + 1);
  const u64 doc_len = (doc_end_ptr - doc_start_ptr) / sizeof(Token);

  const u64 disp_start_ptr = doc_start_ptr;
  const u64 disp_end_ptr = std::min(doc_end_ptr, doc_start_ptr + sizeof(Token) * max_disp_len);
  const u64 disp_len = (disp_end_ptr - disp_start_ptr) / sizeof(Token);

  std::string metadata;
  if (shard.mt.has_value()) {
    const u64 meta_start = ConvertDocIxToMetaPtr(shard, local);
    const u64 meta_end = ConvertDocIxToMetaPtr(shard, local + 1);
    metadata.assign(reinterpret_cast<const char*>(shard.mt->data() + meta_start),
                    reinterpret_cast<const char*>(shard.mt->data() + meta_end));
  }

  std::vector<Token> token_ids;
  const Token* begin = reinterpret_cast<const Token*>(shard.ds.data() + disp_start_ptr);
  const Token* end = reinterpret_cast<const Token*>(shard.ds.data() + disp_end_ptr);
  token_ids.assign(begin, end);
  if (Version() == 5) {
    std::reverse(token_ids.begin(), token_ids.end());
  }

  return DocResult<Token>{
      .doc_ix = doc_ix,
      .doc_len = doc_len,
      .disp_len = disp_len,
      .needle_offset = 0,
      .metadata = std::move(metadata),
      .token_ids = std::move(token_ids),
  };
}

template <typename Token>
std::vector<DocResult<Token>> Engine<Token>::GetDocsByIxs(const std::vector<u64>& list_of_doc_ix, u64 max_disp_len) const {
  std::vector<DocResult<Token>> out;
  out.reserve(list_of_doc_ix.size());
  for (const auto ix : list_of_doc_ix) {
    out.push_back(GetDocByIx(ix, max_disp_len));
  }
  return out;
}

template <typename Token>
DocResult<Token> Engine<Token>::GetDocByPtr2(std::size_t s, u64 ptr, u64 needle_len, u64 max_ctx_len) const {
  const auto& shard = ShardAt(s);

  u64 lo = 0;
  u64 hi = shard.doc_cnt;
  while (hi - lo > 1) {
    const u64 mi = (lo + hi) >> 1;
    const u64 p = ConvertDocIxToPtr(shard, mi);
    if (p <= ptr) {
      lo = mi;
    } else {
      hi = mi;
    }
  }

  const u64 local_doc_ix = lo;
  u64 doc_ix = 0;
  for (std::size_t i = 0; i < s; i++) {
    doc_ix += ShardAt(i).doc_cnt;
  }
  doc_ix += local_doc_ix;

  const u64 doc_start_ptr = ConvertDocIxToPtr(shard, local_doc_ix) + sizeof(Token);
  const u64 doc_end_ptr = ConvertDocIxToPtr(shard, local_doc_ix + 1);
  const u64 doc_len = (doc_end_ptr - doc_start_ptr) / sizeof(Token);

  const u64 disp_start_ptr =
      std::max(doc_start_ptr, ptr < sizeof(Token) * max_ctx_len ? static_cast<u64>(0) : (ptr - sizeof(Token) * max_ctx_len));
  const u64 disp_end_ptr = std::min(doc_end_ptr, ptr + sizeof(Token) * (needle_len + max_ctx_len));
  const u64 disp_len = (disp_end_ptr - disp_start_ptr) / sizeof(Token);
  const u64 needle_offset = (ptr - disp_start_ptr) / sizeof(Token);

  std::string metadata;
  if (shard.mt.has_value()) {
    const u64 meta_start = ConvertDocIxToMetaPtr(shard, local_doc_ix);
    const u64 meta_end = ConvertDocIxToMetaPtr(shard, local_doc_ix + 1);
    metadata.assign(reinterpret_cast<const char*>(shard.mt->data() + meta_start),
                    reinterpret_cast<const char*>(shard.mt->data() + meta_end));
  }

  std::vector<Token> token_ids;
  const Token* begin = reinterpret_cast<const Token*>(shard.ds.data() + disp_start_ptr);
  const Token* end = reinterpret_cast<const Token*>(shard.ds.data() + disp_end_ptr);
  token_ids.assign(begin, end);
  if (Version() == 5) {
    std::reverse(token_ids.begin(), token_ids.end());
  }

  return DocResult<Token>{
      .doc_ix = doc_ix,
      .doc_len = doc_len,
      .disp_len = disp_len,
      .needle_offset = needle_offset,
      .metadata = std::move(metadata),
      .token_ids = std::move(token_ids),
  };
}

template <typename Token>
DocResult<Token> Engine<Token>::GetDocByRank2(std::size_t s, u64 rank, u64 needle_len, u64 max_ctx_len) const {
  const auto& shard = ShardAt(s);
  const u64 ptr = ConvertRankToPtr(shard, rank);
  return GetDocByPtr2(s, ptr, needle_len, max_ctx_len);
}

template <typename Token>
std::vector<DocResult<Token>> Engine<Token>::GetDocsByRanks2(
    const std::vector<std::tuple<std::size_t, u64, u64, u64>>& requests) const {
  std::vector<DocResult<Token>> out;
  out.reserve(requests.size());
  for (const auto& [s, rank, needle_len, max_ctx_len] : requests) {
    out.push_back(GetDocByRank2(s, rank, needle_len, max_ctx_len));
  }
  return out;
}

template <typename Token>
std::vector<DocResult<Token>> Engine<Token>::GetDocsByPtrs2(const std::vector<std::tuple<std::size_t, u64, u64, u64>>& requests) const {
  std::vector<DocResult<Token>> out;
  out.reserve(requests.size());
  for (const auto& [s, ptr, needle_len, max_ctx_len] : requests) {
    out.push_back(GetDocByPtr2(s, ptr, needle_len, max_ctx_len));
  }
  return out;
}

template <typename Token>
DocResult<Token> Engine<Token>::GetDocByIx2(u64 doc_ix, u64 max_ctx_len) const {
  (void)max_ctx_len;
  return GetDocByIx(doc_ix, max_ctx_len);
}

template <typename Token>
std::vector<DocResult<Token>> Engine<Token>::GetDocsByIxs2(const std::vector<std::tuple<u64, u64>>& requests) const {
  std::vector<DocResult<Token>> out;
  out.reserve(requests.size());
  for (const auto& [doc_ix, max_ctx_len] : requests) {
    out.push_back(GetDocByIx2(doc_ix, max_ctx_len));
  }
  return out;
}

template <typename Token>
SearchDocsResult<Token> Engine<Token>::SearchDocs(const std::vector<Token>& input_ids, std::size_t maxnum, u64 max_disp_len) const {
  if (maxnum == 0) {
    throw std::invalid_argument("maxnum must be > 0");
  }
  auto fr = Find(input_ids);
  if (fr.cnt == 0) {
    return SearchDocsResult<Token>{.cnt = 0, .approx = false, .idxs = {}, .docs = {}};
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<u64> cnt_by_shard;
  cnt_by_shard.reserve(num_shards());
  for (const auto& seg : fr.segment_by_shard) {
    cnt_by_shard.push_back(seg.second - seg.first);
  }

  std::vector<u64> idxs;
  std::vector<DocResult<Token>> docs;
  idxs.reserve(maxnum);
  docs.reserve(maxnum);

  for (std::size_t d = 0; d < maxnum; d++) {
    std::discrete_distribution<std::size_t> choose_shard(cnt_by_shard.begin(), cnt_by_shard.end());
    const std::size_t s = choose_shard(gen);
    const auto [start, end] = fr.segment_by_shard[s];
    std::uniform_int_distribution<u64> choose_rank(start, end - 1);
    const u64 rank = choose_rank(gen);
    const u64 ptr = ConvertRankToPtr(ShardAt(s), rank);
    const u64 idx = std::accumulate(cnt_by_shard.begin(), cnt_by_shard.begin() + static_cast<std::ptrdiff_t>(s), static_cast<u64>(0)) +
                    (rank - start);
    idxs.push_back(idx);
    docs.push_back(GetDocByPtr(s, ptr, max_disp_len));
  }

  return SearchDocsResult<Token>{.cnt = fr.cnt, .approx = false, .idxs = std::move(idxs), .docs = std::move(docs)};
}

template <typename Token>
SearchDocsResult<Token> Engine<Token>::SearchDocsCnf(const std::vector<std::vector<std::vector<Token>>>& cnf,
                                                     std::size_t maxnum,
                                                     u64 max_disp_len,
                                                     u64 max_clause_freq,
                                                     u64 max_diff_tokens) const {
  if (maxnum == 0) {
    throw std::invalid_argument("maxnum must be > 0");
  }
  auto r = FindCnf(cnf, max_clause_freq, max_diff_tokens);
  if (r.cnt == 0) {
    return SearchDocsResult<Token>{.cnt = 0, .approx = false, .idxs = {}, .docs = {}};
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<u64> ptr_cnt_by_shard;
  ptr_cnt_by_shard.reserve(num_shards());
  for (const auto& ptrs : r.ptrs_by_shard) {
    ptr_cnt_by_shard.push_back(ptrs.size());
  }
  const u64 ptr_cnt = std::accumulate(ptr_cnt_by_shard.begin(), ptr_cnt_by_shard.end(), static_cast<u64>(0));

  std::vector<u64> idxs;
  std::vector<DocResult<Token>> docs;
  idxs.reserve(maxnum);
  docs.reserve(maxnum);

  for (std::size_t d = 0; d < maxnum; d++) {
    std::discrete_distribution<std::size_t> choose_shard(ptr_cnt_by_shard.begin(), ptr_cnt_by_shard.end());
    const std::size_t s = choose_shard(gen);
    const auto& ptrs = r.ptrs_by_shard[s];
    std::uniform_int_distribution<u64> choose_i(0, ptrs.size() - 1);
    const u64 i = choose_i(gen);
    const u64 ptr = ptrs[i];
    const double percentile =
        static_cast<double>(std::accumulate(ptr_cnt_by_shard.begin(), ptr_cnt_by_shard.begin() + static_cast<std::ptrdiff_t>(s), static_cast<u64>(0)) + i) /
        ptr_cnt;
    const u64 idx = static_cast<u64>(percentile * r.cnt);
    idxs.push_back(idx);
    docs.push_back(GetDocByPtr(s, ptr, max_disp_len));
  }

  return SearchDocsResult<Token>{.cnt = r.cnt, .approx = r.approx, .idxs = std::move(idxs), .docs = std::move(docs)};
}

template <typename Token>
std::size_t Engine<Token>::GetLcpLen(const u8* a, std::size_t len_a, const u8* b, std::size_t len_b) const {
  std::size_t i = 0;
  while (i < len_a && i < len_b && a[i] == b[i]) {
    i++;
  }
  return i;
}

template <typename Token>
void Engine<Token>::ComputeLongestPrefixLenThread(const std::vector<Token>* input_ids,
                                                  std::size_t s,
                                                  std::size_t* out_longest_prefix_len) const {
  const auto& shard = ShardAt(s);
  const u8* input_buf = reinterpret_cast<const u8*>(input_ids->data());
  const u64 num_bytes = static_cast<u64>(input_ids->size() * sizeof(Token));
  std::pair<u64, u64> seg;
  FindThread(s, input_buf, num_bytes, {0, shard.tok_cnt}, &seg);
  const u64 start = seg.first;
  const u64 end = seg.second;
  if (start != end) {
    *out_longest_prefix_len = input_ids->size();
    return;
  }
  *out_longest_prefix_len = 0;
  const u64 lo = (start == 0) ? 0 : (start - 1);
  const u64 hi = std::min(shard.tok_cnt, start + 1);
  for (u64 rank = lo; rank < hi; rank++) {
    const u64 ptr = ConvertRankToPtr(shard, rank);
    const std::size_t prefix_len =
        GetLcpLen(shard.ds.data() + ptr, shard.ds_size - ptr, input_buf, input_ids->size() * sizeof(Token)) / sizeof(Token);
    *out_longest_prefix_len = std::max(*out_longest_prefix_len, prefix_len);
  }
}

template <typename Token>
std::size_t Engine<Token>::ComputeLongestPrefixLen(const std::vector<Token>& input_ids,
                                                   const std::vector<Token>& delim_ids,
                                                   bool enforce_bow) const {
  std::vector<std::size_t> by_shard(num_shards());
  for (std::size_t s = 0; s < num_shards(); s++) {
    ComputeLongestPrefixLenThread(&input_ids, s, &by_shard[s]);
  }
  std::size_t best = 0;
  for (const auto v : by_shard) {
    best = std::max(best, v);
  }

  if (!delim_ids.empty()) {
    for (std::size_t pos = 0; pos + 1 < best; pos++) {
      if (std::find(delim_ids.begin(), delim_ids.end(), input_ids[pos]) != delim_ids.end()) {
        best = pos + 1;
        break;
      }
    }
  }

  if (enforce_bow) {
    while (best > 0) {
      if (best == input_ids.size() || opts_.bow_ids.find(static_cast<u32>(input_ids[best])) != opts_.bow_ids.end()) {
        break;
      }
      best--;
    }
  }

  return best;
}

template <typename Token>
void Engine<Token>::CreativityThread(const std::vector<Token>* input_ids, std::size_t l, std::size_t* out_r) const {
  std::vector<Token> suffix(input_ids->begin() + static_cast<std::ptrdiff_t>(l), input_ids->end());
  const std::vector<Token> delim;
  const std::size_t len = ComputeLongestPrefixLen(suffix, delim, false);
  *out_r = l + len;
}

template <typename Token>
CreativityResult Engine<Token>::Creativity(const std::vector<Token>& input_ids) const {
  std::vector<std::size_t> rs(input_ids.size());
  for (std::size_t l = 0; l < input_ids.size(); l++) {
    CreativityThread(&input_ids, l, &rs[l]);
  }
  return CreativityResult{.rs = std::move(rs)};
}

template <typename Token>
void Engine<Token>::ComputeInterestingSpansThread(
    const std::vector<Token>* input_ids,
    std::size_t l,
    const std::vector<Token>* delim_ids,
    std::size_t min_len,
    std::size_t max_cnt,
    bool enforce_bow,
    std::vector<std::pair<std::pair<std::size_t, std::size_t>, FindResult>>* out_span_find_pairs) const {
  std::vector<Token> suffix(input_ids->begin() + static_cast<std::ptrdiff_t>(l), input_ids->end());
  const std::size_t len = ComputeLongestPrefixLen(suffix, *delim_ids, enforce_bow);
  if (len < min_len) {
    return;
  }
  std::vector<Token> span(input_ids->begin() + static_cast<std::ptrdiff_t>(l),
                          input_ids->begin() + static_cast<std::ptrdiff_t>(l + len));
  auto fr = Find(span);
  if (fr.cnt > max_cnt) {
    return;
  }
  out_span_find_pairs->push_back({{l, l + len}, std::move(fr)});
}

template <typename Token>
std::vector<std::pair<std::pair<std::size_t, std::size_t>, FindResult>> Engine<Token>::ComputeInterestingSpans(
    const std::vector<Token>& input_ids,
    const std::vector<Token>& delim_ids,
    std::size_t min_len,
    std::size_t max_cnt,
    bool enforce_bow) const {
  std::vector<std::vector<std::pair<std::pair<std::size_t, std::size_t>, FindResult>>> by_l(input_ids.size());
  for (std::size_t l_block = 0; l_block < input_ids.size(); l_block += opts_.attribution_block_size) {
    for (std::size_t l = l_block; l < std::min(l_block + opts_.attribution_block_size, input_ids.size()); l++) {
      if (enforce_bow && opts_.bow_ids.find(static_cast<u32>(input_ids[l])) == opts_.bow_ids.end()) {
        continue;
      }
      ComputeInterestingSpansThread(&input_ids, l, &delim_ids, min_len, max_cnt, enforce_bow, &by_l[l]);
    }
  }

  std::vector<std::pair<std::pair<std::size_t, std::size_t>, FindResult>> flat;
  for (auto& v : by_l) {
    flat.insert(flat.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
  }

  std::vector<std::pair<std::pair<std::size_t, std::size_t>, FindResult>> filtered;
  std::size_t last_r = 0;
  for (const auto& p : flat) {
    const auto [l, r] = p.first;
    if (r > last_r) {
      last_r = r;
      filtered.push_back(p);
    }
  }
  return filtered;
}

template <typename Token>
void Engine<Token>::GetAttributionSpanThread(
    const std::pair<std::pair<std::size_t, std::size_t>, FindResult>* span_find_pair,
    AttributionSpan* out_attribution_span) const {
  const auto& span = span_find_pair->first;
  const auto& fr = span_find_pair->second;

  std::vector<std::vector<u64>> ptrs_by_shard(num_shards());
  for (std::size_t s = 0; s < num_shards(); s++) {
    const auto& shard = ShardAt(s);
    const auto [start, end] = fr.segment_by_shard[s];
    ptrs_by_shard[s].reserve(end - start);
    for (u64 rank = start; rank < end; rank++) {
      ptrs_by_shard[s].push_back(ConvertRankToPtr(shard, rank));
    }
  }

  std::vector<AttributionDoc> docs;
  for (std::size_t s = 0; s < num_shards(); s++) {
    for (const auto ptr : ptrs_by_shard[s]) {
      docs.push_back(AttributionDoc{.s = s, .ptr = ptr});
    }
  }

  out_attribution_span->l = span.first;
  out_attribution_span->r = span.second;
  out_attribution_span->length = span.second - span.first;
  out_attribution_span->count = fr.cnt;
  out_attribution_span->docs = std::move(docs);
}

template <typename Token>
AttributionResult Engine<Token>::Attribute(const std::vector<Token>& input_ids,
                                           const std::vector<Token>& delim_ids,
                                           std::size_t min_len,
                                           std::size_t max_cnt,
                                           bool enforce_bow) const {
  auto span_find_pairs = ComputeInterestingSpans(input_ids, delim_ids, min_len, max_cnt, enforce_bow);
  std::vector<AttributionSpan> spans(span_find_pairs.size());
  for (std::size_t i = 0; i < span_find_pairs.size(); i++) {
    GetAttributionSpanThread(&span_find_pairs[i], &spans[i]);
  }
  for (auto& span : spans) {
    double sum = 0.0;
    for (std::size_t i = span.l; i < span.r; i++) {
      const auto it = unigram_logprobs_.find(input_ids[i]);
      sum += (it == unigram_logprobs_.end()) ? -10000.0 : it->second;
    }
    span.unigram_logprob_sum = sum;
  }
  return AttributionResult{.spans = std::move(spans)};
}

template <typename Token>
EngineDiff<Token>::EngineDiff(Index main_index, Index diff_index, EngineOptions opts)
    : Engine<Token>(std::move(main_index), opts) {
  diff_ = std::make_unique<Engine<Token>>(std::move(diff_index), opts);
}

template <typename Token>
std::size_t EngineDiff<Token>::ComputeLongestPrefixLen(const std::vector<Token>& input_ids,
                                                       const std::vector<Token>& delim_ids,
                                                       bool enforce_bow) const {
  std::size_t best = Engine<Token>::ComputeLongestPrefixLen(input_ids, delim_ids, enforce_bow);
  while (best > 0) {
    const auto main_cnt =
        Engine<Token>::Count(std::vector<Token>(input_ids.begin(), input_ids.begin() + static_cast<std::ptrdiff_t>(best))).count;
    const auto diff_cnt =
        diff_->Count(std::vector<Token>(input_ids.begin(), input_ids.begin() + static_cast<std::ptrdiff_t>(best))).count;
    if (main_cnt > diff_cnt) {
      break;
    }
    best--;
  }
  return best;
}

template <typename Token>
std::vector<std::vector<DocResult<Token>>> EngineDiff<Token>::GetDocsByPtrs2Grouped(
    const std::vector<std::tuple<std::vector<std::pair<std::size_t, u64>>, std::vector<Token>, u64, u64>>& requests) const {
  std::vector<std::vector<DocResult<Token>>> docs_by_request(requests.size());
  std::vector<std::vector<DocResult<Token>>> docs_diff_by_request(requests.size());

  for (std::size_t r = 0; r < requests.size(); r++) {
    const auto& request = requests[r];
    std::vector<std::tuple<std::size_t, u64, u64, u64>> main_reqs;
    for (const auto& [s, ptr] : std::get<0>(request)) {
      main_reqs.emplace_back(s, ptr, std::get<2>(request), std::get<3>(request));
    }
    docs_by_request[r] = Engine<Token>::GetDocsByPtrs2(main_reqs);

    const auto& span_ids = std::get<1>(request);
    const auto fr = diff_->Find(span_ids);
    std::vector<std::tuple<std::size_t, u64, u64, u64>> diff_reqs;
    for (std::size_t s = 0; s < diff_->num_shards(); s++) {
      const auto [start, end] = fr.segment_by_shard[s];
      for (u64 rank = start; rank < end; rank++) {
        diff_reqs.emplace_back(s, rank, std::get<2>(request), std::get<3>(request));
      }
    }
    docs_diff_by_request[r] = diff_->GetDocsByRanks2(diff_reqs);
  }

  for (std::size_t r = 0; r < requests.size(); r++) {
    auto& main_docs = docs_by_request[r];
    const auto& diff_docs = docs_diff_by_request[r];
    for (auto& doc_main : main_docs) {
      const bool blocked = std::any_of(diff_docs.begin(), diff_docs.end(),
                                       [&](const DocResult<Token>& doc_diff) { return doc_main.token_ids == doc_diff.token_ids; });
      if (blocked) {
        doc_main.token_ids.clear();
        doc_main.blocked = true;
      }
    }
  }

  return docs_by_request;
}

template class Cursor<u8>;
template class Cursor<u16>;
template class Cursor<u32>;

template class Engine<u8>;
template class Engine<u16>;
template class Engine<u32>;

template class EngineDiff<u8>;
template class EngineDiff<u16>;
template class EngineDiff<u32>;

}  // namespace gram
