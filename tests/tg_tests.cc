#include "gram/engine.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

void WriteFile(const std::filesystem::path& p, std::string_view bytes) {
  std::ofstream f(p, std::ios::binary);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::uint64_t NaiveCount(const std::vector<std::uint16_t>& ds, const std::vector<std::uint16_t>& query) {
  if (query.empty()) {
    return ds.size();
  }
  std::uint64_t cnt = 0;
  for (std::size_t i = 0; i + query.size() <= ds.size(); i++) {
    bool ok = true;
    for (std::size_t j = 0; j < query.size(); j++) {
      if (ds[i + j] != query[j]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      cnt++;
    }
  }
  return cnt;
}

std::vector<std::uint64_t> NaiveNtdCounts(const std::vector<std::uint16_t>& ds,
                                         std::uint16_t doc_sep,
                                         std::uint16_t eos,
                                         const std::vector<std::uint16_t>& prompt,
                                         std::uint16_t vocab_size) {
  std::vector<std::uint64_t> freq(static_cast<std::size_t>(vocab_size) + 1, 0);
  for (std::size_t i = 0; i + prompt.size() <= ds.size(); i++) {
    bool ok = true;
    for (std::size_t j = 0; j < prompt.size(); j++) {
      if (ds[i + j] != prompt[j]) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      continue;
    }
    std::size_t next = i + prompt.size();
    std::uint16_t tok = eos;
    if (next < ds.size()) {
      tok = ds[next];
      if (tok == doc_sep) {
        tok = eos;
      }
    }
    if (tok <= vocab_size) {
      freq[tok]++;
    }
  }
  return freq;
}

}  // namespace

int main() {
  const auto index_path = std::filesystem::temp_directory_path() / "gram_test_index";
  std::error_code ec;
  std::filesystem::remove_all(index_path, ec);
  std::filesystem::create_directories(index_path);

  const std::uint16_t doc_sep = 0xFFFF;
  const std::uint16_t eos = 0;

  const std::vector<std::uint16_t> ds = {
      doc_sep, 10, 20, 30, 40,
      doc_sep, 20, 30, 50,
  };
  const std::vector<std::uint64_t> offsets = {
      0,
      5 * 2,
  };

  std::vector<std::uint64_t> sa(ds.size());
  for (std::size_t i = 0; i < ds.size(); i++) {
    sa[i] = static_cast<std::uint64_t>(i * 2);
  }

  const auto* ds_bytes = reinterpret_cast<const std::uint8_t*>(ds.data());
  const std::size_t ds_bytes_len = ds.size() * 2;
  std::sort(sa.begin(), sa.end(), [&](std::uint64_t a, std::uint64_t b) {
    const std::size_t la = ds_bytes_len - static_cast<std::size_t>(a);
    const std::size_t lb = ds_bytes_len - static_cast<std::size_t>(b);
    const std::size_t m = std::min(la, lb);
    const int cmp = std::memcmp(ds_bytes + a, ds_bytes + b, m);
    if (cmp != 0) {
      return cmp < 0;
    }
    return la < lb;
  });

  WriteFile(index_path / "tokenized.0",
            {reinterpret_cast<const char*>(ds.data()), ds.size() * sizeof(ds[0])});
  WriteFile(index_path / "table.0",
            {reinterpret_cast<const char*>(sa.data()), sa.size() * sizeof(sa[0])});
  WriteFile(index_path / "offset.0",
            {reinterpret_cast<const char*>(offsets.data()), offsets.size() * sizeof(offsets[0])});

  const std::string index_dir = index_path.string();
  gram::IndexConfig cfg;
  cfg.index_dirs = {index_dir};
  cfg.version = 4;
  cfg.token_width = 2;
  cfg.eos_token_id = 0;
  cfg.vocab_size = 65535;

  gram::Index index(cfg);
  auto st = index.Load();
  if (!st.ok) {
    std::cerr << st.message << "\n";
    return 2;
  }

  gram::EngineOptions opts;
  opts.thread_count = 1;
  gram::Engine<gram::u16> engine(std::move(index), opts);

  std::vector<std::uint16_t> q1 = {20};
  const auto naive1 = NaiveCount(ds, q1);
  const auto fr1 = engine.Find(q1);
  assert(fr1.cnt == naive1);

  std::vector<std::uint16_t> q2 = {20, 30};
  const auto naive2 = NaiveCount(ds, q2);
  const auto fr2 = engine.Find(q2);
  assert(fr2.cnt == naive2);

  const auto pr = engine.Prob(q2, 40);
  assert(pr.prompt_cnt == naive2);

  const auto ntd = engine.Ntd(q2, 1000);
  const auto naive_ntd = NaiveNtdCounts(ds, doc_sep, eos, q2, 65535);
  std::uint64_t total = 0;
  for (const auto& [tok, res] : ntd.result_by_token_id) {
    total += res.cont_cnt;
    assert(res.cont_cnt == naive_ntd[tok]);
  }
  assert(total == naive2);

  assert(engine.TotalDocCnt() == offsets.size());
  assert(!offsets.empty());

  const std::uint64_t doc0_start_byte = offsets[0];
  const std::uint64_t doc0_end_byte = (offsets.size() > 1) ? offsets[1] : (ds.size() * 2);
  const std::size_t doc0_start_tok = static_cast<std::size_t>(doc0_start_byte / 2) + 1;
  const std::size_t doc0_end_tok = static_cast<std::size_t>(doc0_end_byte / 2);
  const std::vector<std::uint16_t> doc0_tokens(ds.begin() + static_cast<std::ptrdiff_t>(doc0_start_tok),
                                               ds.begin() + static_cast<std::ptrdiff_t>(doc0_end_tok));

  const auto doc0 = engine.GetDocByIx(0, 1000);
  assert(doc0.doc_ix == 0);
  assert(doc0.doc_len == doc0_tokens.size());
  assert(doc0.disp_len == doc0_tokens.size());
  assert(doc0.needle_offset == 0);
  assert(doc0.metadata.empty());
  assert(doc0.token_ids == doc0_tokens);

  const auto doc0_by_ptr = engine.GetDocByPtr(0, doc0_start_byte + 2, 1000);
  assert(doc0_by_ptr.doc_ix == 0);
  assert(doc0_by_ptr.needle_offset == 0);
  assert(doc0_by_ptr.token_ids == doc0_tokens);

  const std::vector<std::vector<std::vector<std::uint16_t>>> cnf = {{{q2}}};
  const auto cnf_res = engine.FindCnf(cnf, 1000000, 0);
  assert(cnf_res.cnt == naive2);
  assert(cnf_res.ptrs_by_shard.size() == 1);
  assert(cnf_res.ptrs_by_shard[0].size() == naive2);
  for (const auto ptr : cnf_res.ptrs_by_shard[0]) {
    const std::size_t tok_ix = static_cast<std::size_t>(ptr / 2);
    assert(tok_ix + q2.size() <= ds.size());
    for (std::size_t j = 0; j < q2.size(); j++) {
      assert(ds[tok_ix + j] == q2[j]);
    }
  }

  const auto docs = engine.SearchDocs(q2, 1, 1000);
  assert(docs.cnt == naive2);
  assert(docs.docs.size() == 1);
  const auto& d0 = docs.docs[0];
  assert(d0.needle_offset + q2.size() <= d0.token_ids.size());
  for (std::size_t j = 0; j < q2.size(); j++) {
    assert(d0.token_ids[d0.needle_offset + j] == q2[j]);
  }

  std::cout << "ok\n";
  std::filesystem::remove_all(index_path, ec);
  return 0;
}
