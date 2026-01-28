#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gram/engine.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

template <typename Token>
class PyEngine {
 public:
  PyEngine(const std::vector<std::string>& index_dirs,
           std::uint64_t eos_token_id,
           std::uint64_t vocab_size,
           std::uint64_t version,
           std::size_t threads,
           bool precompute_unigram_logprobs,
           const std::set<std::uint32_t>& bow_ids,
           std::size_t attribution_block_size) {
    gram::IndexConfig cfg;
    cfg.index_dirs = index_dirs;
    cfg.version = version;
    cfg.token_width = sizeof(Token);
    cfg.eos_token_id = eos_token_id;
    cfg.vocab_size = vocab_size;

    gram::Index index(cfg);
    const gram::Status st = index.Load();
    if (!st.ok) {
      throw std::runtime_error(st.message);
    }

    gram::EngineOptions opts;
    opts.thread_count = threads;
    opts.precompute_unigram_logprobs = precompute_unigram_logprobs;
    opts.bow_ids = bow_ids;
    opts.attribution_block_size = attribution_block_size;

    engine_ = std::make_unique<gram::Engine<Token>>(std::move(index), opts);
  }

  gram::FindResult find(const std::vector<Token>& input_ids) const { return engine_->Find(input_ids); }
  gram::FindResult find_with_hint(const std::vector<Token>& input_ids,
                                  const std::vector<std::pair<std::uint64_t, std::uint64_t>>& hint_segment_by_shard) const {
    return engine_->FindWithHint(input_ids, hint_segment_by_shard);
  }
  gram::FindCnfResult find_cnf(const std::vector<std::vector<std::vector<Token>>>& cnf,
                               std::uint64_t max_clause_freq,
                               std::uint64_t max_diff_tokens) const {
    return engine_->FindCnf(cnf, max_clause_freq, max_diff_tokens);
  }
  gram::CountResult count(const std::vector<Token>& input_ids) const { return engine_->Count(input_ids); }
  gram::CountResult count_cnf(const std::vector<std::vector<std::vector<Token>>>& cnf,
                              std::uint64_t max_clause_freq,
                              std::uint64_t max_diff_tokens) const {
    return engine_->CountCnf(cnf, max_clause_freq, max_diff_tokens);
  }
  gram::ProbResult prob(const std::vector<Token>& prompt_ids, Token cont_id) const { return engine_->Prob(prompt_ids, cont_id); }
  gram::DistResult<Token> ntd(const std::vector<Token>& prompt_ids, std::uint64_t max_support) const {
    return engine_->Ntd(prompt_ids, max_support);
  }
  gram::DistResult<Token> ntd_from_segment(std::size_t num_bytes,
                                           const std::vector<std::pair<std::uint64_t, std::uint64_t>>& segment_by_shard,
                                           std::uint64_t max_support) const {
    return engine_->NtdFromSegment(num_bytes, segment_by_shard, max_support);
  }
  gram::InfgramProbResult infgram_prob(const std::vector<Token>& prompt_ids, Token cont_id) const {
    return engine_->InfgramProb(prompt_ids, cont_id);
  }
  gram::InfgramDistResult<Token> infgram_ntd(const std::vector<Token>& prompt_ids, std::uint64_t max_support) const {
    return engine_->InfgramNtd(prompt_ids, max_support);
  }
  gram::SearchDocsResult<Token> search_docs(const std::vector<Token>& input_ids, std::size_t maxnum, std::uint64_t max_disp_len) const {
    return engine_->SearchDocs(input_ids, maxnum, max_disp_len);
  }
  gram::SearchDocsResult<Token> search_docs_cnf(const std::vector<std::vector<std::vector<Token>>>& cnf,
                                                std::size_t maxnum,
                                                std::uint64_t max_disp_len,
                                                std::uint64_t max_clause_freq,
                                                std::uint64_t max_diff_tokens) const {
    return engine_->SearchDocsCnf(cnf, maxnum, max_disp_len, max_clause_freq, max_diff_tokens);
  }
  gram::DocResult<Token> get_doc_by_rank(std::size_t s, std::uint64_t rank, std::uint64_t max_disp_len) const {
    return engine_->GetDocByRank(s, rank, max_disp_len);
  }
  std::vector<gram::DocResult<Token>> get_docs_by_ranks(const std::vector<std::pair<std::size_t, std::uint64_t>>& list_of_s_and_rank,
                                                        std::uint64_t max_disp_len) const {
    return engine_->GetDocsByRanks(list_of_s_and_rank, max_disp_len);
  }
  gram::DocResult<Token> get_doc_by_ptr(std::size_t s, std::uint64_t ptr, std::uint64_t max_disp_len) const {
    return engine_->GetDocByPtr(s, ptr, max_disp_len);
  }
  std::vector<gram::DocResult<Token>> get_docs_by_ptrs(const std::vector<std::pair<std::size_t, std::uint64_t>>& list_of_s_and_ptr,
                                                       std::uint64_t max_disp_len) const {
    return engine_->GetDocsByPtrs(list_of_s_and_ptr, max_disp_len);
  }
  gram::DocResult<Token> get_doc_by_ix(std::uint64_t doc_ix, std::uint64_t max_disp_len) const { return engine_->GetDocByIx(doc_ix, max_disp_len); }
  std::vector<gram::DocResult<Token>> get_docs_by_ixs(const std::vector<std::uint64_t>& list_of_doc_ix, std::uint64_t max_disp_len) const {
    return engine_->GetDocsByIxs(list_of_doc_ix, max_disp_len);
  }
  std::size_t get_num_shards() const { return engine_->num_shards(); }
  std::uint64_t get_tok_cnt(std::size_t s) const { return engine_->tok_cnt(s); }
  std::uint64_t get_ds_size(std::size_t s) const { return engine_->ds_size(s); }
  std::uint64_t get_total_doc_cnt() const { return engine_->TotalDocCnt(); }
  std::map<Token, std::uint64_t> compute_unigram_counts(std::size_t s) const { return engine_->ComputeUnigramCounts(s); }
  gram::CreativityResult creativity(const std::vector<Token>& input_ids) const { return engine_->Creativity(input_ids); }
  gram::AttributionResult attribute(const std::vector<Token>& input_ids,
                                    const std::vector<Token>& delim_ids,
                                    std::size_t min_len,
                                    std::size_t max_cnt,
                                    bool enforce_bow) const {
    return engine_->Attribute(input_ids, delim_ids, min_len, max_cnt, enforce_bow);
  }

  gram::Cursor<Token> cursor() const { return engine_->MakeCursor(); }

 private:
  std::unique_ptr<gram::Engine<Token>> engine_;
};

}  // namespace

PYBIND11_MODULE(cpp_engine, m) {
  py::class_<gram::FindResult>(m, "FindResult").def_readwrite("cnt", &gram::FindResult::cnt).def_readwrite(
      "segment_by_shard", &gram::FindResult::segment_by_shard);

  py::class_<gram::FindCnfResult>(m, "FindCnfResult")
      .def_readwrite("cnt", &gram::FindCnfResult::cnt)
      .def_readwrite("approx", &gram::FindCnfResult::approx)
      .def_readwrite("ptrs_by_shard", &gram::FindCnfResult::ptrs_by_shard);

  py::class_<gram::CountResult>(m, "CountResult")
      .def_readwrite("count", &gram::CountResult::count)
      .def_readwrite("approx", &gram::CountResult::approx);

  py::class_<gram::ProbResult>(m, "ProbResult")
      .def_readwrite("prompt_cnt", &gram::ProbResult::prompt_cnt)
      .def_readwrite("cont_cnt", &gram::ProbResult::cont_cnt)
      .def_readwrite("prob", &gram::ProbResult::prob);

  py::class_<gram::DistTokenResult>(m, "DistTokenResult")
      .def_readwrite("cont_cnt", &gram::DistTokenResult::cont_cnt)
      .def_readwrite("prob", &gram::DistTokenResult::prob);

  py::class_<gram::DistResult<gram::u8>>(m, "DistResult_U8")
      .def_readwrite("prompt_cnt", &gram::DistResult<gram::u8>::prompt_cnt)
      .def_readwrite("result_by_token_id", &gram::DistResult<gram::u8>::result_by_token_id)
      .def_readwrite("approx", &gram::DistResult<gram::u8>::approx);

  py::class_<gram::DistResult<gram::u16>>(m, "DistResult_U16")
      .def_readwrite("prompt_cnt", &gram::DistResult<gram::u16>::prompt_cnt)
      .def_readwrite("result_by_token_id", &gram::DistResult<gram::u16>::result_by_token_id)
      .def_readwrite("approx", &gram::DistResult<gram::u16>::approx);

  py::class_<gram::DistResult<gram::u32>>(m, "DistResult_U32")
      .def_readwrite("prompt_cnt", &gram::DistResult<gram::u32>::prompt_cnt)
      .def_readwrite("result_by_token_id", &gram::DistResult<gram::u32>::result_by_token_id)
      .def_readwrite("approx", &gram::DistResult<gram::u32>::approx);

  py::class_<gram::InfgramProbResult>(m, "InfgramProbResult")
      .def_readwrite("prompt_cnt", &gram::InfgramProbResult::prompt_cnt)
      .def_readwrite("cont_cnt", &gram::InfgramProbResult::cont_cnt)
      .def_readwrite("prob", &gram::InfgramProbResult::prob)
      .def_readwrite("suffix_len", &gram::InfgramProbResult::suffix_len);

  py::class_<gram::InfgramDistResult<gram::u8>>(m, "InfgramDistResult_U8")
      .def_readwrite("prompt_cnt", &gram::InfgramDistResult<gram::u8>::prompt_cnt)
      .def_readwrite("result_by_token_id", &gram::InfgramDistResult<gram::u8>::result_by_token_id)
      .def_readwrite("approx", &gram::InfgramDistResult<gram::u8>::approx)
      .def_readwrite("suffix_len", &gram::InfgramDistResult<gram::u8>::suffix_len);

  py::class_<gram::InfgramDistResult<gram::u16>>(m, "InfgramDistResult_U16")
      .def_readwrite("prompt_cnt", &gram::InfgramDistResult<gram::u16>::prompt_cnt)
      .def_readwrite("result_by_token_id", &gram::InfgramDistResult<gram::u16>::result_by_token_id)
      .def_readwrite("approx", &gram::InfgramDistResult<gram::u16>::approx)
      .def_readwrite("suffix_len", &gram::InfgramDistResult<gram::u16>::suffix_len);

  py::class_<gram::InfgramDistResult<gram::u32>>(m, "InfgramDistResult_U32")
      .def_readwrite("prompt_cnt", &gram::InfgramDistResult<gram::u32>::prompt_cnt)
      .def_readwrite("result_by_token_id", &gram::InfgramDistResult<gram::u32>::result_by_token_id)
      .def_readwrite("approx", &gram::InfgramDistResult<gram::u32>::approx)
      .def_readwrite("suffix_len", &gram::InfgramDistResult<gram::u32>::suffix_len);

  py::class_<gram::DocResult<gram::u8>>(m, "DocResult_U8")
      .def_readwrite("doc_ix", &gram::DocResult<gram::u8>::doc_ix)
      .def_readwrite("doc_len", &gram::DocResult<gram::u8>::doc_len)
      .def_readwrite("disp_len", &gram::DocResult<gram::u8>::disp_len)
      .def_readwrite("needle_offset", &gram::DocResult<gram::u8>::needle_offset)
      .def_readwrite("metadata", &gram::DocResult<gram::u8>::metadata)
      .def_readwrite("token_ids", &gram::DocResult<gram::u8>::token_ids)
      .def_readwrite("blocked", &gram::DocResult<gram::u8>::blocked);

  py::class_<gram::DocResult<gram::u16>>(m, "DocResult_U16")
      .def_readwrite("doc_ix", &gram::DocResult<gram::u16>::doc_ix)
      .def_readwrite("doc_len", &gram::DocResult<gram::u16>::doc_len)
      .def_readwrite("disp_len", &gram::DocResult<gram::u16>::disp_len)
      .def_readwrite("needle_offset", &gram::DocResult<gram::u16>::needle_offset)
      .def_readwrite("metadata", &gram::DocResult<gram::u16>::metadata)
      .def_readwrite("token_ids", &gram::DocResult<gram::u16>::token_ids)
      .def_readwrite("blocked", &gram::DocResult<gram::u16>::blocked);

  py::class_<gram::DocResult<gram::u32>>(m, "DocResult_U32")
      .def_readwrite("doc_ix", &gram::DocResult<gram::u32>::doc_ix)
      .def_readwrite("doc_len", &gram::DocResult<gram::u32>::doc_len)
      .def_readwrite("disp_len", &gram::DocResult<gram::u32>::disp_len)
      .def_readwrite("needle_offset", &gram::DocResult<gram::u32>::needle_offset)
      .def_readwrite("metadata", &gram::DocResult<gram::u32>::metadata)
      .def_readwrite("token_ids", &gram::DocResult<gram::u32>::token_ids)
      .def_readwrite("blocked", &gram::DocResult<gram::u32>::blocked);

  py::class_<gram::SearchDocsResult<gram::u8>>(m, "SearchDocsResult_U8")
      .def_readwrite("cnt", &gram::SearchDocsResult<gram::u8>::cnt)
      .def_readwrite("approx", &gram::SearchDocsResult<gram::u8>::approx)
      .def_readwrite("idxs", &gram::SearchDocsResult<gram::u8>::idxs)
      .def_readwrite("docs", &gram::SearchDocsResult<gram::u8>::docs);

  py::class_<gram::SearchDocsResult<gram::u16>>(m, "SearchDocsResult_U16")
      .def_readwrite("cnt", &gram::SearchDocsResult<gram::u16>::cnt)
      .def_readwrite("approx", &gram::SearchDocsResult<gram::u16>::approx)
      .def_readwrite("idxs", &gram::SearchDocsResult<gram::u16>::idxs)
      .def_readwrite("docs", &gram::SearchDocsResult<gram::u16>::docs);

  py::class_<gram::SearchDocsResult<gram::u32>>(m, "SearchDocsResult_U32")
      .def_readwrite("cnt", &gram::SearchDocsResult<gram::u32>::cnt)
      .def_readwrite("approx", &gram::SearchDocsResult<gram::u32>::approx)
      .def_readwrite("idxs", &gram::SearchDocsResult<gram::u32>::idxs)
      .def_readwrite("docs", &gram::SearchDocsResult<gram::u32>::docs);

  py::class_<gram::CreativityResult>(m, "CreativityResult").def_readwrite("rs", &gram::CreativityResult::rs);

  py::class_<gram::AttributionDoc>(m, "AttributionDoc")
      .def_readwrite("s", &gram::AttributionDoc::s)
      .def_readwrite("ptr", &gram::AttributionDoc::ptr);

  py::class_<gram::AttributionSpan>(m, "AttributionSpan")
      .def_readwrite("l", &gram::AttributionSpan::l)
      .def_readwrite("r", &gram::AttributionSpan::r)
      .def_readwrite("length", &gram::AttributionSpan::length)
      .def_readwrite("count", &gram::AttributionSpan::count)
      .def_readwrite("unigram_logprob_sum", &gram::AttributionSpan::unigram_logprob_sum)
      .def_readwrite("docs", &gram::AttributionSpan::docs);

  py::class_<gram::AttributionResult>(m, "AttributionResult").def_readwrite("spans", &gram::AttributionResult::spans);

  py::class_<gram::Cursor<gram::u8>>(m, "Cursor_U8")
      .def("reset", &gram::Cursor<gram::u8>::Reset, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("cnt", &gram::Cursor<gram::u8>::cnt)
      .def_property_readonly("num_bytes", &gram::Cursor<gram::u8>::num_bytes)
      .def("ntd", &gram::Cursor<gram::u8>::Ntd, py::call_guard<py::gil_scoped_release>(), "max_support"_a)
      .def("advance", &gram::Cursor<gram::u8>::Advance, py::call_guard<py::gil_scoped_release>(), "next_token_id"_a)
      .def("advance_ntd",
           &gram::Cursor<gram::u8>::AdvanceNtd,
           py::call_guard<py::gil_scoped_release>(),
           "next_token_id"_a,
           "max_support"_a);

  py::class_<gram::Cursor<gram::u16>>(m, "Cursor_U16")
      .def("reset", &gram::Cursor<gram::u16>::Reset, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("cnt", &gram::Cursor<gram::u16>::cnt)
      .def_property_readonly("num_bytes", &gram::Cursor<gram::u16>::num_bytes)
      .def("ntd", &gram::Cursor<gram::u16>::Ntd, py::call_guard<py::gil_scoped_release>(), "max_support"_a)
      .def("advance", &gram::Cursor<gram::u16>::Advance, py::call_guard<py::gil_scoped_release>(), "next_token_id"_a)
      .def("advance_ntd",
           &gram::Cursor<gram::u16>::AdvanceNtd,
           py::call_guard<py::gil_scoped_release>(),
           "next_token_id"_a,
           "max_support"_a);

  py::class_<gram::Cursor<gram::u32>>(m, "Cursor_U32")
      .def("reset", &gram::Cursor<gram::u32>::Reset, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("cnt", &gram::Cursor<gram::u32>::cnt)
      .def_property_readonly("num_bytes", &gram::Cursor<gram::u32>::num_bytes)
      .def("ntd", &gram::Cursor<gram::u32>::Ntd, py::call_guard<py::gil_scoped_release>(), "max_support"_a)
      .def("advance", &gram::Cursor<gram::u32>::Advance, py::call_guard<py::gil_scoped_release>(), "next_token_id"_a)
      .def("advance_ntd",
           &gram::Cursor<gram::u32>::AdvanceNtd,
           py::call_guard<py::gil_scoped_release>(),
           "next_token_id"_a,
           "max_support"_a);

  py::class_<PyEngine<gram::u8>>(m, "Engine_U8")
      .def(py::init<const std::vector<std::string>&,
                    std::uint64_t,
                    std::uint64_t,
                    std::uint64_t,
                    std::size_t,
                    bool,
                    const std::set<std::uint32_t>&,
                    std::size_t>(),
           "index_dirs"_a,
           "eos_token_id"_a,
           "vocab_size"_a,
           "version"_a,
           "threads"_a,
           "precompute_unigram_logprobs"_a,
           "bow_ids"_a,
           "attribution_block_size"_a)
      .def("cursor", &PyEngine<gram::u8>::cursor, py::keep_alive<0, 1>())
      .def("find", &PyEngine<gram::u8>::find, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
      .def("find_with_hint",
           &PyEngine<gram::u8>::find_with_hint,
           py::call_guard<py::gil_scoped_release>(),
           "input_ids"_a,
           "hint_segment_by_shard"_a)
      .def("find_cnf", &PyEngine<gram::u8>::find_cnf, py::call_guard<py::gil_scoped_release>(), "cnf"_a, "max_clause_freq"_a, "max_diff_tokens"_a)
      .def("count", &PyEngine<gram::u8>::count, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
      .def("count_cnf",
           &PyEngine<gram::u8>::count_cnf,
           py::call_guard<py::gil_scoped_release>(),
           "cnf"_a,
           "max_clause_freq"_a,
           "max_diff_tokens"_a)
      .def("prob", &PyEngine<gram::u8>::prob, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "cont_id"_a)
      .def("ntd", &PyEngine<gram::u8>::ntd, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "max_support"_a)
      .def("ntd_from_segment",
           &PyEngine<gram::u8>::ntd_from_segment,
           py::call_guard<py::gil_scoped_release>(),
           "num_bytes"_a,
           "segment_by_shard"_a,
           "max_support"_a)
      .def("infgram_prob", &PyEngine<gram::u8>::infgram_prob, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "cont_id"_a)
      .def("infgram_ntd", &PyEngine<gram::u8>::infgram_ntd, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "max_support"_a)
      .def("search_docs",
           &PyEngine<gram::u8>::search_docs,
           py::call_guard<py::gil_scoped_release>(),
           "input_ids"_a,
           "maxnum"_a,
           "max_disp_len"_a)
      .def("search_docs_cnf",
           &PyEngine<gram::u8>::search_docs_cnf,
           py::call_guard<py::gil_scoped_release>(),
           "cnf"_a,
           "maxnum"_a,
           "max_disp_len"_a,
           "max_clause_freq"_a,
           "max_diff_tokens"_a)
      .def("get_doc_by_rank",
           &PyEngine<gram::u8>::get_doc_by_rank,
           py::call_guard<py::gil_scoped_release>(),
           "s"_a,
           "rank"_a,
           "max_disp_len"_a)
      .def("get_docs_by_ranks",
           &PyEngine<gram::u8>::get_docs_by_ranks,
           py::call_guard<py::gil_scoped_release>(),
           "list_of_s_and_rank"_a,
           "max_disp_len"_a)
      .def("get_doc_by_ptr", &PyEngine<gram::u8>::get_doc_by_ptr, py::call_guard<py::gil_scoped_release>(), "s"_a, "ptr"_a, "max_disp_len"_a)
      .def("get_docs_by_ptrs",
           &PyEngine<gram::u8>::get_docs_by_ptrs,
           py::call_guard<py::gil_scoped_release>(),
           "list_of_s_and_ptr"_a,
           "max_disp_len"_a)
      .def("get_doc_by_ix", &PyEngine<gram::u8>::get_doc_by_ix, py::call_guard<py::gil_scoped_release>(), "doc_ix"_a, "max_disp_len"_a)
      .def("get_docs_by_ixs",
           &PyEngine<gram::u8>::get_docs_by_ixs,
           py::call_guard<py::gil_scoped_release>(),
           "list_of_doc_ix"_a,
           "max_disp_len"_a)
      .def("get_num_shards", &PyEngine<gram::u8>::get_num_shards, py::call_guard<py::gil_scoped_release>())
      .def("get_tok_cnt", &PyEngine<gram::u8>::get_tok_cnt, py::call_guard<py::gil_scoped_release>(), "s"_a)
      .def("get_ds_size", &PyEngine<gram::u8>::get_ds_size, py::call_guard<py::gil_scoped_release>(), "s"_a)
      .def("get_total_doc_cnt", &PyEngine<gram::u8>::get_total_doc_cnt, py::call_guard<py::gil_scoped_release>())
      .def("compute_unigram_counts", &PyEngine<gram::u8>::compute_unigram_counts, py::call_guard<py::gil_scoped_release>(), "s"_a)
      .def("creativity", &PyEngine<gram::u8>::creativity, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
      .def("attribute",
           &PyEngine<gram::u8>::attribute,
           py::call_guard<py::gil_scoped_release>(),
           "input_ids"_a,
           "delim_ids"_a,
           "min_len"_a,
           "max_cnt"_a,
           "enforce_bow"_a);

  py::class_<PyEngine<gram::u16>>(m, "Engine_U16")
      .def(py::init<const std::vector<std::string>&,
                    std::uint64_t,
                    std::uint64_t,
                    std::uint64_t,
                    std::size_t,
                    bool,
                    const std::set<std::uint32_t>&,
                    std::size_t>(),
           "index_dirs"_a,
           "eos_token_id"_a,
           "vocab_size"_a,
           "version"_a,
           "threads"_a,
           "precompute_unigram_logprobs"_a,
           "bow_ids"_a,
           "attribution_block_size"_a)
      .def("cursor", &PyEngine<gram::u16>::cursor, py::keep_alive<0, 1>())
      .def("find", &PyEngine<gram::u16>::find, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
      .def("find_with_hint",
           &PyEngine<gram::u16>::find_with_hint,
           py::call_guard<py::gil_scoped_release>(),
           "input_ids"_a,
           "hint_segment_by_shard"_a)
      .def("find_cnf",
           &PyEngine<gram::u16>::find_cnf,
           py::call_guard<py::gil_scoped_release>(),
           "cnf"_a,
           "max_clause_freq"_a,
           "max_diff_tokens"_a)
      .def("count", &PyEngine<gram::u16>::count, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
      .def("count_cnf",
           &PyEngine<gram::u16>::count_cnf,
           py::call_guard<py::gil_scoped_release>(),
           "cnf"_a,
           "max_clause_freq"_a,
           "max_diff_tokens"_a)
      .def("prob", &PyEngine<gram::u16>::prob, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "cont_id"_a)
      .def("ntd", &PyEngine<gram::u16>::ntd, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "max_support"_a)
      .def("ntd_from_segment",
           &PyEngine<gram::u16>::ntd_from_segment,
           py::call_guard<py::gil_scoped_release>(),
           "num_bytes"_a,
           "segment_by_shard"_a,
           "max_support"_a)
      .def("infgram_prob", &PyEngine<gram::u16>::infgram_prob, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "cont_id"_a)
      .def("infgram_ntd", &PyEngine<gram::u16>::infgram_ntd, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "max_support"_a)
      .def("search_docs",
           &PyEngine<gram::u16>::search_docs,
           py::call_guard<py::gil_scoped_release>(),
           "input_ids"_a,
           "maxnum"_a,
           "max_disp_len"_a)
      .def("search_docs_cnf",
           &PyEngine<gram::u16>::search_docs_cnf,
           py::call_guard<py::gil_scoped_release>(),
           "cnf"_a,
           "maxnum"_a,
           "max_disp_len"_a,
           "max_clause_freq"_a,
           "max_diff_tokens"_a)
      .def("get_doc_by_rank",
           &PyEngine<gram::u16>::get_doc_by_rank,
           py::call_guard<py::gil_scoped_release>(),
           "s"_a,
           "rank"_a,
           "max_disp_len"_a)
      .def("get_docs_by_ranks",
           &PyEngine<gram::u16>::get_docs_by_ranks,
           py::call_guard<py::gil_scoped_release>(),
           "list_of_s_and_rank"_a,
           "max_disp_len"_a)
      .def("get_doc_by_ptr",
           &PyEngine<gram::u16>::get_doc_by_ptr,
           py::call_guard<py::gil_scoped_release>(),
           "s"_a,
           "ptr"_a,
           "max_disp_len"_a)
      .def("get_docs_by_ptrs",
           &PyEngine<gram::u16>::get_docs_by_ptrs,
           py::call_guard<py::gil_scoped_release>(),
           "list_of_s_and_ptr"_a,
           "max_disp_len"_a)
      .def("get_doc_by_ix", &PyEngine<gram::u16>::get_doc_by_ix, py::call_guard<py::gil_scoped_release>(), "doc_ix"_a, "max_disp_len"_a)
      .def("get_docs_by_ixs",
           &PyEngine<gram::u16>::get_docs_by_ixs,
           py::call_guard<py::gil_scoped_release>(),
           "list_of_doc_ix"_a,
           "max_disp_len"_a)
      .def("get_num_shards", &PyEngine<gram::u16>::get_num_shards, py::call_guard<py::gil_scoped_release>())
      .def("get_tok_cnt", &PyEngine<gram::u16>::get_tok_cnt, py::call_guard<py::gil_scoped_release>(), "s"_a)
      .def("get_ds_size", &PyEngine<gram::u16>::get_ds_size, py::call_guard<py::gil_scoped_release>(), "s"_a)
      .def("get_total_doc_cnt", &PyEngine<gram::u16>::get_total_doc_cnt, py::call_guard<py::gil_scoped_release>())
      .def("compute_unigram_counts", &PyEngine<gram::u16>::compute_unigram_counts, py::call_guard<py::gil_scoped_release>(), "s"_a)
      .def("creativity", &PyEngine<gram::u16>::creativity, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
      .def("attribute",
           &PyEngine<gram::u16>::attribute,
           py::call_guard<py::gil_scoped_release>(),
           "input_ids"_a,
           "delim_ids"_a,
           "min_len"_a,
           "max_cnt"_a,
           "enforce_bow"_a);

  py::class_<PyEngine<gram::u32>>(m, "Engine_U32")
      .def(py::init<const std::vector<std::string>&,
                    std::uint64_t,
                    std::uint64_t,
                    std::uint64_t,
                    std::size_t,
                    bool,
                    const std::set<std::uint32_t>&,
                    std::size_t>(),
           "index_dirs"_a,
           "eos_token_id"_a,
           "vocab_size"_a,
           "version"_a,
           "threads"_a,
           "precompute_unigram_logprobs"_a,
           "bow_ids"_a,
           "attribution_block_size"_a)
      .def("cursor", &PyEngine<gram::u32>::cursor, py::keep_alive<0, 1>())
      .def("find", &PyEngine<gram::u32>::find, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
      .def("find_with_hint",
           &PyEngine<gram::u32>::find_with_hint,
           py::call_guard<py::gil_scoped_release>(),
           "input_ids"_a,
           "hint_segment_by_shard"_a)
      .def("find_cnf",
           &PyEngine<gram::u32>::find_cnf,
           py::call_guard<py::gil_scoped_release>(),
           "cnf"_a,
           "max_clause_freq"_a,
           "max_diff_tokens"_a)
      .def("count", &PyEngine<gram::u32>::count, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
      .def("count_cnf",
           &PyEngine<gram::u32>::count_cnf,
           py::call_guard<py::gil_scoped_release>(),
           "cnf"_a,
           "max_clause_freq"_a,
           "max_diff_tokens"_a)
      .def("prob", &PyEngine<gram::u32>::prob, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "cont_id"_a)
      .def("ntd", &PyEngine<gram::u32>::ntd, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "max_support"_a)
      .def("ntd_from_segment",
           &PyEngine<gram::u32>::ntd_from_segment,
           py::call_guard<py::gil_scoped_release>(),
           "num_bytes"_a,
           "segment_by_shard"_a,
           "max_support"_a)
      .def("infgram_prob", &PyEngine<gram::u32>::infgram_prob, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "cont_id"_a)
      .def("infgram_ntd", &PyEngine<gram::u32>::infgram_ntd, py::call_guard<py::gil_scoped_release>(), "prompt_ids"_a, "max_support"_a)
      .def("search_docs",
           &PyEngine<gram::u32>::search_docs,
           py::call_guard<py::gil_scoped_release>(),
           "input_ids"_a,
           "maxnum"_a,
           "max_disp_len"_a)
      .def("search_docs_cnf",
           &PyEngine<gram::u32>::search_docs_cnf,
           py::call_guard<py::gil_scoped_release>(),
           "cnf"_a,
           "maxnum"_a,
           "max_disp_len"_a,
           "max_clause_freq"_a,
           "max_diff_tokens"_a)
      .def("get_doc_by_rank",
           &PyEngine<gram::u32>::get_doc_by_rank,
           py::call_guard<py::gil_scoped_release>(),
           "s"_a,
           "rank"_a,
           "max_disp_len"_a)
      .def("get_docs_by_ranks",
           &PyEngine<gram::u32>::get_docs_by_ranks,
           py::call_guard<py::gil_scoped_release>(),
           "list_of_s_and_rank"_a,
           "max_disp_len"_a)
      .def("get_doc_by_ptr",
           &PyEngine<gram::u32>::get_doc_by_ptr,
           py::call_guard<py::gil_scoped_release>(),
           "s"_a,
           "ptr"_a,
           "max_disp_len"_a)
      .def("get_docs_by_ptrs",
           &PyEngine<gram::u32>::get_docs_by_ptrs,
           py::call_guard<py::gil_scoped_release>(),
           "list_of_s_and_ptr"_a,
           "max_disp_len"_a)
      .def("get_doc_by_ix", &PyEngine<gram::u32>::get_doc_by_ix, py::call_guard<py::gil_scoped_release>(), "doc_ix"_a, "max_disp_len"_a)
      .def("get_docs_by_ixs",
           &PyEngine<gram::u32>::get_docs_by_ixs,
           py::call_guard<py::gil_scoped_release>(),
           "list_of_doc_ix"_a,
           "max_disp_len"_a)
      .def("get_num_shards", &PyEngine<gram::u32>::get_num_shards, py::call_guard<py::gil_scoped_release>())
      .def("get_tok_cnt", &PyEngine<gram::u32>::get_tok_cnt, py::call_guard<py::gil_scoped_release>(), "s"_a)
      .def("get_ds_size", &PyEngine<gram::u32>::get_ds_size, py::call_guard<py::gil_scoped_release>(), "s"_a)
      .def("get_total_doc_cnt", &PyEngine<gram::u32>::get_total_doc_cnt, py::call_guard<py::gil_scoped_release>())
      .def("compute_unigram_counts", &PyEngine<gram::u32>::compute_unigram_counts, py::call_guard<py::gil_scoped_release>(), "s"_a)
      .def("creativity", &PyEngine<gram::u32>::creativity, py::call_guard<py::gil_scoped_release>(), "input_ids"_a)
      .def("attribute",
           &PyEngine<gram::u32>::attribute,
           py::call_guard<py::gil_scoped_release>(),
           "input_ids"_a,
           "delim_ids"_a,
           "min_len"_a,
           "max_cnt"_a,
           "enforce_bow"_a);
}

