import sys
from typing import Iterable, List, Optional, Set, Union

from . import cpp_engine
from .models import (
    AttributionResponse,
    CnfType,
    CountResponse,
    CreativityResponse,
    DocResult,
    ErrorResponse,
    FindCnfResponse,
    FindResponse,
    InfGramNtdResponse,
    InfGramProbResponse,
    NtdResponse,
    ProbResponse,
    QueryIdsType,
    SearchDocsResponse,
)


class GramEngine:
    def __init__(
        self,
        index_dir: Union[Iterable[str], str],
        eos_token_id: int,
        vocab_size: int = 65535,
        version: int = 4,
        token_dtype: str = "u16",
        threads: int = 0,
        bow_ids: Optional[Set[int]] = None,
        attribution_block_size: int = 512,
        precompute_unigram_logprobs: bool = False,
        max_support: int = 1000,
        max_clause_freq: int = 50000,
        max_diff_tokens: int = 100,
        maxnum: int = 1,
        max_disp_len: int = 1000,
    ) -> None:
        assert sys.byteorder == "little"

        if isinstance(index_dir, str):
            index_dirs = [index_dir]
        else:
            index_dirs = list(index_dir)

        if bow_ids is None:
            bow_ids_set: Set[int] = set()
        else:
            bow_ids_set = set(bow_ids)

        if token_dtype == "u8":
            self.token_id_max = 2**8 - 1
            engine_class = cpp_engine.Engine_U8
        elif token_dtype == "u16":
            self.token_id_max = 2**16 - 1
            engine_class = cpp_engine.Engine_U16
        elif token_dtype == "u32":
            self.token_id_max = 2**32 - 1
            engine_class = cpp_engine.Engine_U32
        else:
            raise ValueError("unsupported token_dtype")

        self.max_support = int(max_support)
        self.max_clause_freq = int(max_clause_freq)
        self.max_diff_tokens = int(max_diff_tokens)
        self.maxnum = int(maxnum)
        self.max_disp_len = int(max_disp_len)

        self.engine = engine_class(
            index_dirs,
            int(eos_token_id),
            int(vocab_size),
            int(version),
            int(threads),
            bool(precompute_unigram_logprobs),
            {int(x) for x in bow_ids_set},
            int(attribution_block_size),
        )

    def _check_ids(self, ids: QueryIdsType, allow_empty: bool) -> bool:
        if not isinstance(ids, list):
            return False
        if not allow_empty and len(ids) == 0:
            return False
        for x in ids:
            if not isinstance(x, int):
                return False
            if x < 0 or x > self.token_id_max:
                return False
        return True

    def _check_cnf(self, cnf: CnfType) -> bool:
        if not isinstance(cnf, list) or len(cnf) == 0:
            return False
        for disj in cnf:
            if not isinstance(disj, list) or len(disj) == 0:
                return False
            for term in disj:
                if not self._check_ids(term, allow_empty=False):
                    return False
        return True

    def find(self, input_ids: QueryIdsType) -> Union[FindResponse, ErrorResponse]:
        if not self._check_ids(input_ids, allow_empty=True):
            return {"error": "input_ids must be a list[int]"}
        r = self.engine.find(input_ids=input_ids)
        return {"cnt": r.cnt, "segment_by_shard": r.segment_by_shard}

    def count(self, input_ids: QueryIdsType) -> Union[CountResponse, ErrorResponse]:
        if not self._check_ids(input_ids, allow_empty=True):
            return {"error": "input_ids must be a list[int]"}
        r = self.engine.count(input_ids=input_ids)
        return {"count": r.count, "approx": r.approx}

    def primitive_prob(self, prompt_ids: QueryIdsType, cont_id: int) -> Union[ProbResponse, ErrorResponse]:
        if not self._check_ids(prompt_ids, allow_empty=True):
            return {"error": "prompt_ids must be a list[int]"}
        if not isinstance(cont_id, int) or cont_id < 0 or cont_id > self.token_id_max:
            return {"error": "cont_id out of range"}
        r = self.engine.primitive_prob(prompt_ids=prompt_ids, cont_id=cont_id)
        return {"prompt_cnt": r.prompt_cnt, "cont_cnt": r.cont_cnt, "prob": r.prob}

    def primitive_ntd(self, prompt_ids: QueryIdsType, max_support: Optional[int] = None) -> Union[NtdResponse, ErrorResponse]:
        if max_support is None:
            max_support = self.max_support
        if not isinstance(max_support, int) or max_support <= 0:
            return {"error": "max_support must be > 0"}
        if not self._check_ids(prompt_ids, allow_empty=True):
            return {"error": "prompt_ids must be a list[int]"}
        r = self.engine.primitive_ntd(prompt_ids=prompt_ids, max_support=max_support)
        total = r.prompt_cnt if r.prompt_cnt > 0 else 1
        out = {int(tok): {"cont_cnt": int(cnt), "prob": float(cnt) / total} for tok, cnt in zip(r.tokens, r.counts)}
        return {"prompt_cnt": r.prompt_cnt, "result_by_token_id": out, "approx": r.approx}

    def find_cnf(
        self,
        cnf: CnfType,
        max_clause_freq: Optional[int] = None,
        max_diff_tokens: Optional[int] = None,
    ) -> Union[FindCnfResponse, ErrorResponse]:
        if max_clause_freq is None:
            max_clause_freq = self.max_clause_freq
        if max_diff_tokens is None:
            max_diff_tokens = self.max_diff_tokens
        if not isinstance(max_clause_freq, int) or max_clause_freq <= 0:
            return {"error": "max_clause_freq must be > 0"}
        if not isinstance(max_diff_tokens, int) or max_diff_tokens < 0:
            return {"error": "max_diff_tokens must be >= 0"}
        if not self._check_cnf(cnf):
            return {"error": "cnf must be list[list[list[int]]]"}
        r = self.engine.find_cnf(cnf=cnf, max_clause_freq=max_clause_freq, max_diff_tokens=max_diff_tokens)
        return {"cnt": r.cnt, "approx": r.approx, "ptrs_by_shard": r.ptrs_by_shard}

    def count_cnf(
        self,
        cnf: CnfType,
        max_clause_freq: Optional[int] = None,
        max_diff_tokens: Optional[int] = None,
    ) -> Union[CountResponse, ErrorResponse]:
        r = self.find_cnf(cnf, max_clause_freq=max_clause_freq, max_diff_tokens=max_diff_tokens)
        if "error" in r:
            return r
        return {"count": r["cnt"], "approx": r["approx"]}

    def prob(self, prompt_ids: QueryIdsType, cont_id: int) -> Union[InfGramProbResponse, ErrorResponse]:
        if not self._check_ids(prompt_ids, allow_empty=True):
            return {"error": "prompt_ids must be a list[int]"}
        if not isinstance(cont_id, int) or cont_id < 0 or cont_id > self.token_id_max:
            return {"error": "cont_id out of range"}
        r = self.engine.prob(prompt_ids=prompt_ids, cont_id=cont_id)
        return {"prompt_cnt": r.prompt_cnt, "cont_cnt": r.cont_cnt, "prob": r.prob, "suffix_len": r.suffix_len}

    def ntd(self, prompt_ids: QueryIdsType, max_support: Optional[int] = None) -> Union[InfGramNtdResponse, ErrorResponse]:
        if max_support is None:
            max_support = self.max_support
        if not isinstance(max_support, int) or max_support <= 0:
            return {"error": "max_support must be > 0"}
        if not self._check_ids(prompt_ids, allow_empty=True):
            return {"error": "prompt_ids must be a list[int]"}
        r = self.engine.ntd(prompt_ids=prompt_ids, max_support=max_support)
        total = r.prompt_cnt if r.prompt_cnt > 0 else 1
        out = {int(tok): {"cont_cnt": int(cnt), "prob": float(cnt) / total} for tok, cnt in zip(r.tokens, r.counts)}
        return {"prompt_cnt": r.prompt_cnt, "result_by_token_id": out, "approx": r.approx, "suffix_len": r.suffix_len}

    def search_docs(
        self, input_ids: QueryIdsType, maxnum: Optional[int] = None, max_disp_len: Optional[int] = None
    ) -> Union[SearchDocsResponse, ErrorResponse]:
        if maxnum is None:
            maxnum = self.maxnum
        if max_disp_len is None:
            max_disp_len = self.max_disp_len
        if not isinstance(maxnum, int) or maxnum <= 0:
            return {"error": "maxnum must be > 0"}
        if not isinstance(max_disp_len, int) or max_disp_len <= 0:
            return {"error": "max_disp_len must be > 0"}
        if not self._check_ids(input_ids, allow_empty=True):
            return {"error": "input_ids must be a list[int]"}

        r = self.engine.search_docs(input_ids=input_ids, maxnum=maxnum, max_disp_len=max_disp_len)
        docs: List[DocResult] = [
            {
                "doc_ix": d.doc_ix,
                "doc_len": d.doc_len,
                "disp_len": d.disp_len,
                "needle_offset": d.needle_offset,
                "metadata": d.metadata,
                "token_ids": d.token_ids,
                "blocked": d.blocked,
            }
            for d in r.docs
        ]
        return {"cnt": r.cnt, "approx": r.approx, "idxs": r.idxs, "documents": docs}

    def search_docs_cnf(
        self,
        cnf: CnfType,
        maxnum: Optional[int] = None,
        max_disp_len: Optional[int] = None,
        max_clause_freq: Optional[int] = None,
        max_diff_tokens: Optional[int] = None,
    ) -> Union[SearchDocsResponse, ErrorResponse]:
        if maxnum is None:
            maxnum = self.maxnum
        if max_disp_len is None:
            max_disp_len = self.max_disp_len
        if max_clause_freq is None:
            max_clause_freq = self.max_clause_freq
        if max_diff_tokens is None:
            max_diff_tokens = self.max_diff_tokens
        if not isinstance(maxnum, int) or maxnum <= 0:
            return {"error": "maxnum must be > 0"}
        if not isinstance(max_disp_len, int) or max_disp_len <= 0:
            return {"error": "max_disp_len must be > 0"}
        if not isinstance(max_clause_freq, int) or max_clause_freq <= 0:
            return {"error": "max_clause_freq must be > 0"}
        if not isinstance(max_diff_tokens, int) or max_diff_tokens < 0:
            return {"error": "max_diff_tokens must be >= 0"}
        if not self._check_cnf(cnf):
            return {"error": "cnf must be list[list[list[int]]]"}

        r = self.engine.search_docs_cnf(
            cnf=cnf,
            maxnum=maxnum,
            max_disp_len=max_disp_len,
            max_clause_freq=max_clause_freq,
            max_diff_tokens=max_diff_tokens,
        )
        docs: List[DocResult] = [
            {
                "doc_ix": d.doc_ix,
                "doc_len": d.doc_len,
                "disp_len": d.disp_len,
                "needle_offset": d.needle_offset,
                "metadata": d.metadata,
                "token_ids": d.token_ids,
                "blocked": d.blocked,
            }
            for d in r.docs
        ]
        return {"cnt": r.cnt, "approx": r.approx, "idxs": r.idxs, "documents": docs}

    def creativity(self, input_ids: QueryIdsType) -> Union[CreativityResponse, ErrorResponse]:
        if not self._check_ids(input_ids, allow_empty=True):
            return {"error": "input_ids must be a list[int]"}
        r = self.engine.creativity(input_ids=input_ids)
        return {"rs": list(r.rs)}

    def attribute(
        self, input_ids: QueryIdsType, delim_ids: QueryIdsType, min_len: int, max_cnt: int, enforce_bow: bool
    ) -> Union[AttributionResponse, ErrorResponse]:
        if not self._check_ids(input_ids, allow_empty=True):
            return {"error": "input_ids must be a list[int]"}
        if not self._check_ids(delim_ids, allow_empty=True):
            return {"error": "delim_ids must be a list[int]"}
        if not isinstance(min_len, int) or min_len <= 0:
            return {"error": "min_len must be > 0"}
        if not isinstance(max_cnt, int) or max_cnt <= 0:
            return {"error": "max_cnt must be > 0"}
        r = self.engine.attribute(
            input_ids=input_ids, delim_ids=delim_ids, min_len=min_len, max_cnt=max_cnt, enforce_bow=bool(enforce_bow)
        )
        spans = []
        for s in r.spans:
            spans.append(
                {
                    "l": int(s.l),
                    "r": int(s.r),
                    "length": int(s.length),
                    "count": int(s.count),
                    "unigram_logprob_sum": float(s.unigram_logprob_sum),
                    "docs": [{"s": int(d.s), "ptr": int(d.ptr)} for d in s.docs],
                }
            )
        return {"spans": spans}
