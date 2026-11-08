"""Hybrid retrieval: BM25 + vectors fused with RRF, light rerank, graph expansion.

Modes (used by the eval harness to ablate each idea):
  bm25 | vector | hybrid | hybrid+graph
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunker import Chunk
from .index import CodeIndex, query_tokens, tokenize

MODES = ("bm25", "vector", "hybrid", "hybrid+graph")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    via: str = "search"


def _name_overlap(c: Chunk, qt: list) -> float:
    toks = set(tokenize(c.name + " " + c.path))
    return len(toks & set(qt)) / max(1, len(set(qt)))


def _expand(index: CodeIndex, hits: list, k: int) -> list:
    exp, seen = [], {h.chunk.id for h in hits}
    for h in hits[:3]:
        c = h.chunk
        base = c.name.split(".")[-1].split("#")[0]
        cand = [(j, f"callee of {c.name}") for n in c.calls for j in index.defs.get(n, [])[:2]]
        cand += [(j, f"caller of {c.name}") for j in index.callers.get(base, [])[:2]]
        for j, why in cand:
            e = index.chunks[j]
            if e.id not in seen:
                seen.add(e.id)
                exp.append(Hit(e, h.score * 0.5, why))
    exp.sort(key=lambda x: -x.score)
    exp = exp[:min(3, max(0, k - 1))]
    return hits[:k - len(exp)] + exp


def search(index: CodeIndex, query: str, k: int = 8, mode: str = "hybrid+graph", llm=None) -> list:
    llm = llm or index.llm
    qt = query_tokens(query)
    bm = index.bm25.scores(qt)
    bm_rank = sorted(bm, key=bm.get, reverse=True)[:50]
    vec_rank = []
    if mode != "bm25" and index.vecs is not None and llm is not None:
        try:
            q = np.asarray(llm.embed([query])[0], dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-9
            vec_rank = [int(i) for i in np.argsort(-(index.vecs @ q))[:50]]
        except Exception:
            vec_rank = []
    lists = {"bm25": [bm_rank], "vector": [vec_rank]}.get(mode, [bm_rank, vec_rank])
    scores: dict = {}
    for lst in lists:
        for r, i in enumerate(lst):
            scores[i] = scores.get(i, 0.0) + 1.0 / (60 + r + 1)
    if mode.startswith("hybrid"):
        for i in scores:
            c = index.chunks[i]
            scores[i] += 0.02 * _name_overlap(c, qt) - (0.004 if c.kind == "module" else 0.0)
    top = sorted(scores, key=scores.get, reverse=True)
    hits = [Hit(index.chunks[i], scores[i]) for i in top[:k]]
    return _expand(index, hits, k) if mode == "hybrid+graph" else hits
