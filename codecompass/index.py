"""Index: chunks + BM25 + (optional) embeddings + a simple call graph.

Incremental: files are re-chunked only when their SHA-1 changes, and embeddings
are cached by chunk-text hash so unchanged chunks are never re-embedded.
"""
from __future__ import annotations

import hashlib
import math
import os
import pickle
import re
from collections import Counter

import numpy as np

from .chunker import Chunk, LANGS, chunk_file

SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
             ".codecompass", ".idea", ".next", ".mypy_cache", ".pytest_cache"}
MAX_BYTES = 300_000
STOP = set("the a an is are was were do does did how what where which who why when to of in on for "
           "and or it this that with from by as at be can i me my we our show find tell about code".split())


def tokenize(text: str) -> list:
    toks = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        toks.append(w.lower())
        parts = [p.lower() for p in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", w)]
        if len(parts) > 1:
            toks.extend(parts)
    return toks


def query_tokens(q: str) -> list:
    return [t for t in tokenize(q) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.post, self.dl, self.avg, self.n = {}, [], 0.0, 0

    def fit(self, docs):
        self.n = len(docs)
        self.dl = [len(d) for d in docs]
        self.avg = sum(self.dl) / max(1, self.n)
        self.post = {}
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.post.setdefault(t, []).append((i, c))

    def scores(self, q) -> dict:
        s = {}
        for t in set(q):
            p = self.post.get(t)
            if not p:
                continue
            idf = math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for i, tf in p:
                norm = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avg or 1))
                s[i] = s.get(i, 0.0) + idf * tf * (self.k1 + 1) / norm
        return s


class CodeIndex:
    def __init__(self, root, llm=None, ignore=()):
        self.root = os.path.abspath(root)
        self.llm = llm
        self.ignore = set(ignore)  # extra top-level dir names to skip
        self.chunks: list[Chunk] = []
        self.hashes: dict = {}
        self.nlines: dict = {}
        self.vec_cache: dict = {}
        self.vecs = None
        self.bm25 = BM25()
        self.defs: dict = {}
        self.callers: dict = {}

    # ---------- build ----------
    def _scan(self):
        for dp, dns, fns in os.walk(self.root):
            skip = SKIP_DIRS | (self.ignore if dp == self.root else set())
            dns[:] = [d for d in dns if d not in skip]
            for fn in fns:
                ab = os.path.join(dp, fn)
                if os.path.splitext(fn)[1].lower() in LANGS:
                    yield os.path.relpath(ab, self.root).replace(os.sep, "/"), ab

    def build(self, embed=True) -> dict:
        cur = {}
        for rel, ab in self._scan():
            try:
                with open(ab, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            if len(data) > MAX_BYTES or b"\0" in data[:2000]:
                continue
            cur[rel] = (hashlib.sha1(data).hexdigest(), data)
        by_path: dict = {}
        for c in self.chunks:
            by_path.setdefault(c.path, []).append(c)
        new, reindexed = [], 0
        for rel, (h, data) in cur.items():
            if self.hashes.get(rel) == h and rel in by_path:
                new.extend(by_path[rel])
            else:
                src = data.decode("utf-8", "ignore")
                new.extend(chunk_file(rel, src))
                self.nlines[rel] = src.count("\n") + 1
                reindexed += 1
            self.hashes[rel] = h
        for rel in [r for r in self.hashes if r not in cur]:
            self.hashes.pop(rel), self.nlines.pop(rel, None)
        self.chunks = new
        self._reindex()
        embedded = self._embed() if embed else 0
        return {"files": len(cur), "chunks": len(new), "reindexed_files": reindexed,
                "embedded_chunks": embedded, "vectors": self.vecs is not None}

    def _reindex(self):
        docs = []
        for c in self.chunks:
            docs.append(tokenize(f"{c.path} {c.name} {c.name} {c.text}"))
        self.bm25 = BM25()
        self.bm25.fit(docs)
        self.defs, self.callers = {}, {}
        for i, c in enumerate(self.chunks):
            if c.kind in ("function", "method", "class"):
                self.defs.setdefault(c.name.split(".")[-1].split("#")[0], []).append(i)
            for name in c.calls:
                self.callers.setdefault(name, []).append(i)

    @staticmethod
    def _key(c: Chunk) -> str:
        return hashlib.sha1((c.path + "\n" + c.text).encode()).hexdigest()

    def _embed(self) -> int:
        if not self.llm or not self.chunks:
            self.vecs = None
            return 0
        keys = [self._key(c) for c in self.chunks]
        missing = [(k, c) for k, c in zip(keys, self.chunks) if k not in self.vec_cache]
        done = 0
        try:
            for i in range(0, len(missing), 32):
                batch = missing[i:i + 32]
                embs = self.llm.embed([f"{c.path}\n{c.text}"[:2000] for _, c in batch])
                for (k, _), e in zip(batch, embs):
                    v = np.asarray(e, dtype=np.float32)
                    self.vec_cache[k] = v / (np.linalg.norm(v) + 1e-9)
                done += len(batch)
        except Exception:
            self.vecs = None
            return done
        keep = set(keys)
        self.vec_cache = {k: v for k, v in self.vec_cache.items() if k in keep}
        self.vecs = np.stack([self.vec_cache[k] for k in keys])
        return done

    # ---------- persistence ----------
    @property
    def store_path(self):
        return os.path.join(self.root, ".codecompass", "index.pkl")

    def save(self):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        state = {k: getattr(self, k) for k in ("chunks", "hashes", "nlines", "vec_cache")}
        with open(self.store_path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_or_new(cls, root, llm=None, ignore=()):
        idx = cls(root, llm, ignore)
        if os.path.exists(idx.store_path):
            try:
                with open(idx.store_path, "rb") as f:
                    for k, v in pickle.load(f).items():
                        setattr(idx, k, v)
            except Exception:
                pass
        return idx
