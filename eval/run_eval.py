"""Retrieval ablation: which retrieval mode finds the right code?

Usage:  python eval/run_eval.py --root . --questions eval/questions.jsonl -k 5

Each question lists `expect_files` and/or `expect_symbols`. A hit@k means any
expected file or symbol appears in the top-k results. Modes needing embeddings
(vector, hybrid) only run when Ollama is reachable; otherwise they are skipped.
Results are measured on YOUR run, never hard-coded.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from codecompass.index import CodeIndex  # noqa: E402
from codecompass.llm import OllamaClient  # noqa: E402
from codecompass.retrieve import MODES, search  # noqa: E402


def hit(hits, q) -> bool:
    """Strict: when expect_symbols is given, the chunk must BE that symbol.
    File-level matching is used only for questions without expect_symbols."""
    files, syms = set(q.get("expect_files", [])), set(q.get("expect_symbols", []))
    for h in hits:
        c = h.chunk
        if syms:
            if c.path in files and c.name.split(".")[-1].split("#")[0] in syms:
                return True
        elif c.path in files:
            return True
    return False


def mrr(hits, q) -> float:
    for r, h in enumerate(hits, 1):
        if hit([h], q):
            return 1.0 / r
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--questions", default=os.path.join(os.path.dirname(__file__), "questions.jsonl"))
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--ignore", nargs="*", default=["tests", "eval"], help="top-level dirs to skip")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    a = ap.parse_args()

    llm = OllamaClient()
    live = llm.available()
    idx = CodeIndex(a.root, llm if live else None, ignore=a.ignore)
    idx.build()
    qs = [json.loads(line) for line in open(a.questions) if line.strip()]

    modes = [m for m in MODES if live or m != "vector"]
    if not live:
        print("Ollama not reachable: vector mode skipped; hybrid modes run without the vector half.")
    rows = {}
    for m in modes:
        h_at_k = r = 0.0
        for q in qs:
            hits = search(idx, q["q"], k=a.k, mode=m, llm=llm if live else None)
            h_at_k += hit(hits, q)
            r += mrr(hits, q)
        rows[m] = {f"hit@{a.k}": round(h_at_k / len(qs), 3), "mrr": round(r / len(qs), 3)}
    print(f"\n{len(qs)} questions, k={a.k}\n{'mode':<14}{'hit@k':>8}{'mrr':>8}")
    for m, v in rows.items():
        print(f"{m:<14}{v[f'hit@{a.k}']:>8}{v['mrr']:>8}")
    json.dump({"k": a.k, "n": len(qs), "embeddings": live, "results": rows}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
