"""CLI:  python -m codecompass.cli index|search|ask <path> ..."""
from __future__ import annotations

import argparse

from .agent import run_agent
from .index import CodeIndex
from .llm import OllamaClient
from .retrieve import MODES, search


def main(argv=None):
    p = argparse.ArgumentParser(prog="codecompass")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("about")
    for name in ("index", "search", "ask"):
        s = sub.add_parser(name)
        s.add_argument("root")
        if name != "index":
            s.add_argument("query")
        if name == "search":
            s.add_argument("--mode", choices=MODES, default="hybrid+graph")
            s.add_argument("-k", type=int, default=8)
    a = p.parse_args(argv)
    if a.cmd == "about":
        print("CodeCompass v0.1.0 - created by Manikanta A M "
              "(https://github.com/manikantaswamy7026-max). MIT License.")
        return

    llm = OllamaClient()
    live = llm.available()
    idx = CodeIndex.load_or_new(a.root, llm if live else None)
    stats = idx.build()
    idx.save()
    if a.cmd == "index":
        print(stats)
        return
    if a.cmd == "search":
        for h in search(idx, a.query, k=a.k, mode=a.mode, llm=llm if live else None):
            print(f"{h.score:.4f}  {h.chunk.id}  [{h.chunk.kind} {h.chunk.name}]  {h.via}")
        return
    if not live:
        raise SystemExit("Ollama is not reachable. Start it (ollama serve) and pull the models.")
    for ev in run_agent(a.query, idx, llm):
        if ev["type"] == "step":
            print(f"step {ev['step']}: {ev['tool']}({ev['args']})")
        else:
            print("\n" + ev["text"])
            for c in ev["citations"]:
                print(("  ✔ " if c["verified"] else "  ✘ ") + c["cite"])


if __name__ == "__main__":
    main()
