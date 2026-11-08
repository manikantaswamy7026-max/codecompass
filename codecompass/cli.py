"""CLI:  python -m codecompass.cli <command> <repo> [args]

No model needed: index, search, stats, find-def, callers, about
Needs Ollama:    ask, explain, impact, onboard, chat
"""
from __future__ import annotations

import argparse

from . import commands
from .agent import run_agent
from .index import CodeIndex
from .llm import OllamaClient
from .retrieve import MODES, search

NEEDS_ARG = {"search": "query", "ask": "query", "find-def": "symbol", "callers": "symbol",
             "explain": "path", "impact": "symbol"}


def _print_run(question, idx, llm, history=None):
    answer = ""
    for ev in run_agent(question, idx, llm, history=history):
        if ev["type"] == "step":
            print(f"  step {ev['step']}: {ev['tool']}({ev['args']})")
        else:
            answer = ev["text"]
            print("\n" + answer)
            for c in ev["citations"]:
                print(("  ✔ " if c["verified"] else "  ✘ ") + c["cite"])
    return answer


def main(argv=None):
    p = argparse.ArgumentParser(prog="codecompass")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("about")
    for name in ("index", "stats", "onboard", "chat", "search", "ask", "find-def", "callers",
                 "explain", "impact"):
        s = sub.add_parser(name)
        s.add_argument("root")
        if name in NEEDS_ARG:
            s.add_argument(NEEDS_ARG[name])
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
    idx_stats = idx.build()
    idx.save()

    if a.cmd == "index":
        print(idx_stats)
    elif a.cmd == "stats":
        print(commands.stats(idx))
    elif a.cmd == "find-def":
        print(commands.find_def(idx, a.symbol))
    elif a.cmd == "callers":
        print(commands.callers(idx, a.symbol))
    elif a.cmd == "search":
        for h in search(idx, a.query, k=a.k, mode=a.mode, llm=llm if live else None):
            print(f"{h.score:.4f}  {h.chunk.id}  [{h.chunk.kind} {h.chunk.name}]  {h.via}")
    else:  # model-backed commands
        if not live:
            raise SystemExit("Ollama is not reachable. Start it (ollama serve) and pull the models.")
        if a.cmd == "ask":
            _print_run(a.query, idx, llm)
        elif a.cmd == "explain":
            _print_run(commands.explain_prompt(a.path), idx, llm)
        elif a.cmd == "impact":
            _print_run(commands.impact_prompt(a.symbol), idx, llm)
        elif a.cmd == "onboard":
            _print_run(commands.onboard_prompt(), idx, llm)
        elif a.cmd == "chat":
            print("CodeCompass chat. Follow-up questions are remembered. Type /quit to exit.")
            history = []
            while True:
                try:
                    q = input("\nyou> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if q in ("/quit", "/exit"):
                    break
                if q:
                    history.append((q, _print_run(q, idx, llm, history)))


if __name__ == "__main__":
    main()
