"""Tools the agent can call. Every tool returns plain text and records what the
agent has actually seen, so citations in the final answer can be verified."""
from __future__ import annotations

import os
import re

from .index import CodeIndex
from .retrieve import search

SPECS = """Tools (call exactly one per turn):
- search_code(query, k=6): hybrid search over the codebase; returns path:start-end hits.
- read_file(path, start=1, end=80): read numbered lines (max 120 per call).
- find_references(symbol): where a function/class is called or mentioned.
- go_to_definition(symbol): where a function/class/method is defined.
- list_dir(path="."): list files and folders."""


class ToolBox:
    def __init__(self, index: CodeIndex):
        self.index = index
        self.seen: list = []  # (path, start, end) ranges the agent has been shown

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return f"Unknown tool '{name}'. Available: search_code, read_file, find_references, go_to_definition, list_dir."
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return f"Bad arguments for {name}: {e}"
        except Exception as e:  # tools must never crash the loop
            return f"Tool error: {e}"

    def t_search_code(self, query: str, k: int = 6) -> str:
        hits = search(self.index, query, k=int(k))
        if not hits:
            return "No results."
        out = []
        for h in hits:
            c = h.chunk
            self.seen.append((c.path, c.start, c.end))
            head = " | ".join(ln.strip() for ln in c.text.splitlines()[:3])[:200]
            tag = "" if h.via == "search" else f" ({h.via})"
            out.append(f"{c.id} [{c.kind} {c.name}]{tag}\n    {head}")
        return "\n".join(out)

    def t_read_file(self, path: str, start: int = 1, end: int = 80) -> str:
        full = os.path.abspath(os.path.join(self.index.root, path))
        if not full.startswith(self.index.root + os.sep) or not os.path.isfile(full):
            return f"File not found: {path}"
        start, end = max(1, int(start)), int(end)
        end = min(end, start + 119)
        with open(full, encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        end = min(end, len(lines))
        if start > len(lines):
            return f"{path} has only {len(lines)} lines."
        self.seen.append((path, start, end))
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        return f"{path} lines {start}-{end} of {len(lines)}\n{body}"

    def t_go_to_definition(self, symbol: str) -> str:
        base = symbol.split(".")[-1]
        idxs = self.index.defs.get(base, [])
        if not idxs:
            return f"No definition found for {symbol}."
        out = []
        for j in idxs[:5]:
            c = self.index.chunks[j]
            self.seen.append((c.path, c.start, c.end))
            out.append(f"{c.id} [{c.kind} {c.name}]")
        return "\n".join(out)

    def t_find_references(self, symbol: str) -> str:
        base = symbol.split(".")[-1]
        out = [f"called in {self.index.chunks[j].id} [{self.index.chunks[j].name}]"
               for j in self.index.callers.get(base, [])[:10]]
        pat = re.compile(r"\b" + re.escape(base) + r"\b")
        for c in self.index.chunks:
            if len(out) >= 15:
                break
            for off, ln in enumerate(c.text.splitlines()):
                if pat.search(ln):
                    out.append(f"mentioned at {c.path}:{c.start + off}: {ln.strip()[:120]}")
                    self.seen.append((c.path, c.start + off, c.start + off))
                    break
        return "\n".join(out) or f"No references to {symbol}."

    def t_list_dir(self, path: str = ".") -> str:
        full = os.path.abspath(os.path.join(self.index.root, path))
        if not full.startswith(self.index.root) or not os.path.isdir(full):
            return f"Not a directory: {path}"
        items = sorted(os.listdir(full))
        return "\n".join(i + ("/" if os.path.isdir(os.path.join(full, i)) else "")
                         for i in items if not i.startswith("."))[:2000]
