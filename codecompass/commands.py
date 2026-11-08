"""Commands that need no model (pure index lookups) and prompts for ones that do."""
from __future__ import annotations

from collections import Counter

from .index import CodeIndex


def stats(idx: CodeIndex) -> str:
    langs = Counter(c.lang for c in idx.chunks)
    kinds = Counter(c.kind for c in idx.chunks)
    return (f"{len(idx.hashes)} files, {len(idx.chunks)} chunks\n"
            f"languages: {dict(langs)}\nchunk kinds: {dict(kinds)}\n"
            f"vectors: {'yes' if idx.vecs is not None else 'no (keyword search only)'}")


def find_def(idx: CodeIndex, symbol: str) -> str:
    base = symbol.split(".")[-1]
    hits = [idx.chunks[j] for j in idx.defs.get(base, [])]
    if not hits:
        return f"No definition found for '{symbol}'."
    return "\n".join(f"{c.id}  [{c.kind} {c.name}]" for c in hits)


def callers(idx: CodeIndex, symbol: str) -> str:
    base = symbol.split(".")[-1]
    hits = [idx.chunks[j] for j in idx.callers.get(base, [])]
    if not hits:
        return f"Nothing in the index calls '{symbol}'."
    return "\n".join(f"{c.id}  [{c.kind} {c.name}]" for c in hits)


def explain_prompt(path: str) -> str:
    return (f"Explain what the file {path} does: its purpose, its main functions or classes, and how it "
            "connects to the rest of the codebase. Read the file first and cite line ranges.")


def impact_prompt(symbol: str) -> str:
    return (f"What could break if I change {symbol}? Find its definition, every caller, and any tests "
            "that cover it. List each affected place with file:line citations, and say which callers "
            "appear to have no test coverage.")


def onboard_prompt() -> str:
    return ("I am new to this repository. Give me an architecture overview and list the 8 files I "
            "should read first, in order, with one sentence on why for each. List the directories "
            "first, then read the key files. Cite what you read.")
