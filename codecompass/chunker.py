"""Syntax-aware chunking.

Python files are split with the stdlib `ast` module (functions, methods, class
headers, module-level code). Other languages use a boundary-regex splitter and
fall back to overlapping line windows. A tree-sitter backend can be dropped in
behind `chunk_file` without changing the rest of the pipeline.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field

LANGS = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".java": "java", ".go": "go", ".rs": "rust", ".c": "c",
    ".h": "c", ".cpp": "cpp", ".cs": "csharp", ".php": "php", ".rb": "ruby",
    ".kt": "kotlin", ".md": "markdown",
}
MAX_LINES = 120


@dataclass
class Chunk:
    path: str
    start: int
    end: int
    kind: str
    name: str
    text: str
    lang: str
    calls: list = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.path}:{self.start}-{self.end}"


def _py_calls(node) -> list:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return sorted(out)


def chunk_python(path: str, src: str) -> list:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return chunk_window(path, src, "python")
    lines = src.splitlines()
    chunks, covered = [], set()
    funcs = (ast.FunctionDef, ast.AsyncFunctionDef)

    def seg(node):
        s = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        return s, node.end_lineno

    def add(node, kind, name):
        s, e = seg(node)
        chunks.append(Chunk(path, s, e, kind, name, "\n".join(lines[s - 1:e]), "python", _py_calls(node)))
        covered.update(range(s, e + 1))

    for node in tree.body:
        if isinstance(node, funcs):
            add(node, "function", node.name)
        elif isinstance(node, ast.ClassDef):
            methods = [m for m in node.body if isinstance(m, funcs)]
            for m in methods:
                add(m, "method", f"{node.name}.{m.name}")
            s, e = seg(node)
            end = min(seg(m)[0] for m in methods) - 1 if methods else e
            while end > s and not lines[end - 1].strip():
                end -= 1
            if end >= s:
                chunks.append(Chunk(path, s, end, "class", node.name, "\n".join(lines[s - 1:end]), "python"))
                covered.update(range(s, end + 1))
    # module-level leftovers (imports, constants, scripts)
    run = []
    for i in range(1, len(lines) + 2):
        if i <= len(lines) and i not in covered:
            run.append(i)
            continue
        body = [n for n in run if lines[n - 1].strip()]
        if body:
            s, e = body[0], body[-1]
            chunks.append(Chunk(path, s, e, "module", "<module>", "\n".join(lines[s - 1:e]), "python"))
        run = []
    chunks.sort(key=lambda c: c.start)
    return chunks


def chunk_window(path: str, src: str, lang: str, size: int = 50, overlap: int = 10) -> list:
    lines = src.splitlines()
    out, i = [], 0
    while i < len(lines):
        j = min(i + size, len(lines))
        text = "\n".join(lines[i:j])
        if text.strip():
            out.append(Chunk(path, i + 1, j, "window", f"L{i + 1}", text, lang))
        if j == len(lines):
            break
        i = j - overlap
    return out


_BOUND = re.compile(
    r"^ {0,2}(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:(?:public|private|protected|static|final|abstract)\s+)*"
    r"(function|class|interface|def|func|fn|struct|impl|type|enum)\b"
)
_NAME = re.compile(r"\b(?:function|class|interface|def|func|fn|struct|impl|type|enum)\s+(\w+)")


def chunk_generic(path: str, src: str, lang: str) -> list:
    lines = src.splitlines()
    starts = [i for i, ln in enumerate(lines) if _BOUND.match(ln)]
    if not starts:
        return chunk_window(path, src, lang)
    out = []
    if starts[0] > 0 and "\n".join(lines[:starts[0]]).strip():
        out.append(Chunk(path, 1, starts[0], "module", "<module>", "\n".join(lines[:starts[0]]), lang))
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else len(lines)
        while e > s + 1 and not lines[e - 1].strip():
            e -= 1
        m = _NAME.search(lines[s])
        kw = _BOUND.match(lines[s]).group(1)
        kind = "class" if kw in ("class", "interface", "struct", "enum", "impl", "type") else "function"
        out.append(Chunk(path, s + 1, e, kind, m.group(1) if m else f"L{s + 1}", "\n".join(lines[s:e]), lang))
    return out


def _split_big(c: Chunk) -> list:
    if c.end - c.start + 1 <= MAX_LINES:
        return [c]
    lines, parts = c.text.splitlines(), []
    for n, i in enumerate(range(0, len(lines), MAX_LINES)):
        seg = lines[i:i + MAX_LINES]
        parts.append(Chunk(c.path, c.start + i, c.start + i + len(seg) - 1, c.kind,
                           f"{c.name}#{n + 1}", "\n".join(seg), c.lang, c.calls))
    return parts


def chunk_file(path: str, src: str) -> list:
    lang = LANGS.get(os.path.splitext(path)[1].lower())
    if lang is None:
        chunks = chunk_window(path, src, "text")
    elif lang == "python":
        chunks = chunk_python(path, src)
    elif lang == "markdown":
        chunks = chunk_window(path, src, lang)
    else:
        chunks = chunk_generic(path, src, lang)
    return [p for c in chunks for p in _split_big(c)]
