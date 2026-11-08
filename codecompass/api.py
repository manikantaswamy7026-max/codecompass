"""FastAPI server. Streams agent steps as Server-Sent Events for a web UI.

Run:  uvicorn codecompass.api:app --reload --port 8000
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .agent import run_agent
from .index import CodeIndex
from .llm import OllamaClient
from .retrieve import MODES, search

app = FastAPI(title="CodeCompass", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"],
                   allow_methods=["*"], allow_headers=["*"])
llm = OllamaClient()
INDEXES: dict = {}


class IndexReq(BaseModel):
    root: str


@app.get("/")
def chat_ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "chat.html"))


@app.get("/health")
def health():
    return {"ok": True, "ollama": llm.available(), "indexed": list(INDEXES)}


@app.post("/index")
def build_index(req: IndexReq):
    idx = CodeIndex.load_or_new(req.root, llm if llm.available() else None)
    stats = idx.build()
    idx.save()
    INDEXES[idx.root] = idx
    return {"root": idx.root, **stats}


def _get(root: str) -> CodeIndex:
    key = os.path.abspath(root)
    if key not in INDEXES:
        raise HTTPException(404, "Index this root first with POST /index")
    return INDEXES[key]


@app.get("/search")
def search_ep(root: str, q: str, k: int = 8, mode: str = "hybrid+graph"):
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    hits = search(_get(root), q, k=k, mode=mode, llm=llm)
    return [{"id": h.chunk.id, "kind": h.chunk.kind, "name": h.chunk.name,
             "score": round(h.score, 4), "via": h.via} for h in hits]


@app.get("/ask")
def ask_ep(root: str, q: str, history: str = "[]"):
    idx = _get(root)
    try:
        hist = [(str(h[0]), str(h[1])) for h in json.loads(history)][-4:]
    except Exception:
        hist = []

    def stream():
        try:
            for ev in run_agent(q, idx, llm, history=hist):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
