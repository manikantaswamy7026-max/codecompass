"""Minimal Ollama client (stdlib only). Any object with chat()/embed() works."""
from __future__ import annotations

import json
import urllib.request

from .config import SETTINGS


class OllamaClient:
    def __init__(self, url=None, chat_model=None, embed_model=None):
        self.url = (url or SETTINGS.ollama_url).rstrip("/")
        self.chat_model = chat_model or SETTINGS.chat_model
        self.embed_model = embed_model or SETTINGS.embed_model

    def _post(self, path, payload, timeout=180):
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def available(self) -> bool:
        try:
            urllib.request.urlopen(self.url + "/api/tags", timeout=3).read()
            return True
        except Exception:
            return False

    def chat(self, messages, temperature=0.1) -> str:
        out = self._post("/api/chat", {
            "model": self.chat_model, "messages": messages, "stream": False,
            "options": {"temperature": temperature}})
        return out["message"]["content"]

    def embed(self, texts):
        out = self._post("/api/embed", {"model": self.embed_model, "input": texts})
        return out["embeddings"]
