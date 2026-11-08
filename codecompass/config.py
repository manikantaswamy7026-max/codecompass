import os
from dataclasses import dataclass


@dataclass
class Settings:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    chat_model: str = os.getenv("CC_CHAT_MODEL", "qwen2.5-coder:7b")
    embed_model: str = os.getenv("CC_EMBED_MODEL", "nomic-embed-text")
    max_steps: int = int(os.getenv("CC_MAX_STEPS", "6"))


SETTINGS = Settings()
