"""Pluggable LLM backend.

Set AUDIORAG_LLM=anthropic (default) or =ollama.
- anthropic: needs ANTHROPIC_API_KEY. Defaults to the cheap Haiku model to keep
  cost near zero while you iterate; bump AUDIORAG_MODEL when you want quality.
- ollama: fully free/local. Run `ollama serve` and `ollama pull llama3.1` first.
"""
import json
import os
import urllib.request


def complete(prompt: str) -> str:
    backend = os.getenv("AUDIORAG_LLM")
    if backend == "ollama":
        return _ollama(prompt)
    raise ValueError(f"Unknown AUDIORAG_LLM backend: {backend!r}")


def _ollama(prompt: str) -> str:
    payload = json.dumps({
        "model": os.getenv("OLLAMA_MODEL", "llama3.1"),
        "prompt": prompt,
        "stream": False,
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=payload)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["response"]
