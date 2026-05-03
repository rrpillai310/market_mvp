from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import OpenAI


def _client() -> OpenAI:
    host = os.getenv("OLLAMA_HOST", "http://spark-1dca.local:11434")
    return OpenAI(base_url=f"{host}/v1", api_key="ollama")


def _default_model(reasoning: bool = False) -> str:
    if reasoning:
        return os.getenv("OLLAMA_MODEL_REASONING", "deepseek-r1:70b")
    return os.getenv("OLLAMA_MODEL", "qwen2.5:72b")


def extract_json(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    reasoning: bool = False,
    retries: int = 3,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Call Ollama and parse the response as JSON. Retries on failure."""
    client = _client()
    model = model or _default_model(reasoning=reasoning)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                timeout=timeout,
            )
            raw = resp.choices[0].message.content or ""
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s backoff
    raise RuntimeError(f"LLM extraction failed after {retries} attempts: {last_err}")


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    reasoning: bool = False,
    timeout: float = 120.0,
) -> str:
    """Call Ollama and return raw text response."""
    client = _client()
    model = model or _default_model(reasoning=reasoning)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        timeout=timeout,
    )
    return resp.choices[0].message.content or ""
