from __future__ import annotations

"""LLM client supporting Ollama (default) and Claude (Anthropic) as providers.

Set LLM_PROVIDER=claude to use the Anthropic API instead of Ollama.
Set LLM_PROVIDER=ollama (default) to use Ollama via its OpenAI-compatible endpoint.
"""

import json
import os
import time
from typing import Any


# ── Provider selection ────────────────────────────────────────────────────────

def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "ollama").lower()


# ── Ollama (OpenAI-compatible) ────────────────────────────────────────────────

def _ollama_client():
    from openai import OpenAI
    host = os.getenv("OLLAMA_HOST", "http://10.0.0.2:11434")
    return OpenAI(base_url=f"{host}/v1", api_key="ollama")


def _ollama_model(reasoning: bool = False) -> str:
    if reasoning:
        return os.getenv("OLLAMA_MODEL_REASONING", "deepseek-r1:70b")
    return os.getenv("OLLAMA_MODEL", "qwen3.6:latest")


def _ollama_extract_json(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    reasoning: bool = False,
    retries: int = 3,
    timeout: float = 120.0,
) -> dict[str, Any]:
    client = _ollama_client()
    model = model or _ollama_model(reasoning=reasoning)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    extra = {} if reasoning else {"think": False}

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                timeout=timeout,
                extra_body=extra,
            )
            raw = resp.choices[0].message.content or ""
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM extraction failed after {retries} attempts: {last_err}")


def _ollama_complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    reasoning: bool = False,
    timeout: float = 120.0,
) -> str:
    client = _ollama_client()
    model = model or _ollama_model(reasoning=reasoning)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    extra = {} if reasoning else {"think": False}
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        timeout=timeout,
        extra_body=extra,
    )
    return resp.choices[0].message.content or ""


# ── Claude (Anthropic) ────────────────────────────────────────────────────────

def _claude_client():
    import anthropic
    return anthropic.Anthropic()


def _claude_model(reasoning: bool = False) -> str:
    if reasoning:
        return os.getenv("CLAUDE_MODEL_REASONING", os.getenv("CLAUDE_MODEL", "claude-opus-4-7"))
    return os.getenv("CLAUDE_MODEL", "claude-opus-4-7")


def _parse_json_from_text(raw: str) -> dict[str, Any]:
    """Strip optional markdown fences and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _claude_extract_json(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    import anthropic
    client = _claude_client()
    model = model or _claude_model(reasoning=False)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        # Cache the system prompt — stable across repeated extraction calls
        kwargs["system"] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.messages.create(**kwargs)
            raw = next((b.text for b in resp.content if b.type == "text"), "")
            return _parse_json_from_text(raw)
        except anthropic.APIError as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Claude extraction failed after {retries} attempts: {last_err}")


def _claude_complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    reasoning: bool = False,
) -> str:
    client = _claude_client()
    model = model or _claude_model(reasoning=reasoning)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        # Cache system prompt — useful for repeated Fed minutes analysis with same instructions
        kwargs["system"] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
    if reasoning:
        kwargs["thinking"] = {"type": "adaptive"}

    resp = client.messages.create(**kwargs)
    return next((b.text for b in resp.content if b.type == "text"), "")


# ── Public API ────────────────────────────────────────────────────────────────

def extract_json(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    reasoning: bool = False,
    retries: int = 3,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Call the active LLM provider and parse the response as JSON."""
    if _provider() == "claude":
        return _claude_extract_json(prompt, system=system, model=model, retries=retries)
    return _ollama_extract_json(
        prompt, system=system, model=model, reasoning=reasoning,
        retries=retries, timeout=timeout,
    )


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    reasoning: bool = False,
    timeout: float = 120.0,
) -> str:
    """Call the active LLM provider and return raw text."""
    if _provider() == "claude":
        return _claude_complete(prompt, system=system, model=model, reasoning=reasoning)
    return _ollama_complete(
        prompt, system=system, model=model, reasoning=reasoning, timeout=timeout,
    )
