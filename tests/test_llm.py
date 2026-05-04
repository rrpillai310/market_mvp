from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from market_mvp.llm import _ollama_model, _provider, complete, extract_json


# ── Provider selection ────────────────────────────────────────────────────────

def test_provider_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert _provider() == "ollama"


def test_provider_reads_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    assert _provider() == "claude"


# ── Ollama model selection ────────────────────────────────────────────────────

def test_ollama_model_returns_qwen_by_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert _ollama_model() == "qwen3.6:latest"


def test_ollama_model_reads_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.3:70b")
    assert _ollama_model() == "llama3.3:70b"


def test_ollama_model_reasoning_returns_deepseek(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL_REASONING", raising=False)
    assert _ollama_model(reasoning=True) == "deepseek-r1:70b"


def test_ollama_model_reasoning_reads_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL_REASONING", "qwen2.5:72b")
    assert _ollama_model(reasoning=True) == "qwen2.5:72b"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_ollama_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_claude_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


# ── extract_json — Ollama path ────────────────────────────────────────────────

def test_extract_json_parses_clean_json(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    payload = {"eps_actual": 1.53, "revenue": 90_000_000}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_ollama_response(json.dumps(payload))

    with patch("market_mvp.llm._ollama_client", return_value=mock_client):
        result = extract_json("extract earnings")
    assert result == payload


def test_extract_json_strips_markdown_fences(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    payload = {"score": 0.7}
    content = f"```json\n{json.dumps(payload)}\n```"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_ollama_response(content)

    with patch("market_mvp.llm._ollama_client", return_value=mock_client):
        result = extract_json("prompt")
    assert result == payload


def test_extract_json_strips_plain_code_fences(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    payload = {"x": 1}
    content = f"```\n{json.dumps(payload)}\n```"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_ollama_response(content)

    with patch("market_mvp.llm._ollama_client", return_value=mock_client):
        result = extract_json("prompt")
    assert result == payload


def test_extract_json_retries_on_failure(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    good_payload = {"ok": True}
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise ConnectionError("simulated WiFi jitter")
        return _mock_ollama_response(json.dumps(good_payload))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("market_mvp.llm._ollama_client", return_value=mock_client):
        with patch("time.sleep"):
            result = extract_json("prompt", retries=3)
    assert result == good_payload
    assert call_count["n"] == 2


def test_extract_json_raises_after_max_retries(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ConnectionError("always fails")

    with patch("market_mvp.llm._ollama_client", return_value=mock_client):
        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="LLM extraction failed"):
                extract_json("prompt", retries=2)


def test_complete_returns_string(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_ollama_response("Hawkish stance.")

    with patch("market_mvp.llm._ollama_client", return_value=mock_client):
        result = complete("summarize this")
    assert isinstance(result, str)
    assert "Hawkish" in result


def test_complete_includes_system_message_when_provided(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_ollama_response("ok")

    with patch("market_mvp.llm._ollama_client", return_value=mock_client):
        complete("prompt", system="You are a finance expert.")

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a finance expert."


# ── extract_json — Claude path ────────────────────────────────────────────────

def test_extract_json_uses_claude_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    payload = {"score": 0.5}
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response(json.dumps(payload))

    with patch("market_mvp.llm._claude_client", return_value=mock_client):
        result = extract_json("prompt")
    assert result == payload


def test_claude_extract_json_strips_markdown_fences(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    payload = {"hawkish": True}
    content = f"```json\n{json.dumps(payload)}\n```"
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response(content)

    with patch("market_mvp.llm._claude_client", return_value=mock_client):
        result = extract_json("prompt")
    assert result == payload


def test_complete_uses_claude_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response("Dovish tone noted.")

    with patch("market_mvp.llm._claude_client", return_value=mock_client):
        result = complete("summarize")
    assert "Dovish" in result


def test_claude_complete_passes_thinking_when_reasoning(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response("analysis")

    with patch("market_mvp.llm._claude_client", return_value=mock_client):
        complete("analyze Fed minutes", reasoning=True)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs.get("thinking") == {"type": "adaptive"}


def test_claude_complete_no_thinking_when_not_reasoning(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response("result")

    with patch("market_mvp.llm._claude_client", return_value=mock_client):
        complete("extract", reasoning=False)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "thinking" not in call_kwargs
