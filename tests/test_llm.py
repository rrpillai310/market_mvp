from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from market_mvp.llm import _client, _default_model, complete, extract_json


def test_default_model_returns_qwen_by_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert _default_model() == "qwen2.5:72b"


def test_default_model_reads_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.3:70b")
    assert _default_model() == "llama3.3:70b"


def test_default_model_reasoning_returns_deepseek(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL_REASONING", raising=False)
    assert _default_model(reasoning=True) == "deepseek-r1:70b"


def test_default_model_reasoning_reads_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL_REASONING", "qwen2.5:72b")
    assert _default_model(reasoning=True) == "qwen2.5:72b"


def _mock_completion(content: str):
    """Build a mock openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_extract_json_parses_clean_json():
    payload = {"eps_actual": 1.53, "revenue": 90_000_000}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion(json.dumps(payload))

    with patch("market_mvp.llm._client", return_value=mock_client):
        result = extract_json("extract earnings")
    assert result == payload


def test_extract_json_strips_markdown_fences():
    payload = {"score": 0.7}
    content = f"```json\n{json.dumps(payload)}\n```"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion(content)

    with patch("market_mvp.llm._client", return_value=mock_client):
        result = extract_json("prompt")
    assert result == payload


def test_extract_json_strips_plain_code_fences():
    payload = {"x": 1}
    content = f"```\n{json.dumps(payload)}\n```"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion(content)

    with patch("market_mvp.llm._client", return_value=mock_client):
        result = extract_json("prompt")
    assert result == payload


def test_extract_json_retries_on_failure():
    good_payload = {"ok": True}
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise ConnectionError("simulated WiFi jitter")
        return _mock_completion(json.dumps(good_payload))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("market_mvp.llm._client", return_value=mock_client):
        with patch("time.sleep"):  # don't actually sleep in tests
            result = extract_json("prompt", retries=3)
    assert result == good_payload
    assert call_count["n"] == 2


def test_extract_json_raises_after_max_retries():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ConnectionError("always fails")

    with patch("market_mvp.llm._client", return_value=mock_client):
        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="LLM extraction failed"):
                extract_json("prompt", retries=2)


def test_complete_returns_string():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion("Hawkish stance.")

    with patch("market_mvp.llm._client", return_value=mock_client):
        result = complete("summarize this")
    assert isinstance(result, str)
    assert "Hawkish" in result


def test_complete_includes_system_message_when_provided():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion("ok")

    with patch("market_mvp.llm._client", return_value=mock_client):
        complete("prompt", system="You are a finance expert.")

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a finance expert."
