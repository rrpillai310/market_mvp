"""Tests for analyst.py — Claude tool-use agent."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from market_mvp.analyst import _build_tools, analyze


# ── analyze() gating ─────────────────────────────────────────────────────────

def test_analyze_returns_none_when_provider_is_ollama(monkeypatch, con, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert analyze("SPY", 5, con, tmp_path) is None


def test_analyze_returns_none_when_provider_not_set(monkeypatch, con, tmp_path):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert analyze("SPY", 5, con, tmp_path) is None


# ── Tool implementations ──────────────────────────────────────────────────────

def test_get_features_returns_dict_with_expected_keys(loaded_features, tmp_path):
    tools = _build_tools("SPY", 5, loaded_features, tmp_path)
    result = tools["get_features"]()
    assert isinstance(result, dict)
    assert "error" not in result
    assert "rsi_14" in result
    assert "pcr" in result
    assert "fed_net_score" in result


def test_get_features_returns_error_for_missing_symbol(con, tmp_path):
    tools = _build_tools("UNKNOWN", 5, con, tmp_path)
    result = tools["get_features"]()
    assert "error" in result


def test_get_price_history_returns_list(loaded_prices, tmp_path):
    tools = _build_tools("SPY", 5, loaded_prices, tmp_path)
    result = tools["get_price_history"](days=10)
    assert isinstance(result, list)
    assert len(result) <= 10
    assert "date" in result[0]
    assert "close" in result[0]


def test_get_price_history_returns_error_for_missing_symbol(con, tmp_path):
    tools = _build_tools("UNKNOWN", 5, con, tmp_path)
    result = tools["get_price_history"](days=5)
    assert "error" in result


def test_get_fed_score_returns_dict(loaded_features, tmp_path):
    tools = _build_tools("SPY", 5, loaded_features, tmp_path)
    result = tools["get_fed_score"]()
    assert isinstance(result, dict)
    assert "fed_net_score" in result or "error" in result


def test_get_news_sentiment_returns_list(loaded_features, tmp_path):
    tools = _build_tools("SPY", 5, loaded_features, tmp_path)
    result = tools["get_news_sentiment"](days=5)
    assert isinstance(result, list)
    assert len(result) <= 5


def test_get_options_data_returns_list(loaded_features, tmp_path):
    tools = _build_tools("SPY", 5, loaded_features, tmp_path)
    result = tools["get_options_data"](days=5)
    assert isinstance(result, list)
    assert len(result) <= 5


def test_get_prediction_returns_error_when_no_files(con, tmp_path):
    tools = _build_tools("SPY", 5, con, tmp_path)
    result = tools["get_prediction"]()
    assert "error" in result


def test_get_prediction_reads_tft_json(con, tmp_path):
    tft = {"predicted_return": 0.04, "direction": "UP", "p10": 0.01, "p90": 0.07, "val_loss": 0.002}
    (tmp_path / "SPY_h5_tft_pred.json").write_text(json.dumps(tft))
    tools = _build_tools("SPY", 5, con, tmp_path)
    result = tools["get_prediction"]()
    assert result["tft_predicted_return_pct"] == pytest.approx(4.0, abs=0.01)
    assert result["tft_direction"] == "UP"


# ── Full agent loop ───────────────────────────────────────────────────────────

def _make_tool_use_block(tool_name: str, tool_id: str, input_data: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = input_data
    return block


def _make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_response(stop_reason: str, content: list):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content
    return resp


def test_analyze_runs_loop_and_writes_json(monkeypatch, tmp_path, loaded_features):
    monkeypatch.setenv("LLM_PROVIDER", "claude")

    commentary_text = "SPY outlook is cautiously bullish. Models agree on +1.8% over 5 days."

    tool_call = _make_tool_use_block("get_features", "tu_001", {})
    tool_response = _make_response("tool_use", [tool_call])
    final_response = _make_response("end_turn", [_make_text_block(commentary_text)])

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [tool_response, final_response]

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = analyze("SPY", 5, loaded_features, tmp_path)

    assert result is not None
    assert result["symbol"] == "SPY"
    assert result["horizon"] == 5
    assert result["commentary"] == commentary_text
    assert "get_features" in result["signals_used"]
    assert "generated_at" in result

    out_path = tmp_path / "SPY_h5_analyst.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert saved["commentary"] == commentary_text


def test_analyze_calls_multiple_tools_before_commentary(monkeypatch, tmp_path, loaded_features):
    monkeypatch.setenv("LLM_PROVIDER", "claude")

    tool_call_1 = _make_tool_use_block("get_prediction", "tu_001", {})
    tool_call_2 = _make_tool_use_block("get_fed_score", "tu_002", {})
    resp_1 = _make_response("tool_use", [tool_call_1])
    resp_2 = _make_response("tool_use", [tool_call_2])
    resp_3 = _make_response("end_turn", [_make_text_block("Neutral outlook.")])

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [resp_1, resp_2, resp_3]

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = analyze("SPY", 5, loaded_features, tmp_path)

    assert result is not None
    assert "get_prediction" in result["signals_used"]
    assert "get_fed_score" in result["signals_used"]
    assert mock_client.messages.create.call_count == 3


def test_analyze_returns_none_on_claude_error(monkeypatch, tmp_path, con):
    monkeypatch.setenv("LLM_PROVIDER", "claude")

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API unavailable")

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = analyze("SPY", 5, con, tmp_path)

    assert result is None
    assert not (tmp_path / "SPY_h5_analyst.json").exists()


def test_analyze_deduplicates_signals_used(monkeypatch, tmp_path, loaded_features):
    monkeypatch.setenv("LLM_PROVIDER", "claude")

    # Claude calls get_features twice
    tool_call_1 = _make_tool_use_block("get_features", "tu_001", {})
    tool_call_2 = _make_tool_use_block("get_features", "tu_002", {})
    resp_1 = _make_response("tool_use", [tool_call_1])
    resp_2 = _make_response("tool_use", [tool_call_2])
    resp_3 = _make_response("end_turn", [_make_text_block("Outlook: neutral.")])

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [resp_1, resp_2, resp_3]

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = analyze("SPY", 5, loaded_features, tmp_path)

    assert result["signals_used"].count("get_features") == 1
