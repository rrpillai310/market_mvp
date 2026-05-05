"""Market analyst agent using Claude tool use.

Runs after model training. Gives Claude tools to query the DB and model
outputs, then generates a concise qualitative commentary on the outlook.

Only activates when LLM_PROVIDER=claude. Returns None gracefully otherwise
(Ollama models don't have reliable tool use).

Output: {symbol}_h{horizon}_analyst.json in models_dir.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Tool schemas (static — cache-friendly) ────────────────────────────────────

_TOOLS: list[dict] = [
    {
        "name": "get_prediction",
        "description": (
            "Get the LightGBM and TFT model predictions for this symbol and horizon. "
            "Returns predicted return (%), direction (UP/DOWN), TFT confidence interval, "
            "and model accuracy metrics from walk-forward validation."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_features",
        "description": (
            "Get today's 30 feature values: price momentum, volatility, RSI, "
            "moving average ratios, options PCR/VOI, news sentiment, earnings "
            "surprise, Fed hawkish/dovish score, and social sentiment."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_price_history",
        "description": "Get recent daily close prices and volume.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Trading days to look back (default 20)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_fed_score",
        "description": (
            "Get the latest Fed hawkish/dovish sentiment score and days since "
            "the last FOMC meeting. Positive net score = hawkish (rate-hike bias), "
            "negative = dovish (easing bias)."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_news_sentiment",
        "description": "Get recent daily news sentiment scores and article counts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Days to look back (default 10)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_options_data",
        "description": (
            "Get recent put/call ratio (PCR) and volume-to-OI ratio (VOI). "
            "High PCR suggests bearish hedging. Rising VOI signals unusual conviction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Days to look back (default 10)",
                }
            },
            "required": [],
        },
    },
]

_SYSTEM = (
    "You are a quantitative market analyst. You have tools to query the latest "
    "model predictions, market features, price history, Fed sentiment, news, and "
    "options data for the ticker under review. Call whichever tools you need, "
    "reason across the signals, then write a concise 2-4 sentence market commentary. "
    "Focus on whether the quantitative models agree with each other, whether the "
    "macro signals (Fed, news) support or contradict the models, and any notable "
    "outliers. Do not give trading advice. Be specific about numbers."
)


# ── Tool implementations (closures over symbol/horizon/con/models_dir) ────────

def _build_tools(
    symbol: str,
    horizon: int,
    con,
    models_dir: Path,
) -> dict[str, Any]:

    def get_prediction(**_) -> dict:
        result: dict[str, Any] = {}

        metrics_path = models_dir / f"{symbol}_h{horizon}_metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            result["lgbm_dir_accuracy"] = m.get("mean_dir_acc")
            result["lgbm_val_rmse"] = m.get("mean_rmse")

        try:
            from market_mvp.predict import load_and_predict
            pred = load_and_predict(symbol, horizon, con, models_dir)
            if pred is not None:
                result["lgbm_predicted_return_pct"] = round(float(pred) * 100, 3)
                result["lgbm_direction"] = "UP" if pred > 0 else "DOWN"
        except Exception:
            pass

        tft_path = models_dir / f"{symbol}_h{horizon}_tft_pred.json"
        if tft_path.exists():
            tft = json.loads(tft_path.read_text())
            result["tft_predicted_return_pct"] = round(float(tft.get("predicted_return", 0)) * 100, 3)
            result["tft_direction"] = tft.get("direction")
            result["tft_p10_pct"] = round(float(tft.get("p10", 0)) * 100, 3)
            result["tft_p90_pct"] = round(float(tft.get("p90", 0)) * 100, 3)
            result["tft_val_loss"] = tft.get("val_loss")

        return result if result else {"error": "no prediction files found"}

    def get_features(**_) -> dict:
        try:
            row = con.execute(
                "SELECT * FROM features_daily WHERE symbol=? AND horizon=? ORDER BY date DESC LIMIT 1",
                [symbol, horizon],
            ).df()
            if row.empty:
                return {"error": "no features found"}
            d = row.iloc[0].dropna().to_dict()
            return {
                k: round(float(v), 4)
                for k, v in d.items()
                if isinstance(v, (int, float)) and k not in ("horizon",)
            }
        except Exception as e:
            return {"error": str(e)}

    def get_price_history(days: int = 20, **_) -> list | dict:
        try:
            rows = con.execute(
                "SELECT date, close, volume FROM prices_daily "
                "WHERE symbol=? ORDER BY date DESC LIMIT ?",
                [symbol, days],
            ).df()
            if rows.empty:
                return {"error": "no price data"}
            rows["date"] = rows["date"].astype(str)
            return rows.to_dict(orient="records")
        except Exception as e:
            return {"error": str(e)}

    def get_fed_score(**_) -> dict:
        try:
            row = con.execute(
                "SELECT date, fed_hawkish_score, fed_net_score, fed_days_since "
                "FROM features_daily WHERE symbol=? ORDER BY date DESC LIMIT 1",
                [symbol],
            ).df()
            if row.empty:
                return {"error": "no Fed data"}
            d = row.iloc[0].dropna().to_dict()
            return {k: round(float(v), 4) if isinstance(v, float) else v for k, v in d.items()}
        except Exception as e:
            return {"error": str(e)}

    def get_news_sentiment(days: int = 10, **_) -> list | dict:
        try:
            rows = con.execute(
                "SELECT date, news_sent, news_sent_chg_5, news_count "
                "FROM features_daily WHERE symbol=? ORDER BY date DESC LIMIT ?",
                [symbol, days],
            ).df()
            if rows.empty:
                return {"error": "no news data"}
            rows["date"] = rows["date"].astype(str)
            return rows.to_dict(orient="records")
        except Exception as e:
            return {"error": str(e)}

    def get_options_data(days: int = 10, **_) -> list | dict:
        try:
            rows = con.execute(
                "SELECT date, pcr, pcr_chg_5, voi, voi_chg_5 "
                "FROM features_daily WHERE symbol=? ORDER BY date DESC LIMIT ?",
                [symbol, days],
            ).df()
            if rows.empty:
                return {"error": "no options data"}
            rows["date"] = rows["date"].astype(str)
            return rows.to_dict(orient="records")
        except Exception as e:
            return {"error": str(e)}

    return {
        "get_prediction": get_prediction,
        "get_features": get_features,
        "get_price_history": get_price_history,
        "get_fed_score": get_fed_score,
        "get_news_sentiment": get_news_sentiment,
        "get_options_data": get_options_data,
    }


# ── Agent loop ────────────────────────────────────────────────────────────────

def analyze(
    symbol: str,
    horizon: int,
    con,
    models_dir: Path,
) -> dict | None:
    """Run the Claude analyst agent for one symbol/horizon.

    Always uses the Claude SDK (Anthropic tool use). LLM_PROVIDER controls
    the extraction calls in llm.py independently — set LLM_PROVIDER=ollama
    for Fed/extraction and still use Claude here by setting ANTHROPIC_API_KEY.
    Returns None if ANTHROPIC_API_KEY is missing or on any error.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[analyst] ANTHROPIC_API_KEY not set — skipping analyst commentary")
        return None

    try:
        import anthropic
    except ImportError:
        print("[analyst] anthropic package not installed — skipping")
        return None

    client = anthropic.Anthropic()
    model = os.getenv("CLAUDE_MODEL", "claude-opus-4-7")
    tools_impl = _build_tools(symbol, horizon, con, models_dir)

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Analyze the current outlook for {symbol} "
                f"over the next {horizon} trading days."
            ),
        }
    ]

    signals_used: list[str] = []
    commentary = ""

    try:
        for _ in range(10):
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=_TOOLS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "end_turn":
                commentary = next(
                    (b.text for b in resp.content if b.type == "text"), ""
                )
                break

            if resp.stop_reason != "tool_use":
                break

            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                signals_used.append(block.name)
                fn = tools_impl.get(block.name)
                result = fn(**block.input) if fn else {"error": "unknown tool"}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

            messages.append({"role": "user", "content": tool_results})

    except Exception as e:
        print(f"[analyst] Claude error for {symbol} h={horizon}: {e}")
        return None

    if not commentary:
        return None

    output = {
        "symbol": symbol,
        "horizon": horizon,
        "commentary": commentary,
        "signals_used": list(dict.fromkeys(signals_used)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = models_dir / f"{symbol}_h{horizon}_analyst.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[analyst] {symbol} h={horizon}: {commentary[:100]}...")
    return output
