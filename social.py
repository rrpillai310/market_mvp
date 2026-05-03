from __future__ import annotations

"""Social sentiment ingestion: StockTwits (free) + Reddit via PRAW.

StockTwits: no auth needed for symbol stream (free public API).
Reddit: requires PRAW credentials (free app registration at reddit.com/prefs/apps).
"""

import argparse
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from market_mvp.db import DB, init_db

STOCKTWITS_API = "https://api.stocktwits.com/api/2"
_HEADERS = {"User-Agent": "market_mvp/1.0 rakesh@rakeshpillai.com"}


# --- StockTwits ---

def _fetch_stocktwits(symbol: str, limit: int = 30) -> list[dict]:
    """Fetch recent messages for a symbol. Returns list of message dicts."""
    url = f"{STOCKTWITS_API}/streams/symbol/{symbol}.json"
    try:
        time.sleep(1.0)  # StockTwits free tier: ~200 req/hour
        r = requests.get(url, params={"limit": limit}, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("messages", [])
    except Exception as e:
        print(f"  [social] StockTwits error for {symbol}: {e}")
        return []


def ingest_stocktwits(con, symbol: str) -> None:
    """Fetch StockTwits stream for symbol and store daily aggregates."""
    messages = _fetch_stocktwits(symbol)
    if not messages:
        return

    daily: dict[date, dict] = {}
    for msg in messages:
        created_at = msg.get("created_at") or ""
        try:
            ts = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            d = ts.date()
        except Exception:
            continue

        sentiment = (msg.get("entities") or {}).get("sentiment") or {}
        label = sentiment.get("basic")  # "Bullish" or "Bearish" or None

        if d not in daily:
            daily[d] = {"bull": 0, "bear": 0, "neutral": 0, "total": 0}
        daily[d]["total"] += 1
        if label == "Bullish":
            daily[d]["bull"] += 1
        elif label == "Bearish":
            daily[d]["bear"] += 1
        else:
            daily[d]["neutral"] += 1

    rows = []
    for d, counts in daily.items():
        total = counts["total"]
        bull_ratio = counts["bull"] / total if total else None
        # sentiment_score: bullish=+1, bearish=-1, weighted average
        sentiment_score = (counts["bull"] - counts["bear"]) / total if total else None
        rows.append({
            "symbol": symbol,
            "date": str(d),
            "source": "stocktwits",
            "bull_ratio": bull_ratio,
            "sentiment_score": sentiment_score,
            "message_count": total,
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    con.register("tmp_social_st", df)
    con.execute(
        """
        INSERT OR REPLACE INTO social_sentiment_daily
        SELECT symbol, CAST(date AS DATE), source, bull_ratio, sentiment_score, CAST(message_count AS INTEGER)
        FROM tmp_social_st
        """
    )
    con.unregister("tmp_social_st")
    print(f"  [social] StockTwits {symbol}: {len(rows)} day(s) stored.")


# --- Reddit ---

def _praw_client():
    """Return a PRAW Reddit client from env vars."""
    try:
        import praw
    except ImportError:
        raise ImportError("Install praw: pip install praw")

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "market_mvp/1.0")

    if not client_id or not client_secret:
        raise RuntimeError("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")

    import praw
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def _simple_sentiment(text: str) -> float:
    """Very lightweight sentiment: positive/negative keyword ratio in [-1, +1]."""
    pos = frozenset([
        "bullish", "buy", "long", "moon", "rally", "breakout", "strong",
        "beat", "surge", "gain", "up", "calls", "growth", "undervalued",
    ])
    neg = frozenset([
        "bearish", "sell", "short", "crash", "drop", "puts", "weak",
        "miss", "fall", "loss", "down", "overvalued", "recession", "dump",
    ])
    words = text.lower().split()
    p = sum(1 for w in words if w in pos)
    n = sum(1 for w in words if w in neg)
    total = p + n
    if total == 0:
        return 0.0
    return (p - n) / total


def ingest_reddit(con, symbol: str, subreddits: list[str] | None = None, limit: int = 50) -> None:
    """Fetch recent Reddit posts mentioning symbol, store daily sentiment."""
    if subreddits is None:
        subreddits = ["wallstreetbets", "investing", "stocks", "options"]

    try:
        reddit = _praw_client()
    except Exception as e:
        print(f"  [social] Reddit skipped: {e}")
        return

    daily: dict[date, dict] = {}
    query = symbol.upper()

    for sub_name in subreddits:
        try:
            sub = reddit.subreddit(sub_name)
            for post in sub.search(query, time_filter="week", limit=limit):
                d = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).date()
                text = f"{post.title} {post.selftext}"
                score = _simple_sentiment(text)

                if d not in daily:
                    daily[d] = {"scores": [], "count": 0}
                daily[d]["scores"].append(score)
                daily[d]["count"] += 1
            time.sleep(1.0)
        except Exception as e:
            print(f"  [social] Reddit error for r/{sub_name}: {e}")

    rows = []
    for d, data in daily.items():
        avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0
        rows.append({
            "symbol": symbol,
            "date": str(d),
            "source": "reddit",
            "bull_ratio": None,
            "sentiment_score": avg_score,
            "message_count": data["count"],
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    con.register("tmp_social_rd", df)
    con.execute(
        """
        INSERT OR REPLACE INTO social_sentiment_daily
        SELECT symbol, CAST(date AS DATE), source, bull_ratio, sentiment_score, CAST(message_count AS INTEGER)
        FROM tmp_social_rd
        """
    )
    con.unregister("tmp_social_rd")
    print(f"  [social] Reddit {symbol}: {len(rows)} day(s) stored.")


def build_social_features(con, symbol: str) -> pd.DataFrame:
    """Aggregate StockTwits + Reddit into daily features for features.py."""
    df = con.execute(
        """
        SELECT date,
               AVG(CASE WHEN source='stocktwits' THEN bull_ratio END) AS stocktwits_bull_ratio,
               AVG(CASE WHEN source='reddit' THEN sentiment_score END) AS reddit_sentiment,
               SUM(message_count) AS total_messages
        FROM social_sentiment_daily
        WHERE symbol=?
        GROUP BY date
        ORDER BY date
        """,
        (symbol,),
    ).df()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Volume ratio vs 20-day rolling mean
    df = df.sort_values("date").reset_index(drop=True)
    df["social_volume_ratio"] = df["total_messages"] / df["total_messages"].rolling(20, min_periods=1).mean()
    return df[["date", "stocktwits_bull_ratio", "reddit_sentiment", "social_volume_ratio"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest StockTwits + Reddit social sentiment.")
    ap.add_argument("--db", default="data/market_mvp.duckdb")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    ap.add_argument("--skip-reddit", action="store_true")
    ns = ap.parse_args()

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    db = DB(path=Path(ns.db))
    con = db.connect()
    init_db(con)

    for sym in ns.symbols:
        print(f"[social] {sym}...")
        ingest_stocktwits(con, sym)
        if not ns.skip_reddit:
            ingest_reddit(con, sym)
    con.execute("CHECKPOINT")
    con.close()
    print("[social] done.")
