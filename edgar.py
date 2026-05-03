from __future__ import annotations

"""EDGAR XBRL fast-path earnings ingestion.

Uses data.sec.gov (free, no API key). Covers ~90% of modern filings
via structured XBRL company facts. Playwright/OCR fallback is a future phase.
"""

import argparse
import json
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from market_mvp.db import DB, init_db

EDGAR_API = "https://data.sec.gov"
# SEC requires a descriptive User-Agent with contact info
_HEADERS = {"User-Agent": "market_mvp/1.0 rakesh@rakeshpillai.com"}

# Top holdings by ETF — drives which company earnings we ingest.
# Weighted by approximate index weight; top 20 covers ~40-50% of ETF movement.
ETF_HOLDINGS: dict[str, list[str]] = {
    "SPY": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO",
        "TSLA", "BRK-B", "LLY", "JPM", "V", "UNH", "XOM", "MA", "COST",
        "HD", "NFLX", "AMD",
    ],
    "QQQ": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO",
        "TSLA", "COST", "NFLX", "AMD", "ADBE", "QCOM", "INTC", "MU",
        "INTU", "AMAT", "LRCX", "PYPL",
    ],
}


def _get(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            time.sleep(0.12)  # SEC rate limit: max 10 req/sec
            r = requests.get(url, headers=_HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def lookup_cik(ticker: str, con=None) -> str | None:
    """Return zero-padded 10-digit CIK for a ticker. Caches in edgar_cik_map."""
    # Check DB cache first
    if con is not None:
        row = con.execute(
            "SELECT cik FROM edgar_cik_map WHERE symbol=?", (ticker.upper(),)
        ).fetchone()
        if row:
            return row[0]

    tickers_data = _get(f"{EDGAR_API}/files/company_tickers.json")
    for entry in tickers_data.values():
        if entry.get("ticker", "").upper() == ticker.upper():
            cik = str(entry["cik_str"]).zfill(10)
            if con is not None:
                con.execute(
                    "INSERT OR REPLACE INTO edgar_cik_map(symbol, cik, name) VALUES (?,?,?)",
                    (ticker.upper(), cik, entry.get("title")),
                )
            return cik
    return None


def fetch_company_facts(cik: str) -> dict:
    return _get(f"{EDGAR_API}/api/xbrl/companyfacts/CIK{cik}.json")


def _extract_concept(facts: dict, namespace: str, *concept_names: str, unit: str) -> dict[str, float]:
    """Pull the most recent value per period-end date from XBRL company facts."""
    ns_data = facts.get("facts", {}).get(namespace, {})
    result: dict[str, float] = {}
    for concept in concept_names:
        if concept not in ns_data:
            continue
        units = ns_data[concept].get("units", {})
        items = units.get(unit, [])
        for item in items:
            if item.get("form") not in ("10-Q", "10-K"):
                continue
            end = item.get("end")
            val = item.get("val")
            if end and val is not None:
                # Keep first match per period (concept list is priority-ordered)
                if end not in result:
                    result[end] = float(val)
    return result


def extract_earnings_xbrl(facts: dict, symbol: str) -> list[dict]:
    """Parse XBRL company facts into a list of per-period earnings rows."""
    eps = _extract_concept(
        facts, "us-gaap",
        "EarningsPerShareBasic", "EarningsPerShareDiluted",
        unit="USD/shares",
    )
    rev = _extract_concept(
        facts, "us-gaap",
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        unit="USD",
    )

    periods = sorted(set(list(eps.keys()) + list(rev.keys())))
    rows = []
    for period in periods:
        rows.append({
            "symbol": symbol,
            "period_end": period,
            "eps_actual": eps.get(period),
            "rev_actual": rev.get(period),
        })
    return rows


def ingest_earnings_for_symbol(con, symbol: str, etf_symbol: str | None = None) -> int:
    """Fetch XBRL facts for symbol, store in edgar_earnings. Returns rows written."""
    cik = lookup_cik(symbol, con=con)
    if not cik:
        print(f"  [edgar] No CIK found for {symbol}, skipping.")
        return 0

    try:
        facts = fetch_company_facts(cik)
    except Exception as e:
        print(f"  [edgar] Failed to fetch facts for {symbol} (CIK {cik}): {e}")
        return 0

    rows = extract_earnings_xbrl(facts, symbol)
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    df["filed_date"] = None
    df["form_type"] = "XBRL"
    df["eps_surprise_pct"] = None
    df["rev_surprise_pct"] = None
    df["guidance_text"] = None
    df["guidance_sentiment"] = None
    df["raw_json"] = None

    con.register("tmp_edgar", df)
    con.execute(
        """
        INSERT OR REPLACE INTO edgar_earnings
        SELECT symbol,
               CAST(period_end AS DATE),
               CAST(filed_date AS DATE),
               form_type,
               eps_actual,
               rev_actual,
               eps_surprise_pct,
               rev_surprise_pct,
               guidance_text,
               guidance_sentiment,
               raw_json
        FROM tmp_edgar
        WHERE period_end IS NOT NULL
        """
    )
    con.unregister("tmp_edgar")
    return len(rows)


def ingest_etf_holdings_earnings(con, etf_symbol: str) -> None:
    """Ingest earnings for all top holdings of an ETF."""
    holdings = ETF_HOLDINGS.get(etf_symbol.upper(), [])
    print(f"[edgar] Ingesting earnings for {len(holdings)} holdings of {etf_symbol}...")
    for ticker in holdings:
        print(f"  [edgar] {ticker}...")
        n = ingest_earnings_for_symbol(con, ticker, etf_symbol=etf_symbol)
        print(f"  [edgar] {ticker}: {n} periods written.")


def build_earnings_features(con, etf_symbol: str) -> pd.DataFrame:
    """Compute per-date earnings proximity features for an ETF.

    Returns a DataFrame indexed by date with:
      eps_surprise_pct, rev_surprise_pct, days_to_earnings
    as aggregate signals across top holdings.
    """
    holdings = ETF_HOLDINGS.get(etf_symbol.upper(), [])
    if not holdings:
        return pd.DataFrame()

    placeholders = ",".join("?" * len(holdings))
    df = con.execute(
        f"""
        SELECT symbol, period_end, eps_actual, rev_actual, eps_surprise_pct, rev_surprise_pct
        FROM edgar_earnings
        WHERE symbol IN ({placeholders})
        ORDER BY symbol, period_end
        """,
        holdings,
    ).df()

    if df.empty:
        return pd.DataFrame()

    df["period_end"] = pd.to_datetime(df["period_end"]).dt.date

    # Aggregate: mean EPS surprise across holdings on each filing date
    agg = (
        df.groupby("period_end")
        .agg(
            eps_surprise_pct=("eps_surprise_pct", "mean"),
            rev_surprise_pct=("rev_surprise_pct", "mean"),
        )
        .reset_index()
        .rename(columns={"period_end": "date"})
    )
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest EDGAR earnings for ETF top holdings.")
    ap.add_argument("--db", default="data/market_mvp.duckdb")
    ap.add_argument("--etf", nargs="+", default=["SPY", "QQQ"])
    ns = ap.parse_args()

    db = DB(path=Path(ns.db))
    con = db.connect()
    init_db(con)

    for etf in ns.etf:
        ingest_etf_holdings_earnings(con, etf)
    con.execute("CHECKPOINT")
    con.close()
    print("[edgar] done.")
