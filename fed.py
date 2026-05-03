from __future__ import annotations

"""Fed minutes scraper: FOMC statements + minutes → hawkish/dovish scoring + LLM summary.

Scrapes federalreserve.gov for FOMC documents.
Primary scoring: keyword-based (free, offline).
Optional LLM summary: deepseek-r1:70b via Ollama for richer interpretation.
"""

import argparse
import re
import time
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

from market_mvp.db import DB, init_db

FED_BASE = "https://www.federalreserve.gov"
FOMC_CALENDAR_URL = f"{FED_BASE}/monetarypolicy/fomccalendars.htm"

_HEADERS = {"User-Agent": "market_mvp/1.0 rakesh@rakeshpillai.com"}

# Hawkish language signals tighter policy / concern about inflation
HAWKISH = frozenset([
    "inflation", "inflationary", "tighten", "tightening", "restrictive",
    "hike", "raise rates", "rate increase", "elevated", "persistent",
    "vigilant", "overshoot", "above target", "concerned", "upside risk",
    "supply constraints", "wage growth", "overheating",
])

# Dovish language signals looser policy / support for growth
DOVISH = frozenset([
    "ease", "easing", "accommodative", "pause", "hold rates", "cut",
    "lower rates", "rate decrease", "below target", "undershoot",
    "support", "stimulus", "transitory", "temporary", "moderate",
    "patient", "labor market", "unemployment", "slack", "downside risk",
])


def _get(url: str) -> requests.Response:
    time.sleep(0.5)
    return requests.get(url, headers=_HEADERS, timeout=30)


def _score_text(text: str) -> dict:
    """Count hawkish/dovish keyword hits per 1000 words."""
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    total = max(len(words), 1)

    hawkish_count = 0
    dovish_count = 0
    for term in HAWKISH:
        hawkish_count += text_lower.count(term)
    for term in DOVISH:
        dovish_count += text_lower.count(term)

    # Net score > 0 = hawkish, < 0 = dovish, per 1000 words
    net = (hawkish_count - dovish_count) / total * 1000
    return {
        "hawkish_score": hawkish_count / total * 1000,
        "dovish_score": dovish_count / total * 1000,
        "net_score": net,
    }


def _pdf_to_text(content: bytes) -> str:
    try:
        with pdfplumber.open(BytesIO(content)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        return ""


def _html_to_text(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _fetch_fomc_links() -> list[dict]:
    """Scrape FOMC calendar page for meeting dates and document links.

    Parses links by URL pattern — robust to page layout changes.
    Statement URLs: /newsevents/pressreleases/monetary{YYYYMMDD}a.htm
    Minutes URLs:   /monetarypolicy/fomcminutes{YYYYMMDD}.htm
    """
    resp = _get(FOMC_CALENDAR_URL)
    soup = BeautifulSoup(resp.content, "html.parser")
    results = []
    seen = set()

    stmt_re = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm")
    mins_re = re.compile(r"/monetarypolicy/fomcminutes(\d{8})\.htm")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        for pattern, doc_type in [(stmt_re, "statement"), (mins_re, "minutes")]:
            m = pattern.search(href)
            if m:
                date_str = m.group(1)   # YYYYMMDD
                key = (date_str, doc_type)
                if key in seen:
                    break
                seen.add(key)
                url = href if href.startswith("http") else FED_BASE + href
                results.append({
                    "date_text": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
                    "doc_type": doc_type,
                    "url": url,
                })
                break

    return results


def _parse_meeting_date(date_text: str) -> date | None:
    """Parse FOMC meeting date string to a date object.

    Accepts ISO format YYYY-MM-DD (from URL pattern) or legacy text like
    'January 28-29, 2025'.
    """
    # ISO format from URL-based extraction
    iso = re.match(r"(\d{4}-\d{2}-\d{2})$", date_text.strip())
    if iso:
        try:
            return datetime.strptime(iso.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass
    # Legacy text format
    match = re.search(r"(\w+ \d+(?:-\d+)?,?\s*\d{4})", date_text)
    if not match:
        return None
    cleaned = re.sub(r"\d+-(\d+)", r"\1", match.group(1))
    for fmt in ("%B %d, %Y", "%B %d %Y"):
        try:
            return datetime.strptime(cleaned.strip(), fmt).date()
        except ValueError:
            continue
    return None


def ingest_fed_minutes(con, *, use_llm: bool = False, limit: int | None = None) -> None:
    """Scrape FOMC documents, score sentiment, optionally summarize with LLM."""
    print("[fed] Fetching FOMC calendar...")
    links = _fetch_fomc_links()
    print(f"[fed] Found {len(links)} documents.")
    if limit:
        links = links[:limit]

    for item in links:
        meeting_date = _parse_meeting_date(item["date_text"])
        if not meeting_date:
            continue
        doc_type = item["doc_type"]

        # Skip if already cached
        existing = con.execute(
            "SELECT 1 FROM fed_minutes WHERE meeting_date=? AND document_type=?",
            (str(meeting_date), doc_type),
        ).fetchone()
        if existing:
            continue

        print(f"  [fed] {meeting_date} {doc_type}: fetching {item['url']}")
        try:
            resp = _get(item["url"])
        except Exception as e:
            print(f"  [fed] fetch error: {e}")
            continue

        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type or item["url"].endswith(".pdf"):
            raw_text = _pdf_to_text(resp.content)
        else:
            raw_text = _html_to_text(resp.content)

        if not raw_text.strip():
            print(f"  [fed] empty text for {meeting_date} {doc_type}, skipping.")
            continue

        scores = _score_text(raw_text)
        llm_summary = None

        if use_llm:
            try:
                from market_mvp import llm
                # Truncate to ~4000 chars to fit context
                excerpt = raw_text[:4000]
                llm_summary = llm.complete(
                    f"""Analyze this Federal Reserve FOMC {doc_type} excerpt.
Provide a 2-3 sentence summary focusing on:
1. The overall policy stance (hawkish/dovish/neutral)
2. Key concerns mentioned (inflation, employment, growth)
3. Forward guidance signals

Excerpt:
{excerpt}""",
                    reasoning=True,  # Use deepseek-r1 for nuanced policy analysis
                )
            except Exception as e:
                print(f"  [fed] LLM error: {e}")

        con.execute(
            """
            INSERT OR REPLACE INTO fed_minutes
            (meeting_date, document_type, raw_text, hawkish_score, dovish_score, net_score, llm_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(meeting_date), doc_type,
                raw_text[:50000],  # cap storage
                scores["hawkish_score"], scores["dovish_score"], scores["net_score"],
                llm_summary,
            ),
        )
        print(f"  [fed] {meeting_date} {doc_type}: net_score={scores['net_score']:.2f}")

    con.execute("CHECKPOINT")


def build_fed_features(con, as_of_dates: list[date]) -> pd.DataFrame:
    """For each date, compute fed_hawkish_score, fed_net_score, fed_days_since.

    Looks back to the most recent FOMC meeting before each date.
    """
    minutes_df = con.execute(
        "SELECT meeting_date, net_score, hawkish_score FROM fed_minutes ORDER BY meeting_date"
    ).df()
    if minutes_df.empty:
        return pd.DataFrame({"date": as_of_dates, "fed_hawkish_score": None, "fed_net_score": None, "fed_days_since": None})

    minutes_df["meeting_date"] = pd.to_datetime(minutes_df["meeting_date"]).dt.date

    rows = []
    for d in as_of_dates:
        past = minutes_df[minutes_df["meeting_date"] <= d]
        if past.empty:
            rows.append({"date": d, "fed_hawkish_score": None, "fed_net_score": None, "fed_days_since": None})
        else:
            last = past.iloc[-1]
            days_since = (d - last["meeting_date"]).days
            rows.append({
                "date": d,
                "fed_hawkish_score": last["hawkish_score"],
                "fed_net_score": last["net_score"],
                "fed_days_since": days_since,
            })
    return pd.DataFrame(rows)


# Fix missing import
from datetime import datetime


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest Fed FOMC minutes and statements.")
    ap.add_argument("--db", default="data/market_mvp.duckdb")
    ap.add_argument("--use-llm", action="store_true", help="Summarize with Ollama (deepseek-r1:70b)")
    ap.add_argument("--limit", type=int, default=None, help="Max documents to fetch (for testing)")
    ns = ap.parse_args()

    db = DB(path=Path(ns.db))
    con = db.connect()
    init_db(con)
    ingest_fed_minutes(con, use_llm=ns.use_llm, limit=ns.limit)
    con.close()
    print("[fed] done.")
