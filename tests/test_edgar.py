from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from market_mvp.edgar import (
    ETF_HOLDINGS,
    _extract_concept,
    extract_earnings_xbrl,
    lookup_cik,
)


MOCK_COMPANY_TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    "2": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon.com Inc"},
}

MOCK_XBRL_FACTS = {
    "facts": {
        "us-gaap": {
            "EarningsPerShareBasic": {
                "units": {
                    "USD/shares": [
                        {"form": "10-Q", "end": "2024-03-31", "val": 1.53},
                        {"form": "10-Q", "end": "2024-06-30", "val": 1.40},
                        {"form": "10-K", "end": "2023-09-30", "val": 6.13},
                        {"form": "8-K", "end": "2024-03-31", "val": 99.9},  # should be ignored
                    ]
                }
            },
            "Revenues": {
                "units": {
                    "USD": [
                        {"form": "10-Q", "end": "2024-03-31", "val": 90_753_000_000},
                        {"form": "10-Q", "end": "2024-06-30", "val": 85_777_000_000},
                        {"form": "10-K", "end": "2023-09-30", "val": 383_285_000_000},
                    ]
                }
            },
        }
    }
}


def test_extract_concept_returns_only_10q_and_10k():
    result = _extract_concept(
        MOCK_XBRL_FACTS, "us-gaap",
        "EarningsPerShareBasic",
        unit="USD/shares",
    )
    assert "2024-03-31" in result
    assert "2024-06-30" in result
    assert "2023-09-30" in result
    # 8-K entries should be ignored; val=99.9 from 8-K should not appear
    assert result.get("2024-03-31") == pytest.approx(1.53)


def test_extract_concept_picks_first_matching_concept():
    # EarningsPerShareBasic takes priority over EarningsPerShareDiluted
    facts = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareBasic": {
                    "units": {"USD/shares": [{"form": "10-Q", "end": "2024-03-31", "val": 1.53}]}
                },
                "EarningsPerShareDiluted": {
                    "units": {"USD/shares": [{"form": "10-Q", "end": "2024-03-31", "val": 1.50}]}
                },
            }
        }
    }
    result = _extract_concept(facts, "us-gaap", "EarningsPerShareBasic", "EarningsPerShareDiluted", unit="USD/shares")
    assert result["2024-03-31"] == pytest.approx(1.53)


def test_extract_concept_returns_empty_for_missing_namespace():
    result = _extract_concept({}, "us-gaap", "EarningsPerShareBasic", unit="USD/shares")
    assert result == {}


def test_extract_earnings_xbrl_returns_rows_for_all_periods():
    rows = extract_earnings_xbrl(MOCK_XBRL_FACTS, "AAPL")
    periods = {r["period_end"] for r in rows}
    assert "2024-03-31" in periods
    assert "2024-06-30" in periods
    assert "2023-09-30" in periods


def test_extract_earnings_xbrl_values_match_facts():
    rows = extract_earnings_xbrl(MOCK_XBRL_FACTS, "AAPL")
    q1_row = next(r for r in rows if r["period_end"] == "2024-03-31")
    assert q1_row["eps_actual"] == pytest.approx(1.53)
    assert q1_row["rev_actual"] == pytest.approx(90_753_000_000)


def test_extract_earnings_xbrl_symbol_is_preserved():
    rows = extract_earnings_xbrl(MOCK_XBRL_FACTS, "AAPL")
    assert all(r["symbol"] == "AAPL" for r in rows)


def test_lookup_cik_finds_known_ticker(con):
    with patch("market_mvp.edgar._get", return_value=MOCK_COMPANY_TICKERS):
        cik = lookup_cik("AAPL", con=con)
    assert cik == "0000320193"


def test_lookup_cik_is_zero_padded_to_10_digits(con):
    with patch("market_mvp.edgar._get", return_value=MOCK_COMPANY_TICKERS):
        cik = lookup_cik("AAPL", con=con)
    assert len(cik) == 10


def test_lookup_cik_returns_none_for_unknown_ticker(con):
    with patch("market_mvp.edgar._get", return_value=MOCK_COMPANY_TICKERS):
        cik = lookup_cik("ZZZZ", con=con)
    assert cik is None


def test_lookup_cik_caches_result_in_db(con):
    with patch("market_mvp.edgar._get", return_value=MOCK_COMPANY_TICKERS):
        cik1 = lookup_cik("AAPL", con=con)
    # Second call should use DB cache (no mock needed for _get)
    cik2 = lookup_cik("AAPL", con=con)
    assert cik1 == cik2


def test_etf_holdings_contains_expected_symbols():
    spy = ETF_HOLDINGS["SPY"]
    assert "AAPL" in spy
    assert "MSFT" in spy
    assert len(spy) >= 10


@pytest.mark.parametrize("etf", ["SPY", "QQQ"])
def test_etf_holdings_no_duplicates(etf):
    holdings = ETF_HOLDINGS[etf]
    assert len(holdings) == len(set(holdings))
