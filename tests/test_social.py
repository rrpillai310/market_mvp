from __future__ import annotations

import pytest

from market_mvp.social import _simple_sentiment, build_social_features


def test_simple_sentiment_bullish_text_is_positive():
    text = "SPY is bullish, strong rally, buy the dip, calls are printing"
    score = _simple_sentiment(text)
    assert score > 0


def test_simple_sentiment_bearish_text_is_negative():
    text = "SPY bearish crash dump selling puts weak market falling"
    score = _simple_sentiment(text)
    assert score < 0


def test_simple_sentiment_neutral_text_is_zero():
    score = _simple_sentiment("the market opened today")
    assert score == 0.0


def test_simple_sentiment_bounded_between_minus_one_and_one():
    texts = [
        "buy buy buy bullish moon calls growth",
        "sell sell sell bearish crash puts dump",
        "mixed bullish bearish signals",
        "",
    ]
    for text in texts:
        score = _simple_sentiment(text)
        assert -1.0 <= score <= 1.0, f"Out of range for: '{text}'"


def test_simple_sentiment_empty_string_returns_zero():
    assert _simple_sentiment("") == 0.0


def _insert_social_rows(con, rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    con.register("tmp_soc", df)
    con.execute(
        """
        INSERT OR REPLACE INTO social_sentiment_daily
        SELECT symbol, CAST(date AS DATE), source,
               bull_ratio, sentiment_score, CAST(message_count AS INTEGER)
        FROM tmp_soc
        """
    )
    con.unregister("tmp_soc")


def test_build_social_features_aggregates_by_date(con):
    _insert_social_rows(con, [
        {"symbol": "SPY", "date": "2024-01-02", "source": "stocktwits",
         "bull_ratio": 0.6, "sentiment_score": 0.2, "message_count": 100},
        {"symbol": "SPY", "date": "2024-01-02", "source": "reddit",
         "bull_ratio": None, "sentiment_score": 0.1, "message_count": 50},
        {"symbol": "SPY", "date": "2024-01-03", "source": "stocktwits",
         "bull_ratio": 0.4, "sentiment_score": -0.1, "message_count": 80},
    ])
    result = build_social_features(con, "SPY")
    assert len(result) == 2
    assert set(result.columns) >= {"date", "stocktwits_bull_ratio", "reddit_sentiment", "social_volume_ratio"}


def test_build_social_features_stocktwits_bull_ratio_correct(con):
    _insert_social_rows(con, [
        {"symbol": "SPY", "date": "2024-01-02", "source": "stocktwits",
         "bull_ratio": 0.65, "sentiment_score": 0.3, "message_count": 200},
    ])
    result = build_social_features(con, "SPY")
    assert result.iloc[0]["stocktwits_bull_ratio"] == pytest.approx(0.65)


def test_build_social_features_returns_empty_for_unknown_symbol(con):
    result = build_social_features(con, "UNKNOWN")
    assert result.empty


def test_build_social_features_social_volume_ratio_positive(con):
    _insert_social_rows(con, [
        {"symbol": "QQQ", "date": f"2024-01-{i:02d}", "source": "stocktwits",
         "bull_ratio": 0.5, "sentiment_score": 0.0, "message_count": 100}
        for i in range(2, 12)
    ])
    result = build_social_features(con, "QQQ")
    assert (result["social_volume_ratio"] > 0).all()
