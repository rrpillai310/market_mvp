from __future__ import annotations

from datetime import date

import pytest

from market_mvp.fed import _parse_meeting_date, _score_text, build_fed_features, HAWKISH, DOVISH


def test_score_text_hawkish_text_has_positive_net_score():
    text = (
        "Inflation remains elevated and persistent. The committee is vigilant "
        "about upside risks. Tightening is appropriate. Rate hike is necessary. "
        "Inflationary pressures remain above target. Restrictive policy continues."
    )
    scores = _score_text(text)
    assert scores["net_score"] > 0
    assert scores["hawkish_score"] > 0


def test_score_text_dovish_text_has_negative_net_score():
    text = (
        "The labor market shows slack. Accommodative stance is appropriate. "
        "We will ease policy as needed. Transitory factors are at play. "
        "Downside risk to growth warrants patience. Cut rates to support growth."
    )
    scores = _score_text(text)
    assert scores["net_score"] < 0
    assert scores["dovish_score"] > 0


def test_score_text_neutral_text_scores_near_zero():
    text = "The committee met today. Economic data was reviewed. No changes were made."
    scores = _score_text(text)
    # Should be very close to zero since no hawkish/dovish terms
    assert abs(scores["net_score"]) < 1.0


def test_score_text_returns_required_keys():
    scores = _score_text("test text")
    assert "hawkish_score" in scores
    assert "dovish_score" in scores
    assert "net_score" in scores


def test_score_text_scores_are_non_negative():
    scores = _score_text("inflation tighten hike ease accommodative cut")
    assert scores["hawkish_score"] >= 0
    assert scores["dovish_score"] >= 0


@pytest.mark.parametrize("text,expected_date", [
    ("January 28-29, 2025", date(2025, 1, 29)),
    ("March 19-20, 2024", date(2024, 3, 20)),
    ("December 17-18, 2024", date(2024, 12, 18)),
    ("May 6-7, 2025", date(2025, 5, 7)),
])
def test_parse_meeting_date_known_formats(text, expected_date):
    result = _parse_meeting_date(text)
    assert result == expected_date


def test_parse_meeting_date_returns_none_for_unparseable():
    result = _parse_meeting_date("no date here")
    assert result is None


def test_hawkish_wordlist_has_no_overlap_with_dovish():
    overlap = HAWKISH & DOVISH
    assert not overlap, f"Overlapping terms: {overlap}"


def test_build_fed_features_returns_all_dates(con):
    import pandas as pd
    # Insert a fake FOMC meeting
    con.execute(
        """
        INSERT INTO fed_minutes(meeting_date, document_type, net_score, hawkish_score)
        VALUES ('2024-01-31', 'statement', 1.5, 3.0)
        """
    )
    as_of_dates = [date(2024, 2, 1), date(2024, 2, 15), date(2024, 3, 1)]
    result = build_fed_features(con, as_of_dates)
    assert len(result) == 3


def test_build_fed_features_uses_most_recent_meeting(con):
    con.execute(
        """
        INSERT INTO fed_minutes(meeting_date, document_type, net_score, hawkish_score)
        VALUES
          ('2024-01-31', 'statement', 1.0, 2.0),
          ('2024-03-20', 'statement', 5.0, 8.0)
        """
    )
    result = build_fed_features(con, [date(2024, 4, 1)])
    # Should use March meeting (most recent before April 1)
    assert result.iloc[0]["fed_net_score"] == pytest.approx(5.0)
    assert result.iloc[0]["fed_hawkish_score"] == pytest.approx(8.0)


def test_build_fed_features_days_since_is_correct(con):
    con.execute(
        "INSERT INTO fed_minutes(meeting_date, document_type, net_score, hawkish_score) VALUES ('2024-01-31', 'statement', 1.0, 2.0)"
    )
    result = build_fed_features(con, [date(2024, 2, 10)])
    assert result.iloc[0]["fed_days_since"] == 10


def test_build_fed_features_empty_db_returns_none_scores(con):
    result = build_fed_features(con, [date(2024, 1, 1)])
    assert result.iloc[0]["fed_net_score"] is None
