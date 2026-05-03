from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import duckdb


@dataclass(frozen=True)
class DB:
    path: Path

    def connect(self) -> duckdb.DuckDBPyConnection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(self.path))
        con.execute("PRAGMA threads=4")
        return con


SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS api_cache (
  provider      VARCHAR NOT NULL,
  endpoint      VARCHAR NOT NULL,
  symbol        VARCHAR,
  params_json   VARCHAR NOT NULL,
  fetched_at    TIMESTAMP NOT NULL DEFAULT now(),
  payload_json  VARCHAR NOT NULL,
  PRIMARY KEY(provider, endpoint, symbol, params_json)
);

CREATE TABLE IF NOT EXISTS prices_daily (
  symbol            VARCHAR NOT NULL,
  date              DATE NOT NULL,
  open              DOUBLE,
  high              DOUBLE,
  low               DOUBLE,
  close             DOUBLE,
  adjusted_close    DOUBLE,
  volume            BIGINT,
  dividend_amount   DOUBLE,
  split_coefficient DOUBLE,
  PRIMARY KEY(symbol, date)
);

CREATE TABLE IF NOT EXISTS options_pcr_daily (
  symbol        VARCHAR NOT NULL,
  date          DATE NOT NULL,
  put_call_ratio DOUBLE,
  PRIMARY KEY(symbol, date)
);

CREATE TABLE IF NOT EXISTS options_voi_daily (
  symbol          VARCHAR NOT NULL,
  date            DATE NOT NULL,
  volume_oi_ratio DOUBLE,
  PRIMARY KEY(symbol, date)
);

CREATE TABLE IF NOT EXISTS news_sentiment (
  symbol                  VARCHAR NOT NULL,
  time_published          TIMESTAMP NOT NULL,
  overall_sentiment_score DOUBLE,
  overall_sentiment_label VARCHAR,
  relevance_score         DOUBLE,
  source                  VARCHAR,
  title                   VARCHAR,
  url                     VARCHAR,
  raw_json                VARCHAR,
  PRIMARY KEY(symbol, time_published, url)
);

CREATE TABLE IF NOT EXISTS features_daily (
  symbol          VARCHAR NOT NULL,
  date            DATE NOT NULL,
  horizon         INTEGER NOT NULL,
  y_fwd_return    DOUBLE,

  pcr             DOUBLE,
  pcr_chg_5       DOUBLE,
  voi             DOUBLE,
  voi_chg_5       DOUBLE,

  news_sent       DOUBLE,
  news_sent_chg_5 DOUBLE,
  news_count      INTEGER,

  div_ex_days     INTEGER,
  div_amount      DOUBLE,

  ret_1d          DOUBLE,
  ret_5d          DOUBLE,
  ret_20d         DOUBLE,
  ret_60d         DOUBLE,

  vol_10d         DOUBLE,
  vol_20d         DOUBLE,

  price_ma5_ratio  DOUBLE,
  price_ma20_ratio DOUBLE,
  price_ma60_ratio DOUBLE,

  rsi_14          DOUBLE,
  vol_ratio_20d   DOUBLE,
  intraday_range  DOUBLE,

  eps_surprise_pct  DOUBLE,
  rev_surprise_pct  DOUBLE,
  days_to_earnings  INTEGER,

  fed_hawkish_score  DOUBLE,
  fed_net_score      DOUBLE,
  fed_days_since     INTEGER,

  stocktwits_bull_ratio DOUBLE,
  reddit_sentiment      DOUBLE,
  social_volume_ratio   DOUBLE,

  PRIMARY KEY(symbol, date, horizon)
);

CREATE TABLE IF NOT EXISTS edgar_cik_map (
  symbol  VARCHAR NOT NULL PRIMARY KEY,
  cik     VARCHAR NOT NULL,
  name    VARCHAR
);

CREATE TABLE IF NOT EXISTS edgar_filings (
  cik           VARCHAR NOT NULL,
  accession     VARCHAR NOT NULL,
  symbol        VARCHAR NOT NULL,
  form_type     VARCHAR NOT NULL,
  filed_date    DATE NOT NULL,
  period_end    DATE,
  PRIMARY KEY(cik, accession)
);

CREATE TABLE IF NOT EXISTS edgar_earnings (
  symbol              VARCHAR NOT NULL,
  period_end          DATE NOT NULL,
  filed_date          DATE,
  form_type           VARCHAR,
  eps_actual          DOUBLE,
  rev_actual          DOUBLE,
  eps_surprise_pct    DOUBLE,
  rev_surprise_pct    DOUBLE,
  guidance_text       VARCHAR,
  guidance_sentiment  DOUBLE,
  raw_json            VARCHAR,
  PRIMARY KEY(symbol, period_end)
);

CREATE TABLE IF NOT EXISTS fed_minutes (
  meeting_date   DATE NOT NULL,
  published_date DATE,
  document_type  VARCHAR NOT NULL,
  raw_text       VARCHAR,
  hawkish_score  DOUBLE,
  dovish_score   DOUBLE,
  net_score      DOUBLE,
  rate_change    DOUBLE,
  llm_summary    VARCHAR,
  PRIMARY KEY(meeting_date, document_type)
);

CREATE TABLE IF NOT EXISTS social_sentiment_daily (
  symbol          VARCHAR NOT NULL,
  date            DATE NOT NULL,
  source          VARCHAR NOT NULL,
  bull_ratio      DOUBLE,
  sentiment_score DOUBLE,
  message_count   INTEGER,
  PRIMARY KEY(symbol, date, source)
);
"""


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)
