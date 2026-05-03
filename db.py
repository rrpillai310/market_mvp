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
  symbol        VARCHAR NOT NULL,
  date          DATE NOT NULL,
  open          DOUBLE,
  high          DOUBLE,
  low           DOUBLE,
  close         DOUBLE,
  adjusted_close DOUBLE,
  volume        BIGINT,
  dividend_amount DOUBLE,
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
  symbol        VARCHAR NOT NULL,
  date          DATE NOT NULL,
  volume_oi_ratio DOUBLE,
  PRIMARY KEY(symbol, date)
);

CREATE TABLE IF NOT EXISTS news_sentiment (
  symbol        VARCHAR NOT NULL,
  time_published TIMESTAMP NOT NULL,
  overall_sentiment_score DOUBLE,
  overall_sentiment_label VARCHAR,
  relevance_score DOUBLE,
  source        VARCHAR,
  title         VARCHAR,
  url           VARCHAR,
  raw_json      VARCHAR,
  PRIMARY KEY(symbol, time_published, url)
);

CREATE TABLE IF NOT EXISTS features_daily (
  symbol        VARCHAR NOT NULL,
  date          DATE NOT NULL,
  horizon       INTEGER NOT NULL,
  y_fwd_return  DOUBLE,

  pcr           DOUBLE,
  pcr_chg_5     DOUBLE,
  voi           DOUBLE,
  voi_chg_5     DOUBLE,

  news_sent     DOUBLE,
  news_sent_chg_5 DOUBLE,
  news_count    INTEGER,

  div_ex_days   INTEGER,
  div_amount    DOUBLE,

  PRIMARY KEY(symbol, date, horizon)
);
"""


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)

