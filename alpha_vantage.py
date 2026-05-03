from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests


BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageError(RuntimeError):
    pass


@dataclass
class AlphaVantageClient:
    api_key: str | None = None
    throttle_secs: float = 12.5  # stay well under common free-tier per-minute limits

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise AlphaVantageError("Missing ALPHAVANTAGE_API_KEY env var")

    def get(self, *, function: str, params: dict[str, Any]) -> dict[str, Any]:
        q = {"function": function, "apikey": self.api_key, **params}
        # Throttle to avoid free-tier bans. Also helps keep the MVP predictable.
        time.sleep(self.throttle_secs)
        r = requests.get(BASE_URL, params=q, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and ("Error Message" in data or "Information" in data):
            raise AlphaVantageError(json.dumps(data)[:500])
        return data

