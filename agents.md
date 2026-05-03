# agents.md — AI Agent Guide for market_mvp

Instructions for Codex, Claude Code, and any other AI coding agent working in this repo.
Read CLAUDE.md first for architecture context. This file covers agent-specific rules.

## Orientation (read before any task)

1. **This is a Python package.** Files at repo root are `market_mvp.*` modules.
   Run everything from `/Users/rakeshpillai/` (parent of this repo), never from inside.

2. **Single source of truth for features:** `FEATURE_COLS` list in `train.py:13`.
   If you add a feature anywhere, you must add it there AND in `features_daily` schema
   in `db.py` AND write it from `features.py`. All three or none.

3. **DuckDB is the only persistence layer.** No Redis, no Postgres, no flat files except
   the model `.pkl`. Keep it that way.

4. **All ingestion is idempotent.** Every `INSERT` uses `INSERT OR REPLACE`.
   Every fetch checks `api_cache` first. Never break this — re-running must be safe.

5. **No network calls in tests.** Mock everything at the HTTP boundary.
   `tests/conftest.py` provides in-memory DuckDB fixtures. Use them.

## Task playbook

### Adding a new feature
```
1. Add column to features_daily in db.py SCHEMA_SQL
2. Compute it in features.py:build_features(), join on date index
3. Add column name to FEATURE_COLS in train.py
4. Write test in tests/test_features.py verifying the column exists and
   has a plausible range for synthetic data
```

### Adding a new data source
```
1. Create market_mvp/mysource.py
   - ingest_mysource(con, symbol) → writes to a new table
   - build_mysource_features(con, symbol) → pd.DataFrame with 'date' + feature cols
2. Add table to db.py SCHEMA_SQL
3. Add feature columns to features_daily schema + FEATURE_COLS
4. Wire join into features.py:build_features()
5. Add --skip-mysource flag in pipeline.py
6. Write tests/test_mysource.py (no network, mock HTTP)
```

### Modifying the DB schema
- DuckDB does not support `ALTER TABLE ADD COLUMN IF NOT EXISTS` in all versions.
  Safest approach: add new columns to SCHEMA_SQL `CREATE TABLE IF NOT EXISTS` and
  document that users should delete the `.duckdb` file to pick up schema changes.
- Never rename or drop existing columns — it breaks pickled models that reference feature names.

### Changing model hyperparameters
- Edit `_build_model()` in `train.py`. Keep the LightGBM / sklearn fallback structure.
- Do not remove the sklearn fallback — it enables testing without a GPU/LightGBM install.

### Working with the Ollama LLM
- All LLM calls go through `llm.py`. Never call `openai.OpenAI()` directly in other modules.
- Default model: `qwen2.5:72b` (extraction). Reasoning model: `deepseek-r1:70b` (Fed minutes).
- Always pass `reasoning=True` for Fed/policy analysis tasks.
- LLM features are always optional — every caller must work without LLM output (fall back
  to keyword scoring or None values).
- The DGX Spark WiFi has high jitter (up to 225ms). `llm.py` handles retries; callers
  must not add their own retry loops.

### Working with EDGAR
- SEC requires a descriptive User-Agent header with contact email. It is set in `edgar.py`.
  Do not change or remove it — violating this causes IP bans.
- Rate limit: 10 requests/second. The `_get()` helper sleeps 0.12s between calls. Do not
  bypass this.
- CIK lookup results are cached in `edgar_cik_map`. Always pass `con` to `lookup_cik()`.
- The XBRL fast-path covers ~90% of modern filings. Playwright/OCR fallback is not yet built.
  If a company's facts are missing, log and skip — do not raise.

### Working with the Fed scraper
- `fed.py` scrapes `federalreserve.gov`. Documents are cached in `fed_minutes` by
  `(meeting_date, document_type)`. The scraper skips already-cached entries.
- Keyword scoring runs offline, always. LLM summarization is opt-in (`--use-llm-fed`).
- The hawkish/dovish word lists are in `fed.py` module-level frozensets. Add terms there,
  not inline in functions.

## Code style rules

- **No comments explaining what code does.** Only comment WHY if it's non-obvious
  (e.g., SEC rate limit, WiFi jitter handling, XBRL schema quirks).
- **No docstrings except one-line module docstrings** at the top of new files.
- **No type annotations on local variables.** Function signatures only.
- **No logging framework.** Use `print(f"[module] message")` for progress output.
  Tests should not print (use capsys or suppress in conftest if needed).
- **Imports:** stdlib → third-party → local (`market_mvp.*`), separated by blank lines.
- **No f-strings with complex expressions.** Assign to a variable first.

## Testing rules

- Every new module gets a `tests/test_<module>.py`.
- Every test function name must describe what it verifies, not what it calls.
  Good: `test_rsi_is_bounded_between_0_and_100`
  Bad: `test_rsi_function`
- Use `pytest.mark.parametrize` for testing multiple inputs.
- Do not use `unittest.TestCase`. Plain functions only.
- Mock external HTTP at the `requests.get` / `openai.OpenAI` level, not deeper.
- Use the `con` fixture from `conftest.py` for all DB interactions.

## What NOT to do

- Do not call `con.close()` inside fixtures — the fixture handles teardown.
- Do not commit `.duckdb`, `.pkl`, `.env`, or any file in `data/` or `models/`.
- Do not add retries to Alpha Vantage calls — the throttle is intentional for free tier.
- Do not make `FEATURE_COLS` dynamic at runtime — it must be a static list.
- Do not use `pd.DataFrame.append()` (removed in pandas 2.0). Use `pd.concat()`.
- Do not use `fillna(method=...)` (deprecated). Use `.ffill()` / `.bfill()` directly.
- Do not use `datetime.utcnow()` (deprecated). Use `datetime.now(timezone.utc)`.
- Do not import `market_mvp.pipeline` or `market_mvp.predict` from inside other modules.
  They are entrypoints only.

## Key invariants to preserve

| Invariant | Where enforced |
|---|---|
| All ingestion is idempotent | `INSERT OR REPLACE` everywhere |
| API responses always cached before parsing | `_cache_get` / `_cache_put` pattern in `ingest.py` |
| FEATURE_COLS order is stable | Static list in `train.py` — pkl models depend on column order |
| Feature columns filled with 0.0 if absent | `train.py:walk_forward_eval` and `train_final` |
| No network in tests | `conftest.py` fixtures + mocking |
| SEC rate limit respected | `edgar.py:_get()` sleeps 0.12s |
| Ollama calls have 120s timeout + retries | `llm.py:extract_json` |

## Running tests

```bash
cd /Users/rakeshpillai/market_mvp
pytest tests/ -v                    # all tests
pytest tests/test_features.py -v   # one module
pytest tests/ -k "rsi"             # by keyword
pytest tests/ --tb=short           # compact tracebacks
```
