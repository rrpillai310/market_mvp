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
   model artifacts (`.pkl`, `.ckpt`, `_pred.json`). Keep it that way.

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
- Default model: `qwen3.6:latest` (extraction). Reasoning model: `deepseek-r1:70b` (Fed minutes).
- Extraction calls use `extra_body={"think": False}` — without this, qwen3.6 does a slow
  chain-of-thought pass (40s+) before responding. Always pass `reasoning=False` for extraction.
- Always pass `reasoning=True` for Fed/policy analysis tasks (uses deepseek-r1:70b).
- LLM features are always optional — every caller must work without LLM output (fall back
  to keyword scoring or None values).
- The DGX Spark is on direct 10GbE Ethernet (`10.0.0.2`). Latency is sub-ms.
  `llm.py` handles retries; callers must not add their own retry loops.

### Working with the DGX Spark GPU
- GPU training uses Docker (`nvcr.io/nvidia/pytorch:25.03-py3`). No PyTorch CUDA wheels
  exist for aarch64 — never attempt `pip install torch` outside the container.
- `import lightning.pytorch` **must come before** `import pytorch_forecasting` inside any
  function that uses `Trainer`. The NGC container ships `pytorch_lightning`; pytorch-forecasting
  1.x uses `lightning.pytorch`. Mixing them makes `Trainer` reject the model with a
  "must be a LightningModule" error even though TFT is one.
- Container name is `market_mvp`. Start it with `docker start -ai market_mvp`.
- The container mounts: `~/market_mvp`, `~/data`, `~/models` — changes inside persist.
- Always verify GPU access: `python -c "import torch; print(torch.cuda.is_available())"`.
- Training entry point is `train_dgx.py` at repo root (not inside the package). Run it as
  `python /workspace/market_mvp/train_dgx.py --symbol SPY --horizon 5`.
- Outputs per (symbol, horizon):
  - `{symbol}_h{horizon}_tft.ckpt` — best PyTorch Lightning checkpoint
  - `{symbol}_h{horizon}_tft_metrics.json` — val loss + training metadata
  - `{symbol}_h{horizon}_tft_pred.json` — latest prediction (median + p10/p90 quantiles)
- Sync data from Mac to DGX before training; sync models directory back after:
  ```bash
  rsync -avz /Users/rakeshpillai/data/ rrpillai@10.0.0.2:~/data/
  # ... train on DGX ...
  rsync -avz rrpillai@10.0.0.2:~/models/ /Users/rakeshpillai/models/
  ```
- The Mac Streamlit reads only `_tft_pred.json` via `ui_data.predict_tft()` — it does not
  load the `.ckpt` file and does not require pytorch-forecasting on the Mac.

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

### Adding a new Streamlit page

```
1. Create pages/<N>_<Name>.py — Streamlit executes these alphabetically by N
2. Import data exclusively from market_mvp.ui_data (never open DuckDB directly)
3. Always guard empty DataFrames: if df.empty: st.warning(...) and return early
4. Use st.plotly_chart(fig, width="stretch") — not use_container_width=True (deprecated)
5. Add page-level tests in tests/test_ui_pages.py using AppTest.from_file()
   - Patch all ui_data.* functions with unittest.mock.patch as context managers
   - Assert: not at.exception (always), then content assertions
```

### Adding a new ui_data query function

```
1. Add the function to ui_data.py with @st.cache_data(ttl=300)
2. Add it to the clear_st_caches autouse fixture in tests/test_ui_data.py
3. Write unit tests patching _con() with the in-memory test connection
```

## Testing rules

- Every new module gets a `tests/test_<module>.py`.
- Every test function name must describe what it verifies, not what it calls.
  Good: `test_rsi_is_bounded_between_0_and_100`
  Bad: `test_rsi_function`
- Use `pytest.mark.parametrize` for testing multiple inputs.
- Do not use `unittest.TestCase`. Plain functions only.
- Mock external HTTP at the `requests.get` / `openai.OpenAI` level, not deeper.
- Use the `con` fixture from `conftest.py` for all DB interactions.

**Streamlit-specific testing rules:**
- Use `AppTest.from_file(path).run()` — never start a real server.
- Always wrap `at = AppTest.from_file(...).run()` inside the `patch()` context managers.
- Access markdown content via `e.value` (not `str(e)`) on AppTest markdown elements.
- Clear `@st.cache_data` functions with `fn.clear()` between tests (autouse fixture).
- Do not patch `ui_data` functions with `@st.cache_resource` decorated functions
  (the `_con()` helper uses `@st.cache_resource` — patch the raw function, not the wrapper).

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
- Do not open DuckDB in Streamlit pages directly — always go through `ui_data.py`.
- Do not use `use_container_width=True` in Streamlit calls — use `width="stretch"` for
  `st.plotly_chart()`. The `use_container_width` parameter is deprecated since 1.57.0.
- Do not serialize LightGBM feature importances as `np.int32` to JSON — cast to `float()`
  before writing metrics JSON (numpy integers are not JSON-serializable).
- Do not use `TIME_SERIES_DAILY_ADJUSTED` or options endpoints from Alpha Vantage — they
  are premium-only. Price data comes from yfinance; options PCR is computed from yfinance
  options chains daily.
- Do not hardcode paths assuming the DB is inside the repo. The DB and models live one
  level above: `Path(__file__).parent.parent / "data"` and `/ "models"`.
- Do not run the pipeline while Streamlit is running — DuckDB allows only one writer.
  Stop Streamlit (Ctrl+C) first, run the pipeline, then restart Streamlit.
- Do not use CSS class selectors to scrape sites that change their HTML frequently.
  Match by URL pattern (regex on href) instead — URL schemes change far less often.
- Always set `PYTHONPATH=/Users/rakeshpillai` when running Streamlit or the pipeline from
  a shell that hasn't sourced `~/.zshrc`.
- Do not attempt `pip install torch` or `conda install pytorch` on the DGX outside Docker —
  there are no PyTorch CUDA wheels for aarch64. Always use the NGC container.
- Do not use `rsync --info=progress2` on macOS — the system ships `openrsync` (not GNU
  rsync) which does not support that flag and exits with help text, silently skipped by tee.
  Use `rsync -rz --no-perms --no-owner --no-group` for Mac→DGX transfers.
- Do not call Ollama extraction functions without `reasoning=False` / `think: False` —
  qwen3.6 runs 40s+ chain-of-thought by default, making extraction impractically slow.
- Do not load the TFT `.ckpt` file in Streamlit or `ui_data.py` — the Mac has no GPU and
  loading pytorch-forecasting there adds a heavy optional dependency. Always use the
  pre-computed `_tft_pred.json` written by `train_dgx.py` on the DGX.
- Do not change `FEATURE_COLS` in `train_dgx.py` independently of `train.py` — they must
  stay identical. Both files import from the same logical list; any divergence silently
  causes the TFT and LightGBM models to train on different feature sets.

## Key invariants to preserve

| Invariant | Where enforced |
|---|---|
| All ingestion is idempotent | `INSERT OR REPLACE` everywhere |
| API responses always cached before parsing | `_cache_get` / `_cache_put` pattern in `ingest.py` |
| FEATURE_COLS order is stable | Static list in `train.py` — pkl models depend on column order |
| FEATURE_COLS identical in train.py and train_dgx.py | Must be kept in sync manually |
| Feature columns filled with 0.0 if absent | `train.py:walk_forward_eval` and `train_final` |
| TFT predictions served as JSON, not live inference | `ui_data.predict_tft()` reads `_tft_pred.json` |
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
