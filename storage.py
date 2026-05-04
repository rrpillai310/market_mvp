from __future__ import annotations

"""Storage abstraction: configurable local paths + optional S3 sync.

On Mac (default): reads DATA_DIR / MODELS_DIR env vars, falls back to
  ../data/ and ../models/ relative to the repo root (same as before).

On AWS: set DATA_DIR=/data (EBS mount) and S3_BUCKET=my-bucket to sync
  DuckDB and models to/from S3 for durability across instance stop/start.
"""

import os
import subprocess
from pathlib import Path


def get_data_dir() -> Path:
    """Return the data directory, honouring DATA_DIR env var."""
    env = os.getenv("DATA_DIR")
    if env:
        return Path(env)
    # Default: ../data/ relative to this file's parent (the repo root)
    return Path(__file__).parent.parent / "data"


def get_models_dir() -> Path:
    """Return the models directory, honouring MODELS_DIR env var."""
    env = os.getenv("MODELS_DIR")
    if env:
        return Path(env)
    return Path(__file__).parent.parent / "models"


def maybe_sync_from_s3(local_dir: Path, s3_prefix: str) -> None:
    """Download from S3 → local_dir if S3_BUCKET is configured. No-op otherwise."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return
    local_dir.mkdir(parents=True, exist_ok=True)
    s3_uri = f"s3://{bucket}/{s3_prefix}/"
    print(f"[storage] syncing {s3_uri} → {local_dir}")
    subprocess.run(
        ["aws", "s3", "sync", s3_uri, str(local_dir)],
        check=True,
    )


def maybe_sync_to_s3(local_dir: Path, s3_prefix: str) -> None:
    """Upload local_dir → S3 if S3_BUCKET is configured. No-op otherwise."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket or not local_dir.exists():
        return
    s3_uri = f"s3://{bucket}/{s3_prefix}/"
    print(f"[storage] syncing {local_dir} → {s3_uri}")
    subprocess.run(
        ["aws", "s3", "sync", str(local_dir), s3_uri],
        check=True,
    )
