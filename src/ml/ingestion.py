"""
ingestion.py — CareWatch Data Ingestion
========================================
Handles:
  - Auto-detection of data format (CSV / Parquet / Video / mixed)
  - Memory-efficient streaming via generators
  - Schema validation and type coercion
  - Label normalisation
"""

import os
import re
import glob
import logging
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, DATA_FORMAT, LABEL_COLUMN, LABEL_INT_COLUMN,
    CLASS_LABELS, CLASS_NAMES, CHUNK_ROWS, INDEX_COLUMN,
    JOINT_COORD_PATTERN, VIDEO_EXTENSIONS
)

logger = logging.getLogger("carewatch.ingestion")


# ─────────────────────────────────────────────────────────────────────────────
# File Discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_files(data_dir: str = DATA_DIR) -> dict:
    """
    Walk data_dir and return lists of files by type.
    Returns: {"csv": [...], "parquet": [...], "video": [...]}
    """
    out = {"csv": [], "parquet": [], "video": []}
    for root, _, files in os.walk(data_dir):
        for f in files:
            path = os.path.join(root, f)
            ext  = os.path.splitext(f)[1].lower()
            if ext in (".csv", ".tsv"):
                out["csv"].append(path)
            elif ext in (".parquet", ".pq", ".feather"):
                out["parquet"].append(path)
            elif ext in VIDEO_EXTENSIONS:
                out["video"].append(path)
    total = sum(len(v) for v in out.values())
    logger.info(f"Discovered {total} files: CSV={len(out['csv'])}, "
                f"Parquet={len(out['parquet'])}, Video={len(out['video'])}")
    return out


def detect_format(data_dir: str = DATA_DIR) -> str:
    """Return dominant format if DATA_FORMAT == 'auto'."""
    if DATA_FORMAT != "auto":
        return DATA_FORMAT
    files = discover_files(data_dir)
    counts = {k: len(v) for k, v in files.items()}
    dominant = max(counts, key=counts.get)
    logger.info(f"Auto-detected data format: {dominant}")
    return dominant


# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_joint_columns(df: pd.DataFrame) -> List[str]:
    """Return all column names matching the joint coordinate pattern."""
    pattern = re.compile(JOINT_COORD_PATTERN, re.IGNORECASE)
    return [c for c in df.columns if pattern.match(c)]


def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has an integer 'label' column.
    Handles:
      - String class names  ("fall", "normal", …)
      - Already-integer labels
      - Missing labels (set to -1 for semi-supervised use)
    """
    if LABEL_COLUMN in df.columns:
        raw = df[LABEL_COLUMN]
        if pd.api.types.is_string_dtype(raw):
            df["label_int"] = raw.str.lower().str.strip().map(CLASS_LABELS).fillna(-1).astype(int)
        else:
            df["label_int"] = raw.astype(int)
    elif LABEL_INT_COLUMN and LABEL_INT_COLUMN in df.columns:
        df["label_int"] = df[LABEL_INT_COLUMN].astype(int)
    else:
        logger.warning("No label column found — assigning -1 (unlabelled)")
        df["label_int"] = -1

    return df


def drop_junk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop index artifacts and fully-null columns."""
    cols_to_drop = []
    if INDEX_COLUMN in df.columns:
        cols_to_drop.append(INDEX_COLUMN)
    null_cols = df.columns[df.isnull().all()].tolist()
    cols_to_drop.extend(null_cols)
    return df.drop(columns=cols_to_drop, errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CSV Streaming
# ─────────────────────────────────────────────────────────────────────────────

def stream_csv_chunks(
    file_path: str,
    chunk_rows: int = CHUNK_ROWS
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield DataFrame chunks from a single CSV file.
    Uses pandas TextFileReader — O(chunk_rows) memory at a time.
    """
    reader = pd.read_csv(
        file_path,
        chunksize=chunk_rows,
        low_memory=False,
        na_values=["", "NA", "NaN", "null", "none", "N/A"],
        keep_default_na=True,
    )
    for chunk in reader:
        chunk = drop_junk_columns(chunk)
        chunk = normalise_labels(chunk)
        # Coerce numeric columns — non-numeric → NaN
        numeric_cols = chunk.select_dtypes(exclude=["object"]).columns
        chunk[numeric_cols] = chunk[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        yield chunk


def stream_all_csvs(
    data_dir: str = DATA_DIR,
    chunk_rows: int = CHUNK_ROWS
) -> Generator[pd.DataFrame, None, None]:
    """Stream chunks from every CSV in data_dir."""
    files = discover_files(data_dir)["csv"]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    for f in sorted(files):
        logger.info(f"  Streaming CSV: {f}")
        yield from stream_csv_chunks(f, chunk_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Parquet Streaming
# ─────────────────────────────────────────────────────────────────────────────

def stream_parquet_chunks(
    file_path: str,
    batch_size: int = CHUNK_ROWS
) -> Generator[pd.DataFrame, None, None]:
    """
    Read a Parquet file in row-group batches using pyarrow.
    Significantly faster than CSV for large typed datasets.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pip install pyarrow to read Parquet files")

    pf = pq.ParquetFile(file_path)
    for batch in pf.iter_batches(batch_size=batch_size):
        chunk = batch.to_pandas()
        chunk = drop_junk_columns(chunk)
        chunk = normalise_labels(chunk)
        yield chunk


def stream_all_parquets(
    data_dir: str = DATA_DIR,
    batch_size: int = CHUNK_ROWS
) -> Generator[pd.DataFrame, None, None]:
    files = discover_files(data_dir)["parquet"]
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")
    for f in sorted(files):
        logger.info(f"  Streaming Parquet: {f}")
        yield from stream_parquet_chunks(f, batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# Unified streaming entry point
# ─────────────────────────────────────────────────────────────────────────────

def stream_data(
    data_dir: str = DATA_DIR,
    chunk_rows: int = CHUNK_ROWS,
    fmt: Optional[str] = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Top-level generator — auto or explicit format dispatch.
    Yields clean, label-normalised DataFrame chunks.
    """
    fmt = fmt or detect_format(data_dir)
    if fmt in ("csv", "tsv"):
        yield from stream_all_csvs(data_dir, chunk_rows)
    elif fmt in ("parquet", "feather"):
        yield from stream_all_parquets(data_dir, chunk_rows)
    elif fmt == "video":
        # Video frames come from a separate pipeline in video_pipeline.py
        raise ValueError(
            "Use video_pipeline.stream_video_chunks() for video data."
        )
    else:
        raise ValueError(f"Unknown data format: {fmt}")


# ─────────────────────────────────────────────────────────────────────────────
# Schema Inspector (run once to understand your data)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_schema(data_dir: str = DATA_DIR, n_chunks: int = 3) -> dict:
    """
    Read a few chunks and report schema, label distribution,
    missing-value rates, and detected joint columns.
    Useful for debugging before running the full pipeline.
    """
    fmt   = detect_format(data_dir)
    stats = {"joint_columns": [], "label_dist": {}, "null_rates": {}, "n_rows": 0}

    for i, chunk in enumerate(stream_data(data_dir, fmt=fmt)):
        if i == 0:
            stats["columns"]       = chunk.columns.tolist()
            stats["dtypes"]        = chunk.dtypes.astype(str).to_dict()
            stats["joint_columns"] = detect_joint_columns(chunk)

        # Accumulate label counts
        if "label_int" in chunk.columns:
            for lbl, cnt in chunk["label_int"].value_counts().items():
                stats["label_dist"][lbl] = stats["label_dist"].get(lbl, 0) + cnt

        # Null rates (first chunk only — sufficient estimate)
        if i == 0:
            stats["null_rates"] = (chunk.isnull().mean() * 100).round(2).to_dict()

        stats["n_rows"] += len(chunk)
        if i >= n_chunks - 1:
            break

    logger.info("=== Schema Inspection ===")
    logger.info(f"Rows inspected   : {stats['n_rows']}")
    logger.info(f"Joint columns    : {len(stats['joint_columns'])}")
    logger.info(f"Label distribution: {stats['label_dist']}")
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    info = inspect_schema()
    import json
    print(json.dumps({k: str(v) for k, v in info.items()}, indent=2))
