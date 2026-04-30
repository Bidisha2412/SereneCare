"""
scaler.py — CareWatch Incremental Feature Scaling & Feature Cache
=================================================================
Problem: StandardScaler needs mean/std of the ENTIRE dataset,
         but we can't load 20 GB at once.

Solution:
  Pass 1 → stream all chunks, accumulate running mean/variance (Welford's)
  Pass 2 → stream again, apply fitted scaler, write Parquet feature cache
           (compressed row-groups, ~3–5× smaller than CSV)

The feature cache is a Parquet directory that subsequent training passes
read sequentially — enabling multi-pass algorithms (LightGBM, XGBoost via
DMatrix) without re-running the expensive feature engineering.
"""

import os
import logging
import pickle
from typing import Generator, List, Optional

import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR, FEATURE_CACHE, SCALER_TYPE,
    PARQUET_ROW_GROUP
)

logger = logging.getLogger("carewatch.scaler")
os.makedirs(FEATURE_CACHE, exist_ok=True)
os.makedirs(OUTPUT_DIR,    exist_ok=True)

SCALER_PATH   = os.path.join(OUTPUT_DIR, "carewatch_scaler.pkl")
FEAT_COL_PATH = os.path.join(OUTPUT_DIR, "feature_columns.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Welford's online mean / variance accumulator
# ─────────────────────────────────────────────────────────────────────────────

class WelfordAccumulator:
    """
    Numerically stable one-pass mean and variance estimator.
    No full dataset held in memory — O(features) space.
    """
    def __init__(self, n_features: int):
        self.n     = 0
        self.mean  = np.zeros(n_features, dtype=np.float64)
        self.M2    = np.zeros(n_features, dtype=np.float64)

    def update_batch(self, X: np.ndarray):
        """Accept a 2-D array (n_samples × n_features)."""
        for row in X:
            self.n    += 1
            delta      = row - self.mean
            self.mean += delta / self.n
            delta2     = row - self.mean
            self.M2   += delta * delta2

    @property
    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance)

    @property
    def scale_(self) -> np.ndarray:
        std = self.std
        std[std == 0] = 1.0   # avoid divide-by-zero
        return std


# ─────────────────────────────────────────────────────────────────────────────
# Fitted scaler (thin wrapper, sklearn-compatible interface)
# ─────────────────────────────────────────────────────────────────────────────

class CareWatchScaler:
    """
    Stores mean_ / scale_ arrays and applies them.
    Serialises to pickle for deployment.
    """

    def __init__(self):
        self.mean_:  Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def fit_from_welford(
        self,
        acc: WelfordAccumulator,
        feature_names: List[str],
        scaler_type: str = SCALER_TYPE,
    ):
        """
        Finalise scaler from accumulated statistics.
        scaler_type 'standard'  → (x - mean) / std
                    'robust'    → (x - median) / IQR  (not supported from Welford;
                                   use 'standard' unless you do an extra median pass)
                    'minmax'    → (x - min) / (max - min)  (needs separate min/max pass)
        """
        if scaler_type not in ("standard",):
            logger.warning(
                f"Scaler type '{scaler_type}' not supported in streaming mode. "
                "Falling back to 'standard'."
            )
        self.mean_         = acc.mean.astype(np.float32)
        self.scale_        = acc.scale_.astype(np.float32)
        self.feature_names = feature_names
        logger.info(f"Scaler fitted on {acc.n} samples, {len(feature_names)} features")

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Scaler not fitted yet"
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Only call if you somehow have everything in memory — convenience method
        self.mean_  = X.mean(axis=0).astype(np.float32)
        std         = X.std(axis=0).astype(np.float32)
        std[std==0] = 1.0
        self.scale_ = std
        return self.transform(X)

    def save(self, path: str = SCALER_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Scaler saved → {path}")

    @classmethod
    def load(cls, path: str = SCALER_PATH) -> "CareWatchScaler":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Scaler loaded from {path}")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 — Fit scaler (streaming)
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler_streaming(
    feature_generator: Generator[pd.DataFrame, None, None],
    label_col: str = "label_int",
) -> "CareWatchScaler":
    """
    Stream all feature chunks, accumulate statistics, return fitted scaler.
    Does NOT transform — just fits.
    """
    acc          = None
    feat_cols    = None
    total_chunks = 0

    for chunk in feature_generator:
        if chunk.empty:
            continue
        if feat_cols is None:
            feat_cols = [c for c in chunk.columns if c != label_col]
            acc       = WelfordAccumulator(len(feat_cols))

        X = chunk[feat_cols].values.astype(np.float64)
        # Replace NaN / Inf with 0 before accumulation
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        acc.update_batch(X)
        total_chunks += 1
        if total_chunks % 50 == 0:
            logger.info(f"  Fit scaler — {acc.n} windows accumulated…")

    if acc is None:
        raise RuntimeError("No data received from feature generator")

    scaler = CareWatchScaler()
    scaler.fit_from_welford(acc, feat_cols)
    scaler.save()
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2 — Transform + write feature cache
# ─────────────────────────────────────────────────────────────────────────────

def transform_and_cache(
    feature_generator: Generator[pd.DataFrame, None, None],
    scaler: "CareWatchScaler",
    label_col: str = "label_int",
    cache_dir: str = FEATURE_CACHE,
    overwrite: bool = True,
) -> int:
    """
    Stream feature chunks, apply scaler, write compressed Parquet shards.
    Returns total windows cached.
    """
    if overwrite:
        import shutil
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        USE_PA = True
    except ImportError:
        USE_PA = False
        logger.warning("pyarrow not found — writing CSV shards (slower)")

    shard_idx     = 0
    total_windows = 0
    buffer        = []
    BUFFER_SIZE   = PARQUET_ROW_GROUP

    def _flush(buf, idx):
        chunk_df = pd.concat(buf, ignore_index=True)
        if USE_PA:
            path = os.path.join(cache_dir, f"shard_{idx:06d}.parquet")
            chunk_df.to_parquet(path, index=False, compression="snappy")
        else:
            path = os.path.join(cache_dir, f"shard_{idx:06d}.csv")
            chunk_df.to_csv(path, index=False)
        return idx + 1

    for chunk in feature_generator:
        if chunk.empty:
            continue

        feat_cols = [c for c in chunk.columns if c != label_col]
        X = chunk[feat_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X)

        scaled_df = pd.DataFrame(X_scaled, columns=feat_cols)
        if label_col in chunk.columns:
            scaled_df[label_col] = chunk[label_col].values

        buffer.append(scaled_df)
        total_windows += len(scaled_df)

        if sum(len(b) for b in buffer) >= BUFFER_SIZE:
            shard_idx = _flush(buffer, shard_idx)
            buffer    = []

        if shard_idx % 10 == 0 and shard_idx > 0:
            logger.info(f"  Cache: {shard_idx} shards, {total_windows} windows")

    if buffer:
        _flush(buffer, shard_idx)

    # Save feature column names for deployment
    with open(FEAT_COL_PATH, "wb") as f:
        pickle.dump(scaler.feature_names, f)
    logger.info(f"Feature cache complete: {total_windows} windows → {cache_dir}")
    return total_windows


# ─────────────────────────────────────────────────────────────────────────────
# Stream the feature cache (for training)
# ─────────────────────────────────────────────────────────────────────────────

def stream_cache(
    cache_dir: str = FEATURE_CACHE,
    shuffle_shards: bool = True,
    label_col: str = "label_int",
    drop_unlabelled: bool = True,
) -> Generator[pd.DataFrame, None, None]:
    """
    Stream pre-computed feature shards from disk.
    Optionally shuffle shard order (inter-shard shuffle; intra-shard is not done
    to preserve temporal structure — XGBoost/LightGBM don't need it).
    """
    import glob
    parquet_shards = sorted(glob.glob(os.path.join(cache_dir, "*.parquet")))
    csv_shards     = sorted(glob.glob(os.path.join(cache_dir, "*.csv")))
    shards         = parquet_shards + csv_shards

    if not shards:
        raise FileNotFoundError(f"Feature cache empty: {cache_dir}")

    if shuffle_shards:
        np.random.shuffle(shards)

    for path in shards:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if drop_unlabelled and label_col in df.columns:
            df = df[df[label_col] >= 0]

        if not df.empty:
            yield df


def load_feature_columns() -> List[str]:
    if not os.path.exists(FEAT_COL_PATH):
        raise FileNotFoundError("Run the pipeline first to generate feature_columns.pkl")
    with open(FEAT_COL_PATH, "rb") as f:
        return pickle.load(f)
