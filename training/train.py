"""
training/train.py — SereneCare Model Training Script
=====================================================
Run the full CareWatch training pipeline:
  1. Ingest raw data (CSV / Parquet / Video)
  2. Extract features (sliding window + physics-based)
  3. Fit scaler (streaming Welford's algorithm)
  4. Transform & cache features to Parquet shards
  5. Train model (LightGBM / XGBoost / SGD / LSTM)

Usage:
    python -m training.train                   # defaults: lgbm
    python -m training.train --model xgb       # XGBoost
    python -m training.train --model sgd       # SGD incremental
    python -m training.train --model lstm      # LSTM (needs GPU)
"""

import argparse
import logging
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, OUTPUT_DIR, FEATURE_CACHE, MODEL_TYPE
from src.ml.ingestion import stream_data
from src.ml.features  import process_chunk
from src.ml.scaler    import fit_scaler_streaming, transform_and_cache
from src.ml.trainer   import train


def feature_generator():
    """Stream raw data → feature engineering pipeline."""
    for chunk in stream_data():
        feat = process_chunk(chunk)
        if not feat.empty:
            yield feat


def main():
    parser = argparse.ArgumentParser(description="SereneCare Model Training")
    parser.add_argument("--model", default=MODEL_TYPE,
                        choices=["lgbm", "xgb", "sgd", "lstm"],
                        help="Model type to train")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )
    logger = logging.getLogger("training")

    logger.info("=" * 60)
    logger.info("SereneCare — CareWatch Training Pipeline")
    logger.info("=" * 60)

    # Step 1+2: Fit scaler (pass 1 over data)
    logger.info("Pass 1 — Fitting scaler (streaming)…")
    scaler = fit_scaler_streaming(feature_generator())

    # Step 3: Transform + cache (pass 2 over data)
    logger.info("Pass 2 — Transforming & caching features…")
    n_windows = transform_and_cache(feature_generator(), scaler)
    logger.info(f"Cached {n_windows} feature windows → {FEATURE_CACHE}")

    # Step 4: Train model
    logger.info(f"Training model: {args.model}")
    model = train(args.model, FEATURE_CACHE)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
