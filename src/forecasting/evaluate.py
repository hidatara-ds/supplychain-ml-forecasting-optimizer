"""
Evaluation utilities for forecasting models (metrics, backtests, dsb.).

Time-based split (weekly):
- Train : all weeks up to T-8
- Val   : (T-8, T-4]
- Test  : (T-4, T]  → fokus evaluasi.

Evaluasi minimal:
- Hitung baseline naive & seasonal-naive (berbasis lag_1, lag_52).
- (Opsional) Hitung prediksi model jika artefak model tersedia.
- Simpan `data/processed/predictions.csv`.
- Cetak ringkasan metrik (WAPE + MASE) ke terminal.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import pandas as pd

from ..common.config import PROCESSED_DIR
from ..common.metrics import mase, wape


FEAT_PATH = PROCESSED_DIR / "weekly_features.parquet"
ART_DIR = Path("models/artifacts")
MODEL_PATH = ART_DIR / "model_lgbm.pkl"
STATS_PATH = ART_DIR / "demand_stats.json"
PRED_PATH = PROCESSED_DIR / "predictions.csv"


def _make_time_splits(df: pd.DataFrame):
    """
    Replikasi skema split dari `train.py`, berbasis (year, week).
    """
    if "time_key" not in df.columns:
        df = df.copy()
        df["time_key"] = df["year"] * 100 + df["week"]

    weeks = sorted(df["time_key"].unique())
    n = len(weeks)

    if n >= 12:
        train_weeks = weeks[: n - 8]
        val_weeks = weeks[n - 8 : n - 4]
        test_weeks = weeks[n - 4 :]
    else:
        n_train = max(1, int(n * 0.6))
        n_val = max(1, int(n * 0.2))
        train_weeks = weeks[:n_train]
        val_weeks = weeks[n_train : n_train + n_val]
        test_weeks = weeks[n_train + n_val :]

    train_mask = df["time_key"].isin(train_weeks)
    val_mask = df["time_key"].isin(val_weeks)
    test_mask = df["time_key"].isin(test_weeks)

    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()


def _load_model():
    if MODEL_PATH.exists():
        import joblib

        return joblib.load(MODEL_PATH)
    return None


def _load_stats(df: pd.DataFrame) -> Dict[str, float]:
    if STATS_PATH.exists():
        return json.loads(STATS_PATH.read_text())
    # fallback: hitung dari data
    mean = float(df["units_sold"].mean())
    std = float(df["units_sold"].std())
    return {"mean": mean, "std": std}


def evaluate() -> None:
    """
    Evaluasi model & baseline di test set (time-based).
    """
    if not FEAT_PATH.exists():
        raise FileNotFoundError(
            f"{FEAT_PATH} tidak ditemukan. Jalankan dulu ETL: `python etl/build_features.py`."
        )

    df = pd.read_parquet(FEAT_PATH)

    label_col = "units_sold"
    id_cols = ["store_id", "product_id", "year", "week"]
    feature_cols = [c for c in df.columns if c not in id_cols + [label_col]]

    train_df, val_df, test_df = _make_time_splits(df)

    if len(test_df) == 0:
        raise RuntimeError("Test split kosong; data historis belum cukup untuk skema T-8/T-4/T.")

    stats = _load_stats(df)
    global_mean = stats.get("mean", 5.0)

    model = _load_model()

    # --- Baseline predictions on test split (per-row, per-SKU-location) ---
    test_df = test_df.copy()
    test_df["y_true"] = test_df[label_col]

    # Naive: y_hat(t) = y(t-1) ≈ lag_1
    if "lag_1" in test_df.columns:
        test_df["y_pred_naive"] = test_df["lag_1"].fillna(global_mean)
    else:
        test_df["y_pred_naive"] = global_mean

    # Seasonal naive (weekly): y_hat(t) = y(t-52) ≈ lag_52
    if "lag_52" in test_df.columns:
        test_df["y_pred_seasonal"] = test_df["lag_52"].fillna(test_df["y_pred_naive"])
    else:
        test_df["y_pred_seasonal"] = test_df["y_pred_naive"]

    # Model predictions jika tersedia
    if model is not None:
        Xte = test_df[feature_cols]
        y_model = model.predict(Xte)
        test_df["y_pred_model"] = [float(max(0.0, v)) for v in y_model]

    # Simpan predictions.csv
    cols_for_output = id_cols + [
        "y_true",
        "y_pred_naive",
        "y_pred_seasonal",
    ]
    if "y_pred_model" in test_df.columns:
        cols_for_output.append("y_pred_model")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    test_df[cols_for_output].to_csv(PRED_PATH, index=False)

    # --- Metrics: per SKU-location & global ---
    methods = ["y_pred_naive", "y_pred_seasonal"]
    if "y_pred_model" in test_df.columns:
        methods.append("y_pred_model")

    results_global = {}
    results_per_pair: Dict[str, Dict[str, float]] = {}

    # Global WAPE per method
    for m in methods:
        results_global[m] = wape(test_df["y_true"], test_df[m])

    # Per SKU-location
    for (sid, pid), grp in test_df.groupby(["store_id", "product_id"]):
        key = f"{sid}|{pid}"
        y_true = grp["y_true"].tolist()

        # Insample untuk MASE: pakai seluruh histori kombinasi ini, sebelum test period
        hist = (
            df[(df["store_id"] == sid) & (df["product_id"] == pid)]
            .sort_values(["year", "week"])
        )
        insample = hist[label_col].tolist()

        pair_metrics: Dict[str, float] = {}
        for m in methods:
            y_hat = grp[m].tolist()
            pair_metrics[f"{m}_wape"] = wape(y_true, y_hat)
            pair_metrics[f"{m}_mase"] = mase(y_true, y_hat, insample, m=1)

        results_per_pair[key] = pair_metrics

    # --- Print summary to terminal ---
    print("=== Forecast Evaluation (time-based test split) ===")
    print(f"Test rows: {len(test_df)}, unique SKU-locations: {test_df[['store_id','product_id']].drop_duplicates().shape[0]}")
    print("\nGlobal WAPE (lower is better):")
    for m, val in results_global.items():
        print(f"  {m:15s}: {val:.4f}")

    # Tampilkan beberapa pasangan terburuk berdasarkan naive WAPE
    worst = sorted(
        results_per_pair.items(),
        key=lambda kv: kv[1]["y_pred_naive_wape"],
        reverse=True,
    )[:5]
    print("\nTop 5 worst SKU-locations by naive WAPE:")
    for key, metrics_dict in worst:
        print(f"  {key}: naive_WAPE={metrics_dict['y_pred_naive_wape']:.4f}")


if __name__ == "__main__":
    evaluate()

