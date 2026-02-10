"""
Training entrypoints for forecasting models with proper time-based splits.

Time split (weekly):
- Train : all weeks up to T-8
- Val   : (T-8, T-4]
- Test  : (T-4, T]   → dipakai terutama oleh modul evaluate.

Tidak ada random split; semua berdasarkan urutan waktu.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ..common.config import PROCESSED_DIR


FEAT_PATH = PROCESSED_DIR / "weekly_features.parquet"
ART_DIR = Path("models/artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ART_DIR / "model_lgbm.pkl"
STATS_PATH = ART_DIR / "demand_stats.json"


def _make_time_splits(df: pd.DataFrame):
    """
    Buat train/val/test split berbasis waktu (year, week), tanpa random.
    """
    if "time_key" not in df.columns:
        df = df.copy()
        df["time_key"] = df["year"] * 100 + df["week"]

    weeks = sorted(df["time_key"].unique())
    n = len(weeks)

    if n >= 12:
        # Train: semua kecuali 8 minggu terakhir
        train_weeks = weeks[: n - 8]
        # Val: 4 minggu sebelum 4 minggu terakhir
        val_weeks = weeks[n - 8 : n - 4]
        # Test: 4 minggu terakhir
        test_weeks = weeks[n - 4 :]
    else:
        # Fallback untuk dataset sangat pendek: 60/20/20 tetap berurutan waktu
        n_train = max(1, int(n * 0.6))
        n_val = max(1, int(n * 0.2))
        train_weeks = weeks[:n_train]
        val_weeks = weeks[n_train : n_train + n_val]
        test_weeks = weeks[n_train + n_val :]

    train_mask = df["time_key"].isin(train_weeks)
    val_mask = df["time_key"].isin(val_weeks)
    test_mask = df["time_key"].isin(test_weeks)

    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()


def train():
    """
    Train forecasting model dengan time-based split dan simpan artefak.

    Output:
    - `models/artifacts/model_lgbm.pkl`
    - `models/artifacts/demand_stats.json`
    - Ringkasan MAE di terminal (val dan test).
    """
    if not FEAT_PATH.exists():
        raise FileNotFoundError(
            f"{FEAT_PATH} tidak ditemukan. Jalankan dulu ETL: `python etl/build_features.py`."
        )

    df = pd.read_parquet(FEAT_PATH)

    # Label & fitur
    label_col = "units_sold"
    id_cols = ["store_id", "product_id", "year", "week"]
    feature_cols = [c for c in df.columns if c not in id_cols + [label_col]]

    train_df, val_df, test_df = _make_time_splits(df)

    Xtr = train_df[feature_cols]
    ytr = train_df[label_col]

    Xval = val_df[feature_cols]
    yval = val_df[label_col]

    Xte = test_df[feature_cols]
    yte = test_df[label_col]

    # Model: LightGBM jika ada, fallback ke RandomForest
    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31)
        model.fit(Xtr, ytr)
    except Exception:
        from sklearn.ensemble import RandomForestRegressor as RFR

        model = RFR(n_estimators=300, random_state=42)
        model.fit(Xtr, ytr)

    # Evaluasi kasar di val & test (MAE)
    if len(Xval) > 0:
        pred_val = model.predict(Xval)
        mae_val = mean_absolute_error(yval, pred_val)
        print(f"Val MAE (time-based): {mae_val:.3f} (n={len(Xval)})")

    if len(Xte) > 0:
        pred_test = model.predict(Xte)
        mae_test = mean_absolute_error(yte, pred_test)
        print(f"Test MAE (time-based): {mae_test:.3f} (n={len(Xte)})")

    # Simpan model
    joblib.dump(model, MODEL_PATH)

    # Simpan statistik global untuk baseline
    stats = {
        "mean": float(df[label_col].mean()),
        "std": float(df[label_col].std()),
    }
    STATS_PATH.write_text(__import__("json").dumps(stats, indent=2))

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved stats to {STATS_PATH}")


if __name__ == "__main__":
    train()

