import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

ARTIF_DIR = Path("models/artifacts")
ARTIF_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIF_DIR / "model_lgbm.pkl"
MEAN_STD_PATH = ARTIF_DIR / "demand_stats.json"

try:
    import lightgbm as lgb
    LGB_OK = True
except Exception:
    LGB_OK = False
    from sklearn.ensemble import RandomForestRegressor as RFR

FEATURES_PATH = Path("data/processed/weekly_features.parquet")


def _load_model():
    if MODEL_PATH.exists():
        import joblib
        return joblib.load(MODEL_PATH)
    return None


def _load_stats():
    if MEAN_STD_PATH.exists():
        return json.loads(MEAN_STD_PATH.read_text())
    return {"mean": 5.0, "std": 2.0}


def _naive_forecast(h, mean, std):
    """
    Global baseline jika tidak ada data historis per SKU-location.
    """
    base = np.full(h, mean)
    # simple seasonal-ish bump last weeks of year
    base[-2:] = mean + 0.5 * std
    return base.clip(min=0).tolist()


def _naive_and_seasonal_from_history(sub: pd.DataFrame, h: int, mean: float, std: float):
    """
    Baseline per SKU-location:
    - Naive:       y_hat(t) = y(t-1)  -> ulangi nilai units_sold terakhir untuk semua horizon.
    - Seasonal:    y_hat(t) = y(t-52) -> pakai lag_52 (jika tersedia) sebagai seasonal naive mingguan.
    """
    if sub is None or len(sub) == 0:
        # fallback ke global naive jika tidak ada riwayat
        return _naive_forecast(h, mean, std), None

    sub = sub.sort_values(["year", "week"])

    # Naive: last value
    last_units = float(sub["units_sold"].iloc[-1])
    naive = [float(max(0.0, last_units))] * h

    # Seasonal naive: gunakan lag_52 jika ada dan tidak NaN
    seasonal = None
    if "lag_52" in sub.columns:
        val = sub["lag_52"].iloc[-1]
        if not pd.isna(val):
            seasonal_val = float(val)
            seasonal = [float(max(0.0, seasonal_val))] * h

    return naive, seasonal


def forecast_batch(pairs, horizon):
    model = _load_model()
    stats = _load_stats()
    mean, std = stats.get("mean", 5.0), stats.get("std", 2.0)

    df = None
    if FEATURES_PATH.exists():
        df = pd.read_parquet(FEATURES_PATH)

    outputs = []
    for p in pairs:
        sid = p.get("store_id", "S001")
        pid = p.get("product_id", "P001")

        sub = None
        if df is not None:
            sub = df[(df.store_id == sid) & (df.product_id == pid)]

        if model is None:
            # Jika belum ada model terlatih, pakai baseline:
            # - seasonal naive jika tersedia, jika tidak fallback ke naive / global.
            naive, seasonal = _naive_and_seasonal_from_history(sub, horizon, mean, std)
            fc = seasonal if seasonal is not None else naive
        else:
            # minimal feature pattern: last known week features
            if sub is None or len(sub) == 0:
                fc = _naive_forecast(horizon, mean, std)
            else:
                sub_sorted = sub.sort_values(["year", "week"])
                X_last = sub_sorted.tail(1).drop(
                    columns=["units_sold", "store_id", "product_id", "year", "week"]
                )
                preds = []
                xcur = X_last.iloc[0].values.reshape(1, -1)
                for _ in range(horizon):
                    y = model.predict(xcur)[0]
                    preds.append(float(max(0.0, y)))
                fc = preds

        outputs.append({"store_id": sid, "product_id": pid, "forecast": fc})
    return outputs


