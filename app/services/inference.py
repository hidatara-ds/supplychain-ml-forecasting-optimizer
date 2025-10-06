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
    base = np.full(h, mean)
    # simple seasonal-ish bump last weeks of year
    base[-2:] = mean + 0.5 * std
    return base.clip(min=0).tolist()


def forecast_batch(pairs, horizon):
    model = _load_model()
    stats = _load_stats()
    mean, std = stats.get("mean", 5.0), stats.get("std", 2.0)

    # For brevity, if model not trained yet, return naive
    outputs = []
    for p in pairs:
        sid = p.get("store_id", "S001")
        pid = p.get("product_id", "P001")
        if model is None:
            fc = _naive_forecast(horizon, mean, std)
        else:
            # minimal feature pattern: last known week features averaged
            df = pd.read_parquet(FEATURES_PATH)
            sub = df[(df.store_id == sid) & (df.product_id == pid)]
            if len(sub) == 0:
                fc = _naive_forecast(horizon, mean, std)
            else:
                X_last = sub.sort_values(["year", "week"]).tail(1).drop(columns=["units_sold","store_id","product_id","year","week"]) 
                preds = []
                xcur = X_last.iloc[0].values.reshape(1, -1)
                for _ in range(horizon):
                    y = model.predict(xcur)[0]
                    preds.append(float(max(0.0, y)))
                fc = preds
        outputs.append({"store_id": sid, "product_id": pid, "forecast": fc})
    return outputs


