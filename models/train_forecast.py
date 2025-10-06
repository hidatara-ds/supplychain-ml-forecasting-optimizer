from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

ART = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)
FEAT = Path("data/processed/weekly_features.parquet")

try:
    import lightgbm as lgb
    LGB_OK = True
except Exception:
    LGB_OK = False
    from sklearn.ensemble import RandomForestRegressor as RFR

if __name__ == "__main__":
    df = pd.read_parquet(FEAT)
    y = df["units_sold"]
    X = df.drop(columns=["units_sold","store_id","product_id","year","week"]) 

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    if LGB_OK:
        model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31)
        model.fit(Xtr, ytr)
    else:
        model = RFR(n_estimators=300, random_state=42)
        model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    print(f"MAE: {mae:.3f}")

    joblib.dump(model, ART/"model_lgbm.pkl")
    # simple global stats
    stats = {"mean": float(y.mean()), "std": float(y.std())}
    (ART/"demand_stats.json").write_text(__import__("json").dumps(stats, indent=2))
    print("Saved model and stats to models/artifacts/")


