import numpy as np
import pandas as pd
from scipy.optimize import linprog
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

# Simplified replenishment: meet need = forecast + safety - on_hand - on_order, with budget(capacity)

def compute_replenishment(target_service: float = 0.95, capacity: float = 50000.0):
    price = 50.0  # flat unit price for demo
    safety_z = 1.64 if target_service >= 0.95 else 1.28

    inv = pd.read_parquet(PROCESSED_DIR / "inventory_latest.parquet")
    fc = pd.read_parquet(PROCESSED_DIR / "forecast_baseline.parquet")

    # align
    df = inv.merge(fc, on=["store_id","product_id"], how="left")
    df["forecast_next"] = df["forecast_next"].fillna(5.0)

    # safety stock proxy from volatility
    df["safety"] = safety_z * df["demand_std"].fillna(2.0)

    need = np.maximum(0.0, df["forecast_next"].values + df["safety"].values - df["on_hand"].values - df["on_order"].values)
    n = len(need)
    prices = np.full(n, price)

    # minimize cost: c·x, subject to price·x <= capacity, x >= need
    c = prices
    A = np.vstack([prices])
    b = [capacity]
    bounds = [(need[i], None) for i in range(n)]
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    qty = res.x if res.success else need

    df_out = df[["store_id","product_id"]].copy()
    df_out["order_qty"] = np.maximum(0, np.floor(qty)).astype(int)
    df_out["unit_price"] = price
    df_out["cost"] = df_out["order_qty"] * df_out["unit_price"]
    return df_out.head(100).to_dict(orient="records")


