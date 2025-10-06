from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

# master tables
products = []
for i in range(1, 51):
    products.append({"product_id": f"P{i:03d}", "category": np.random.choice(["Footwear","Apparel","Accessories"]), "brand": "Nike", "cost": float(np.random.uniform(15,35)), "price": float(np.random.uniform(40,120))})

stores = []
for i in range(1, 21):
    stores.append({"store_id": f"S{i:03d}", "region": np.random.choice(["NW","SW","NE","SE"]), "size_tier": np.random.choice(["S","M","L"])})

# calendar (104 weeks)
weeks = pd.date_range("2023-01-02", periods=104, freq="W-MON")
cal = pd.DataFrame({"date": weeks})
cal["year"] = cal["date"].dt.year
cal["week"] = cal["date"].dt.isocalendar().week.astype(int)
cal["is_holiday"] = cal["week"].isin([47,48,49,50,51,52]).astype(int)
cal.to_csv(RAW/"calendar.csv", index=False)

pd.DataFrame(products).to_csv(RAW/"products.csv", index=False)
pd.DataFrame(stores).to_csv(RAW/"stores.csv", index=False)

# sales weekly synthetic
rows = []
for s in stores:
    for p in products:
        base = np.random.uniform(3, 15)
        season_amp = np.random.uniform(0.5, 3.0)
        noise = np.random.normal
        for _, r in cal.iterrows():
            w = r["week"]
            season = season_amp * np.sin(2*np.pi*w/52)
            holiday_boost = 3.0 if r["is_holiday"] else 0.0
            units = max(0, base + season + holiday_boost + noise(0, 2))
            rows.append({"date": r["date"].date(), "store_id": s["store_id"], "product_id": p["product_id"], "units_sold": round(units)})

sales = pd.DataFrame(rows)
sales.to_csv(RAW/"sales.csv", index=False)

# inventory snapshot (latest week)
inv_rows = []
for s in stores:
    for p in products:
        inv_rows.append({"store_id": s["store_id"], "product_id": p["product_id"], "on_hand": int(np.random.uniform(0, 60)), "on_order": int(np.random.uniform(0, 20)), "demand_std": float(np.random.uniform(1.0, 4.0))})

pd.DataFrame(inv_rows).to_csv(RAW/"inventory_latest.csv", index=False)
print("Dummy data generated into data/raw/")


