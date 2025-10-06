from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw"); PROC = Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)

sales = pd.read_csv(RAW/"sales.csv", parse_dates=["date"]) 
cal = pd.read_csv(RAW/"calendar.csv", parse_dates=["date"]) 
products = pd.read_csv(RAW/"products.csv")
stores = pd.read_csv(RAW/"stores.csv")

# weekly aggregate is already weekly; derive keys
sales["year"] = sales["date"].dt.year
sales["week"] = sales["date"].dt.isocalendar().week.astype(int)

# join calendar flags
sales = sales.merge(cal[["date","is_holiday"]], on="date", how="left")

# simple price proxy: map by product
price_map = products.set_index("product_id")["price"].to_dict()
sales["price"] = sales["product_id"].map(price_map)

# lag features per store-product
sales = sales.sort_values(["store_id","product_id","date"]).reset_index(drop=True)
for lag in [1,2,4,52]:
    sales[f"lag_{lag}"] = sales.groupby(["store_id","product_id"])['units_sold'].shift(lag)

# rolling means
for win in [4,8,12]:
    sales[f"rollmean_{win}"] = sales.groupby(["store_id","product_id"])['units_sold'].shift(1).rolling(win).mean()

# cyclical seasonality
sales["woy"] = sales["week"]
sales["sin_woy"] = np.sin(2*np.pi*sales["woy"]/52)
sales["cos_woy"] = np.cos(2*np.pi*sales["woy"]/52)

# drop nas for training rows
feat_cols = ["is_holiday","price","lag_1","lag_2","lag_4","rollmean_4","rollmean_8","rollmean_12","sin_woy","cos_woy"]
train_df = sales.dropna(subset=feat_cols + ["units_sold"]).copy()

# save features
train_df[['store_id','product_id','year','week','units_sold'] + feat_cols].to_parquet(PROC/"weekly_features.parquet", index=False)

# also create baseline forecast_next for optimizer input (mean of last 4)
last4 = sales.groupby(["store_id","product_id"]).tail(4)
base_fc = last4.groupby(["store_id","product_id"])['units_sold'].mean().reset_index(name="forecast_next")
base_fc.to_parquet(PROC/"forecast_baseline.parquet", index=False)

# inventory latest parquet
inv = pd.read_csv(RAW/"inventory_latest.csv")
inv.to_parquet(PROC/"inventory_latest.parquet", index=False)

print("Features & processed artifacts saved to data/processed/")


