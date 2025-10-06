from fastapi import FastAPI
from app.services.inference import forecast_batch
from app.services.optimizer import compute_replenishment

app = FastAPI(title="SupplyChain ML API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast")
def forecast(req: dict):
    pairs = req.get("pairs", [])
    horizon = int(req.get("horizon_weeks", 8))
    preds = forecast_batch(pairs, horizon)
    return {"horizon_weeks": horizon, "forecasts": preds}

@app.post("/replenish")
def replenish(req: dict):
    target_service = float(req.get("target_service", 0.95))
    capacity = float(req.get("capacity", 50000.0))
    result = compute_replenishment(target_service=target_service, capacity=capacity)
    return {"target_service": target_service, "capacity": capacity, "orders": result}


