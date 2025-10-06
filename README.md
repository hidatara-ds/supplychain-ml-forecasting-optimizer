# Supply Chain ML: Forecasting + Replenishment Optimizer

**What**: End-to-end retail supply chain demo. Generate dummy data → train demand forecast → compute replenishment via LP → serve via FastAPI.

**Endpoints**
- `GET /health`
- `POST /forecast` → body: `{ "horizon_weeks": 8, "pairs": [{"store_id":"S001","product_id":"P001"}] }`
- `POST /replenish` → body: `{ "target_service": 0.95, "capacity": 50000 }`

**Quickstart**
```bash
make setup
make data
make train
make run
# in another shell
curl http://localhost:8000/health
```

**Docker**
```bash
docker build -t scm-ml .
docker run -p 8000:8000 scm-ml
```

**Structure**
```
approot/
  app/            # FastAPI app & services
  etl/            # dummy data + features + load
  models/         # training & artifacts
  scripts/        # backtest/sim
  tests/          # pytest
```

**Roadmap**
- ETA/lead-time module
- Anomaly detection
- AWS deploy notes (ECS/Lambda)

# supplychain-ml-forecasting-optimizer