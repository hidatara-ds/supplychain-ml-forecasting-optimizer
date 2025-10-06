from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_forecast():
    r = client.post("/forecast", json={"horizon_weeks": 4, "pairs": [{"store_id":"S001","product_id":"P001"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["horizon_weeks"] == 4
    assert len(body["forecasts"]) == 1


