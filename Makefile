PY=python
PIP=pip

setup:
	$(PIP) install -r requirements.txt

data:
	$(PY) etl/generate_dummy.py
	$(PY) etl/build_features.py

train:
	$(PY) models/train_forecast.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

format:
	$(PY) -m pip install ruff black
	ruff check --fix . || true
	black .

