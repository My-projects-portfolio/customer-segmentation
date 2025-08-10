.PHONY: setup test lint typecheck train retrain run-api run-ui dvc-repro smoke

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

lint:
	black --check . && isort --check-only . && flake8 .

typecheck:
	mypy pipelines app

test:
	pytest -q

train:
	python pipelines/ingest.py \
		--raw_path data/raw/online_retail.csv \
		--out_path data/interim/transactions.parquet
	python pipelines/features.py --in_path data/interim/transactions.parquet --out_path data/processed/rfm.parquet
	python pipelines/train.py --rfm_path data/processed/rfm.parquet --export_path artifacts/models
	python pipelines/evaluate.py --rfm_path data/processed/rfm.parquet --model_dir artifacts/models

retrain:
	python pipelines/ingest.py --raw_path data/raw/online_retail.csv --out_path data/interim/transactions.parquet --full True
	python pipelines/features.py --in_path data/interim/transactions.parquet --out_path data/processed/rfm.parquet
	python pipelines/train.py --rfm_path data/processed/rfm.parquet --export_path artifacts/models
	python pipelines/evaluate.py --rfm_path data/processed/rfm.parquet --model_dir artifacts/models
	python pipelines/register.py --model_dir artifacts/models

run-api:
	uvicorn app.api.main:app --reload --port 8000

run-ui:
	streamlit run app/ui/streamlit_app.py

dvc-repro:
	dvc repro

smoke:
	python pipelines/train.py --rfm_path data/processed/rfm.parquet --export_path artifacts/models --sample 0.01 --ci_gate True
