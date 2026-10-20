.PHONY: setup notebook check paths

setup:
	bash scripts/bootstrap_env.sh

notebook:
	. .venv/bin/activate && jupyter lab notebooks/

check:
	. .venv/bin/activate && python -c "import pandas, numpy, sklearn, statsmodels, matplotlib, seaborn, requests, dotenv, fredapi; print('imports_ok')"

paths:
	@mkdir -p data/raw data/processed notebooks/outputs/latest/figures notebooks/outputs/latest/tables
	@echo "paths ready"
