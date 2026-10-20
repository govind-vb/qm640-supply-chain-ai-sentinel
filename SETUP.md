# Local Setup (macOS)

Run the Global Supply Chain AI Sentinel notebooks on your machine. This matches the submitted synopsis sources and RQs.

## 1. Prerequisites

- Python 3.10+ (`python3 --version`)
- Free API keys:
  - FRED: https://fred.stlouisfed.org/docs/api/api_key.html
  - EIA: https://www.eia.gov/opendata/register.php

## 2. Bootstrap

From the course folder:

```bash
cd github_scaffold
bash scripts/bootstrap_env.sh
```

Or:

```bash
cd github_scaffold
make setup
```

This creates `.venv`, installs packages, registers Jupyter kernel `qm640-sentinel`, and copies `.env.example` to `.env` if needed.

## 3. Add API keys

Edit `github_scaffold/.env`:

```bash
FRED_API_KEY=your_key_here
EIA_API_KEY=your_key_here
```

Never commit `.env`.

## 4. Start Jupyter

```bash
cd github_scaffold
source .venv/bin/activate
jupyter lab notebooks/
```

In each notebook, choose kernel **Python (qm640-sentinel)**.

## 5. Run order

1. `00_setup_and_config.ipynb`
2. `01_collect_daily_panel.ipynb`
3. `02_cleaning_and_eda.ipynb`
4. `03_sample_size.ipynb`
5. `04_rq1_bottleneck_forecast.ipynb`
6. `05_rq2_regime_classification.ipynb`
7. `06_rq3_fusion_omega.ipynb`
8. `07_rq4_contagion_vecm.ipynb`

See [RUNBOOK.md](RUNBOOK.md) for outputs and GitHub push steps.

## 6. Quick health check

```bash
cd github_scaffold
make check
```

## Troubleshooting

- **Kernel missing:** re-run `python -m ipykernel install --user --name qm640-sentinel --display-name "Python (qm640-sentinel)"` inside the venv.
- **Import errors:** `source .venv/bin/activate && pip install -r requirements.txt`
- **API failures:** confirm keys in `.env`, then re-run notebook `01`. Partial source failures are logged; do not invent synthetic market history.
- **FinBERT optional:** RQ2 works with a GDELT/lexicon baseline first. Install `torch` and `transformers` later if you want FinBERT.
