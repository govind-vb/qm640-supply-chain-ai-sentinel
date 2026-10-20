# Global Supply Chain AI Sentinel (QM 640 Capstone)

Student: Govind V B  
Mentor: Sharath Srivatsa  
Course: QM640 Data Analytics Capstone, Walsh College

Public repository for **extracted data**, data dictionary, and analysis notebooks for the submitted synopsis topic.

Repo: https://github.com/govind-vb/qm640-supply-chain-ai-sentinel

## Quick start (local)

```bash
bash scripts/bootstrap_env.sh
# edit .env with FRED_API_KEY and EIA_API_KEY
source .venv/bin/activate
jupyter lab notebooks/
```

Full instructions: see `SETUP.md` in the course `Week4/` folder (also summarized below).

## Notebook order

0. `00_setup_and_config.ipynb`
1. `01_collect_daily_panel.ipynb`
2. `02_cleaning_and_eda.ipynb`
3. `03_sample_size.ipynb`
4. `04_rq1_bottleneck_forecast.ipynb`
5. `05_rq2_regime_classification.ipynb`
6. `06_rq3_fusion_omega.ipynb`
7. `07_rq4_contagion_vecm.ipynb`

## Data policy

- No Kaggle datasets
- Sources: IMF PortWatch (when CSV available), GDELT, EIA, FRED
- No personally identifiable information

## Repository layout

```text
.
├── README.md
├── CHANGELOG.md
├── data_dictionary.md
├── requirements.txt
├── .env.example
├── Makefile
├── scripts/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 00_...ipynb ... 07_...ipynb
│   └── outputs/latest/{figures,tables}
└── src/
```

## After you run collection

Commit and push `data/raw`, `data/processed`, and `notebooks/outputs` so evaluators can access extracts here. Never commit `.env`.

## Iteration log

Record optimizations in `CHANGELOG.md`.
