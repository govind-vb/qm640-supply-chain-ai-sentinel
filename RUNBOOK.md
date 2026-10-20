# Week 4 Runbook

Synopsis-locked interim analysis for **Global Supply Chain AI Sentinel**.

Public repo: https://github.com/govind-vb/qm640-supply-chain-ai-sentinel

## Frozen design (do not reinvent)

| Item | Value |
|---|---|
| RQ1 | Bottleneck forecast (lag baseline, Ridge/RF) |
| RQ2 | Regime classification (news/GDELT; FinBERT optional) |
| RQ3 | Fusion vs single-modality; Omega-style compare |
| RQ4 | Granger/VECM energy to agriculture |
| Final N | 347 (max of RQ minima) |
| Sources | PortWatch, GDELT, EIA, FRED (no Kaggle) |

## Expected outputs after a full run

```text
data/raw/                 # source extracts
data/processed/
  daily_panel.csv
  daily_panel_clean.csv
notebooks/outputs/latest/
  figures/                # report-ready PNGs
  tables/                 # cleaning_log, eda_summary, model tables
  metrics_rq1.csv
  metrics_rq2.csv
  metrics_rq3.csv
  metrics_rq4.csv
```

## After each successful run

1. Confirm figures and metrics exist under `notebooks/outputs/latest/`.
2. Append an entry to `../shared/CHANGELOG.md` and `CHANGELOG.md` in this repo.
3. Push to GitHub (**never** push `.env`):

```bash
cd github_scaffold
git add data/raw data/processed notebooks CHANGELOG.md src requirements.txt
git status   # confirm .env is not staged
git commit -m "Add extracted panel and notebook outputs for interim analysis"
git push origin main
```

4. Share with the report draft step:
   - row count of `daily_panel_clean.csv`
   - whether N >= 347
   - the four `metrics_rq*.csv` files
   - key figures from `outputs/latest/figures/`

## Iteration convention

- Optimization runs: write metrics to `notebooks/outputs/iter_02/`, `iter_03/`, …
- Copy the chosen iteration into `latest/` before drafting the interim PDF
- Log every meaningful change in `shared/CHANGELOG.md`

## Interim report handoff

When outputs are ready, draft the mentor interim PDF from:

- Week4 interim template
- Week4 rubric / interim checklist
- Notebook tables and figures
- Synopsis problem/RQs/literature base (standalone rewrite, not a paste of the synopsis)
