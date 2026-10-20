# Capstone Iteration Log

Low-touch log for notebook, data, and interim-report changes.  
Project: Global Supply Chain AI Sentinel (QM640)  
GitHub: https://github.com/govind-vb/qm640-supply-chain-ai-sentinel  
Synopsis lock: RQ1–RQ4, Final N = 347, PortWatch / GDELT / EIA / FRED only (no Kaggle)

## Entry format

```markdown
## YYYY-MM-DD
- Change: <what changed>
- Why: <reason>
- Files: <paths>
- Result: <pending | better | worse | neutral> <metric note>
```

## 2026-10-20

- Change: Initial Week4 local setup pack, shared changelog, and synopsis-locked notebooks 00–07 scaffolded
- Why: Enable local execution and interim analysis without changing the submitted research design
- Files: shared/CHANGELOG.md, Week4/SETUP.md, Week4/RUNBOOK.md, Week4/INTERIM_DRAFT_HANDOFF.md, github_scaffold/requirements.txt, github_scaffold/scripts/*, github_scaffold/notebooks/*.ipynb, github_scaffold/src/fetch_panel.py
- Result: setup complete

- Change: First successful public-source collection + RQ1–RQ4 pipeline run; push extracts/outputs; draft interim mentor PDF/DOCX
- Why: Satisfy interim data-availability on public GitHub and produce mentor draft from real metrics/figures
- Files: data/raw/*, data/processed/daily_panel*.csv, notebooks/outputs/latest/*, Week4/Govind_VB_Interim_Report_Mentor_Draft.docx|.pdf, scripts/run_analysis_pipeline.py
- Result: cleaned N≈2066 (> Final N 347); RQ1 best=lag_baseline; RQ2 market-only higher F1 than news proxy; RQ3 spatial_only lowest Omega RMSE; RQ4 Granger WTI→grain not significant at 5% in first pass; PortWatch CSV still pending (documented stand-ins)

- Change: Executed notebooks 00–07 via Jupyter (inplace outputs) and regenerated interim PDF/DOCX from those artifacts
- Why: Prior run used scripts/run_analysis_pipeline.py equivalent; user requested notebook execution with outputs in report
- Files: github_scaffold/notebooks/0*.ipynb, notebooks/outputs/latest/*, Week4/Govind_VB_Interim_Report_Mentor_Draft.pdf|.docx, Week4/notebooks/
- Result: all 8 notebooks executed with cell outputs; figures fig01–fig07 and metrics_rq1–rq4 refreshed in report

