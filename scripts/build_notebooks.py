#!/usr/bin/env python3
"""Generate synopsis-locked Week4 notebooks as .ipynb files."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
WEEK4_NB = Path(__file__).resolve().parents[2] / "Week4" / "notebooks"


def nb(cells: list[tuple[str, str]]) -> dict:
    out_cells = []
    for ctype, source in cells:
        src = source.strip("\n") + "\n"
        lines = src.splitlines(keepends=True)
        if ctype == "md":
            out_cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            out_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": lines,
                }
            )
    return {
        "cells": out_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (qm640-sentinel)",
                "language": "python",
                "name": "qm640-sentinel",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name: str, cells: list[tuple[str, str]]) -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    WEEK4_NB.mkdir(parents=True, exist_ok=True)
    payload = nb(cells)
    for dest in (NB_DIR / name, WEEK4_NB / name):
        dest.write_text(json.dumps(payload, indent=1))
        print("wrote", dest)


COMMON_SETUP = '''
import sys
from pathlib import Path
ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from project_paths import load_env, ensure_output_dirs
ROOT = load_env()
PATHS = ensure_output_dirs(ROOT)
print("ROOT", ROOT)
print({k: str(v) for k, v in PATHS.items()})
'''

# ---- 00 ----
write(
    "00_setup_and_config.ipynb",
    [
        ("md", """# 00 Setup and Config

Synopsis-locked project: **Global Supply Chain AI Sentinel** (QM640).

Run this first. Kernel: `Python (qm640-sentinel)`.

Checks: Python version, package imports, `.env` keys, writable paths."""),
        ("code", COMMON_SETUP),
        (
            "code",
            """
import os, platform
print("Python", sys.version)
print("Platform", platform.platform())
required = [
    "pandas", "numpy", "sklearn", "statsmodels", "matplotlib", "seaborn",
    "requests", "dotenv", "fredapi"
]
missing = []
for m in required:
    try:
        __import__(m if m != "sklearn" else "sklearn")
        print("OK", m)
    except Exception as e:
        missing.append((m, str(e)))
        print("MISSING", m, e)
print("FRED_API_KEY set:", bool(os.getenv("FRED_API_KEY")))
print("EIA_API_KEY set:", bool(os.getenv("EIA_API_KEY")))
print(
    "Ready to collect:",
    len(missing) == 0,
    "(FRED public CSV fallback works without key; EIA optional)",
)
if missing:
    raise SystemExit("Install missing packages via scripts/bootstrap_env.sh")
""",
        ),
        (
            "md",
            """If ready is True, continue to `01_collect_daily_panel.ipynb`.

Log setup issues in `../../shared/CHANGELOG.md` (course folder) or `../CHANGELOG.md`.""",
        ),
    ],
)

# ---- 01 ----
write(
    "01_collect_daily_panel.ipynb",
    [
        (
            "md",
            """# 01 Collect Daily Panel

Pulls public sources from the submitted synopsis: **FRED, EIA (optional), GDELT, PortWatch (when CSV available)**.

Writes:
- `data/raw/*.csv`
- `data/processed/daily_panel.csv`
- `data/raw/collection_meta.json`

No Kaggle. No invented market history.""",
        ),
        ("code", COMMON_SETUP),
        (
            "code",
            """
from fetch_panel import build_daily_panel
import pandas as pd
import json

panel, meta = build_daily_panel()
display(panel.tail())
print(panel.shape)
print(json.dumps(meta, indent=2))
panel.head(3)
""",
        ),
        (
            "md",
            """### Source quality check

If `portwatch` is `pending_real_csv`, download a PortWatch chokepoint export into `data/raw/` and re-run this notebook so `chokepoint_vessel_count` / congestion fields are real vessel indicators.""",
        ),
        (
            "code",
            """
meta_path = PATHS["raw"] / "collection_meta.json"
print(meta_path.read_text())
print("Columns:", list(panel.columns))
print("Date range:", panel["date"].min(), "->", panel["date"].max())
print("Non-null physical congestion:", panel["port_congestion_hours"].notna().sum())
""",
        ),
    ],
)

# ---- 02 ----
write(
    "02_cleaning_and_eda.ipynb",
    [
        (
            "md",
            """# 02 Cleaning and EDA

Creates cleaning log, report-ready figures, and `daily_panel_clean.csv`.

Every figure is saved under `notebooks/outputs/latest/figures/` with an insight note for the interim report.""",
        ),
        ("code", COMMON_SETUP),
        (
            "code",
            """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
panel_path = PATHS["processed"] / "daily_panel.csv"
df = pd.read_csv(panel_path, parse_dates=["date"]).sort_values("date")
print(df.shape)
df.head()
""",
        ),
        (
            "code",
            """
# Cleaning log
cleaning_rows = []
n0 = len(df)

# Drop rows with no date or no WTI (core market anchor)
before = len(df)
df = df.dropna(subset=["date", "wti_spot"])
cleaning_rows.append({
    "Issue": "Missing core market anchor",
    "Variables Affected": "date,wti_spot",
    "Detection Method": "isna",
    "Treatment Applied": f"drop {before-len(df)} rows",
    "Rationale": "Cannot model commodity outcomes without observed WTI",
})

# Duplicates
dup = df.duplicated(subset=["date"]).sum()
df = df.drop_duplicates(subset=["date"], keep="last")
cleaning_rows.append({
    "Issue": "Duplicates",
    "Variables Affected": "date",
    "Detection Method": "duplicated(date)",
    "Treatment Applied": f"drop {dup} duplicate dates",
    "Rationale": "One row per day for panel models",
})

# Winsorize extreme return outliers on WTI
ret = df["wti_spot"].pct_change()
lo, hi = ret.quantile(0.01), ret.quantile(0.99)
mask = ret.between(lo, hi) | ret.isna()
removed = (~mask).sum()
df = df.loc[mask | ret.isna()].copy()
cleaning_rows.append({
    "Issue": "Outliers",
    "Variables Affected": "wti_spot returns",
    "Detection Method": "1st/99th pct return fences",
    "Treatment Applied": f"remove {removed} extreme return days",
    "Rationale": "Limit distortion from bad ticks while keeping real crisis days inside fences",
})

# Forward/back fill limited gaps on key numeric columns
num_cols = df.select_dtypes("number").columns
miss_before = df[num_cols].isna().sum()
df[num_cols] = df[num_cols].ffill(limit=3).bfill(limit=1)
miss_after = df[num_cols].isna().sum()
cleaning_rows.append({
    "Issue": "Missing values",
    "Variables Affected": ",".join(num_cols[:8]) + ",...",
    "Detection Method": "isna sum",
    "Treatment Applied": "ffill limit 3, bfill limit 1",
    "Rationale": "Preserve chronology; avoid long leakage-style fills",
})

df = df.reset_index(drop=True)
clean_path = PATHS["processed"] / "daily_panel_clean.csv"
df.to_csv(clean_path, index=False)
log = pd.DataFrame(cleaning_rows)
log_path = PATHS["tables"] / "cleaning_log.csv"
log.to_csv(log_path, index=False)
print("Wrote", clean_path, "rows", len(df), "from", n0)
log
""",
        ),
        (
            "md",
            """### Figure 1. Missingness after cleaning

Insight: shows which synopsis variables still need coverage (especially PortWatch physical IVs).""",
        ),
        (
            "code",
            """
miss = df.isna().mean().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(8, 4.5))
miss.plot(kind="bar", ax=ax, color="#33415c")
ax.set_title("Figure 1. Share of missing values by variable")
ax.set_ylabel("Fraction missing")
ax.set_xlabel("Variable")
fig.tight_layout()
fig.savefig(PATHS["figures"] / "fig01_missingness.png", dpi=200)
plt.show()
""",
        ),
        (
            "md",
            """### Figure 2. WTI and congestion / stress series

Supports RQ1 physical-friction vs market stress timing.""",
        ),
        (
            "code",
            """
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(df["date"], df["wti_spot"], label="WTI spot", color="#1b263b")
ax2 = ax.twinx()
ax2.plot(df["date"], df["port_congestion_hours"], label="Congestion/stress", color="#e76f51", alpha=0.8)
ax.set_title("Figure 2. WTI vs congestion/stress indicator")
ax.set_xlabel("Date")
ax.set_ylabel("WTI USD")
ax2.set_ylabel("Congestion / stress")
fig.tight_layout()
fig.savefig(PATHS["figures"] / "fig02_wti_congestion.png", dpi=200)
plt.show()
""",
        ),
        (
            "md",
            """### Figure 3. Correlation heatmap (RQ-linked features)""",
        ),
        (
            "code",
            """
cols = [c for c in [
    "wti_spot","port_congestion_hours","chokepoint_vessel_count","vessel_draught_anomaly",
    "news_event_impact_score","finbert_risk_probability","bottleneck_index_5d",
    "volatility_regime","omega_ratio_proxy","grain_calendar_spread","commodity_roll_yield"
] if c in df.columns]
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(corr, ax=ax, cmap="vlag", center=0, annot=False)
ax.set_title("Figure 3. Correlation heatmap of synopsis model features")
fig.tight_layout()
fig.savefig(PATHS["figures"] / "fig03_corr_heatmap.png", dpi=200)
plt.show()
""",
        ),
        (
            "code",
            """
# EDA insight summary table for interim template
eda = pd.DataFrame([
    {"Figure/Table Reference": "Figure 1", "What it Shows": "Missingness by variable", "Key Insight": "Identify incomplete PortWatch vs market coverage", "Why it Matters (Link to RQ/objective)": "RQ1 physical IV completeness", "Decision/Next Step": "Add PortWatch CSV if vessel fields sparse"},
    {"Figure/Table Reference": "Figure 2", "What it Shows": "WTI vs congestion/stress", "Key Insight": "Co-movement during stress windows", "Why it Matters (Link to RQ/objective)": "RQ1/RQ3 timing", "Decision/Next Step": "Use lags in forecast models"},
    {"Figure/Table Reference": "Figure 3", "What it Shows": "Feature correlations", "Key Insight": "Which IVs move with targets", "Why it Matters (Link to RQ/objective)": "Feature selection RQ1-RQ3", "Decision/Next Step": "Drop near-duplicate predictors"},
])
eda.to_csv(PATHS["tables"] / "eda_insight_summary.csv", index=False)
eda
""",
        ),
    ],
)

# ---- 03 ----
write(
    "03_sample_size.ipynb",
    [
        (
            "md",
            """# 03 Sample Size (Synopsis Lock)

Reproduces Green / power calculations from the submitted synopsis.

**Final N = max(RQ1–RQ4) = 347**""",
        ),
        ("code", COMMON_SETUP),
        (
            "code",
            """
import math
import pandas as pd
import numpy as np

def green(k, alpha=0.05):
    # Green's rule N >= 50 + 8k
    return 50 + 8 * k

def corr_power_n(r=0.15, alpha=0.05, power=0.80):
    # Fisher z sample size used in synopsis for RQ3
    z_a = 1.96
    z_b = 0.84
    fisher = 0.5 * math.log((1 + r) / (1 - r))
    n = ((z_a + z_b) / fisher) ** 2 + 3
    return math.ceil(n)

rows = [
    {"Research Question": "RQ1", "Method": "Green", "Key parameters": "alpha=.05; k=6", "Minimum N": green(6)},
    {"Research Question": "RQ2", "Method": "Green", "Key parameters": "alpha=.05; k=8", "Minimum N": green(8)},
    {"Research Question": "RQ3", "Method": "Correlation power (Fisher z)", "Key parameters": "alpha=.05; power=.80; r=.15", "Minimum N": corr_power_n()},
    {"Research Question": "RQ4", "Method": "Green", "Key parameters": "alpha=.05; k=10", "Minimum N": green(10)},
]
tbl = pd.DataFrame(rows)
final_n = int(tbl["Minimum N"].max())
tbl.loc[len(tbl)] = {"Research Question": "Final N", "Method": "max(RQ1-RQ4)", "Key parameters": "Protects weakest-powered RQ", "Minimum N": final_n}
assert final_n == 347, final_n
tbl.to_csv(PATHS["tables"] / "sample_size.csv", index=False)
tbl
""",
        ),
        (
            "code",
            """
df = pd.read_csv(PATHS["processed"] / "daily_panel_clean.csv", parse_dates=["date"])
available_n = len(df)
print("Available N:", available_n)
print("Final N required:", 347)
print("Meets Final N:", available_n >= 347)
pd.DataFrame([{
    "available_n": available_n,
    "final_n_required": 347,
    "meets_requirement": available_n >= 347,
}]).to_csv(PATHS["tables"] / "sample_size_vs_available.csv", index=False)
""",
        ),
    ],
)

RQ_HEADER = COMMON_SETUP + '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
sns.set_theme(style="whitegrid", context="talk")
df = pd.read_csv(PATHS["processed"] / "daily_panel_clean.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
print(df.shape)
'''

write(
    "04_rq1_bottleneck_forecast.ipynb",
    [
        ("md", """# 04 RQ1 Bottleneck Forecast

Synopsis RQ1: predict forward bottleneck / congestion stress with spatial-temporal style features.

Models: naive lag baseline, Ridge, Random Forest. Validation: walk-forward time split."""),
        ("code", RQ_HEADER),
        (
            "code",
            """
# Feature set locked to synopsis physical + market lags
feature_cols = [c for c in [
    "chokepoint_vessel_count","port_congestion_hours","vessel_draught_anomaly",
    "commodity_roll_yield","wti_spot","natgas_hh"
] if c in df.columns and df[c].notna().sum() >= 50]
if not feature_cols:
    raise RuntimeError("No usable RQ1 features after coverage filter")
work = df.dropna(subset=feature_cols + ["bottleneck_index_5d"]).copy()
work["lag1_target"] = work["bottleneck_index_5d"].shift(1)
work = work.dropna().reset_index(drop=True)

# Walk-forward: train 70%, test last 30% chronologically
split = int(len(work) * 0.7)
train, test = work.iloc[:split], work.iloc[split:]
X_train, y_train = train[feature_cols], train["bottleneck_index_5d"]
X_test, y_test = test[feature_cols], test["bottleneck_index_5d"]

# Baseline: lag1
pred_base = test["lag1_target"]
rmse_base = mean_squared_error(y_test, pred_base) ** 0.5
mae_base = mean_absolute_error(y_test, pred_base)

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)
ridge = Ridge(alpha=1.0).fit(Xtr, y_train)
pred_ridge = ridge.predict(Xte)
rf = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3).fit(X_train, y_train)
pred_rf = rf.predict(X_test)

rows = []
for name, pred in [("lag_baseline", pred_base), ("ridge", pred_ridge), ("random_forest", pred_rf)]:
    rows.append({
        "Model": name,
        "Feature Set": ",".join(feature_cols),
        "Validation Approach": "chronological 70/30 walk-forward",
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "MAE": mean_absolute_error(y_test, pred),
        "Key Takeaway": "",
    })
metrics = pd.DataFrame(rows)
best = metrics.sort_values("RMSE").iloc[0]["Model"]
metrics.loc[metrics["Model"]==best, "Key Takeaway"] = "Best RMSE on holdout window"
metrics.to_csv(PATHS["outputs"] / "metrics_rq1.csv", index=False)
metrics.to_csv(PATHS["tables"] / "model_performance_rq1.csv", index=False)
metrics
""",
        ),
        (
            "md",
            """### Figure 4. RQ1 forecast vs actual (best model on holdout)""",
        ),
        (
            "code",
            """
pred_best = {"lag_baseline": pred_base, "ridge": pred_ridge, "random_forest": pred_rf}[best]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(test["date"], y_test.values, label="Actual", color="#1b263b")
ax.plot(test["date"], pred_best, label=f"Predicted ({best})", color="#e76f51", alpha=0.85)
ax.set_title("Figure 4. RQ1 bottleneck forecast vs actual (holdout)")
ax.set_xlabel("Date")
ax.set_ylabel("Bottleneck index")
ax.legend()
fig.tight_layout()
fig.savefig(PATHS["figures"] / "fig04_rq1_forecast.png", dpi=200)
plt.show()
print(metrics)
""",
        ),
    ],
)

write(
    "05_rq2_regime_classification.ipynb",
    [
        ("md", """# 05 RQ2 Regime Classification

Synopsis RQ2: news / GDELT risk features vs no-text baseline for volatility regime.

Optional FinBERT: replace `finbert_risk_probability` after installing torch+transformers."""),
        ("code", RQ_HEADER),
        (
            "code",
            """
text_features = [c for c in ["news_event_impact_score","finbert_risk_probability","gdelt_article_volume"] if c in df.columns]
market_features = [c for c in ["wti_spot","commodity_roll_yield","natgas_hh"] if c in df.columns]
work = df.dropna(subset=market_features + ["volatility_regime"]).copy()
split = int(len(work) * 0.7)
train, test = work.iloc[:split], work.iloc[split:]

def eval_clf(name, feats):
    feats = [f for f in feats if f in work.columns]
    tr = train.dropna(subset=feats)
    te = test.dropna(subset=feats)
    Xtr, ytr = tr[feats], tr["volatility_regime"].astype(int)
    Xte, yte = te[feats], te["volatility_regime"].astype(int)
    if ytr.nunique() < 2 or yte.nunique() < 2:
        return {"Model": name, "Feature Set": ",".join(feats), "Validation Approach": "chronological 70/30", "Accuracy": np.nan, "F1": np.nan, "AUC": np.nan, "Key Takeaway": "Need both classes in split"}
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    pred = clf.predict(Xte)
    return {
        "Model": name,
        "Feature Set": ",".join(feats),
        "Validation Approach": "chronological 70/30",
        "Accuracy": accuracy_score(yte, pred),
        "F1": f1_score(yte, pred, zero_division=0),
        "AUC": roc_auc_score(yte, prob) if yte.nunique() > 1 else np.nan,
        "Key Takeaway": "",
        "_yte": yte, "_prob": prob, "_pred": pred, "_clf": clf,
    }

base = eval_clf("no_text_market_baseline", market_features)
full = eval_clf("market_plus_news", market_features + text_features)
metrics = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in [base, full]])
if metrics["F1"].notna().any():
    winner = metrics.sort_values("F1", ascending=False).iloc[0]["Model"]
    metrics.loc[metrics["Model"]==winner, "Key Takeaway"] = "Higher F1 on holdout"
metrics.to_csv(PATHS["outputs"] / "metrics_rq2.csv", index=False)
metrics.to_csv(PATHS["tables"] / "model_performance_rq2.csv", index=False)
metrics
""",
        ),
        (
            "md",
            """### Figure 5. RQ2 ROC and confusion (news-enhanced model when available)""",
        ),
        (
            "code",
            """
chosen = full if full.get("_pred") is not None else base
if chosen.get("_pred") is not None:
    yte, pred, prob = chosen["_yte"], chosen["_pred"], chosen["_prob"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cm = confusion_matrix(yte, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Figure 5a. Confusion matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    RocCurveDisplay.from_predictions(yte, prob, ax=axes[1])
    axes[1].set_title("Figure 5b. ROC curve")
    fig.tight_layout()
    fig.savefig(PATHS["figures"] / "fig05_rq2_roc_cm.png", dpi=200)
    plt.show()
else:
    print("Insufficient class balance for ROC/CM in this split")
metrics
""",
        ),
    ],
)

write(
    "06_rq3_fusion_omega.ipynb",
    [
        ("md", """# 06 RQ3 Multi-Modal Fusion and Omega

Synopsis RQ3: compare spatial-only, news-only, and fused feature sets on stress-window Omega proxy association / ranking."""),
        ("code", RQ_HEADER),
        (
            "code",
            """
from sklearn.linear_model import LinearRegression

spatial = [c for c in ["port_congestion_hours","chokepoint_vessel_count","vessel_draught_anomaly"] if c in df.columns]
news = [c for c in ["news_event_impact_score","finbert_risk_probability"] if c in df.columns]
work = df.dropna(subset=["omega_ratio_proxy"]).copy()
# Stress window: top quartile of WTI realized vol
vol = work["wti_spot"].pct_change().rolling(10).std()
work = work.assign(stress=(vol >= vol.quantile(0.75)))
stress = work[work["stress"]].dropna(subset=spatial + news + ["omega_ratio_proxy"])

split = int(len(work) * 0.7)
results = []
preds_hold = {}
for name, feats in [
    ("spatial_only", spatial),
    ("news_only", news),
    ("early_fusion", spatial + news),
]:
    feats = [f for f in feats if f in work.columns]
    if len(feats) < 1:
        continue
    d = work.dropna(subset=feats + ["omega_ratio_proxy"]).copy()
    tr, te = d.iloc[:int(len(d)*0.7)], d.iloc[int(len(d)*0.7):]
    model = LinearRegression().fit(tr[feats], tr["omega_ratio_proxy"])
    pred = model.predict(te[feats])
    rmse = mean_squared_error(te["omega_ratio_proxy"], pred) ** 0.5
    # Omega-style score: mean predicted omega in stress days within test window
    te_stress = te.copy()
    te_stress["pred"] = pred
    vol_te = te_stress["wti_spot"].pct_change().rolling(10).std()
    thr = vol_te.quantile(0.75)
    omega_stress = te_stress.loc[vol_te >= thr, "pred"].mean()
    results.append({
        "Model": name,
        "Feature Set": ",".join(feats),
        "Validation Approach": "chronological 70/30",
        "RMSE_omega": rmse,
        "MeanPredOmega_stress": omega_stress,
        "Key Takeaway": "",
    })
    preds_hold[name] = (te["date"], te["omega_ratio_proxy"], pred)

metrics = pd.DataFrame(results)
if len(metrics):
    best = metrics.sort_values("RMSE_omega").iloc[0]["Model"]
    metrics.loc[metrics["Model"]==best, "Key Takeaway"] = "Lowest Omega RMSE; check stress mean too"
metrics.to_csv(PATHS["outputs"] / "metrics_rq3.csv", index=False)
metrics.to_csv(PATHS["tables"] / "model_performance_rq3.csv", index=False)
metrics
""",
        ),
        (
            "md",
            """### Figure 6. RQ3 modality comparison""",
        ),
        (
            "code",
            """
fig, ax = plt.subplots(figsize=(7.5, 4.5))
plot_df = metrics.set_index("Model")[["RMSE_omega"]]
plot_df.plot(kind="bar", ax=ax, legend=False, color="#2a9d8f")
ax.set_title("Figure 6. RQ3 Omega RMSE by modality")
ax.set_ylabel("RMSE")
ax.set_xlabel("Model")
fig.tight_layout()
fig.savefig(PATHS["figures"] / "fig06_rq3_modality.png", dpi=200)
plt.show()
""",
        ),
    ],
)

write(
    "07_rq4_contagion_vecm.ipynb",
    [
        ("md", """# 07 RQ4 Cross-Commodity Contagion

Synopsis RQ4: energy transport / energy price stress leading agricultural spreads (Granger / VAR; VECM if cointegrated)."""),
        ("code", RQ_HEADER),
        (
            "code",
            """
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR

cols = [c for c in ["wti_spot", "wheat", "corn", "grain_calendar_spread", "port_congestion_hours"] if c in df.columns]
work = df[["date"] + cols].dropna().copy()
# Use changes for Granger robustness
work["d_wti"] = work["wti_spot"].pct_change()
work["d_grain"] = work["grain_calendar_spread"].pct_change() if "grain_calendar_spread" in work.columns else work["wheat"].pct_change()
work["d_cong"] = work["port_congestion_hours"].pct_change() if "port_congestion_hours" in work.columns else work["d_wti"]
g = work[["d_cong", "d_grain", "d_wti"]].replace([np.inf, -np.inf], np.nan).dropna()

# ADF
adf_rows = []
for c in g.columns:
    stat, p, *_ = adfuller(g[c].astype(float).values, maxlag=5, autolag=None)
    adf_rows.append({"series": c, "adf_stat": stat, "p_value": p, "stationary_5pct": p < 0.05})
adf_tbl = pd.DataFrame(adf_rows)
adf_tbl.to_csv(PATHS["tables"] / "rq4_adf.csv", index=False)
adf_tbl
""",
        ),
        (
            "code",
            """
# Granger: does congestion/energy lead grain?
maxlag = 7
try:
    gc = grangercausalitytests(g[["d_grain", "d_cong"]], maxlag=maxlag, verbose=False)
except TypeError:
    gc = grangercausalitytests(g[["d_grain", "d_cong"]], maxlag=maxlag)
gc_rows = []
for lag, res in gc.items():
    ftest = res[0]["ssr_ftest"]
    gc_rows.append({"direction": "cong->grain", "lag": lag, "F": ftest[0], "p_value": ftest[1]})
try:
    gc2 = grangercausalitytests(g[["d_grain", "d_wti"]], maxlag=maxlag, verbose=False)
except TypeError:
    gc2 = grangercausalitytests(g[["d_grain", "d_wti"]], maxlag=maxlag)
for lag, res in gc2.items():
    ftest = res[0]["ssr_ftest"]
    gc_rows.append({"direction": "wti->grain", "lag": lag, "F": ftest[0], "p_value": ftest[1]})
gc_tbl = pd.DataFrame(gc_rows)
gc_tbl.to_csv(PATHS["tables"] / "rq4_granger.csv", index=False)

# VAR companion forecast (sMAPE on grain changes)
var_data = g[["d_wti", "d_grain"]].dropna().reset_index(drop=True)
split = int(len(var_data) * 0.8)
model = VAR(var_data.iloc[:split]).fit(maxlags=5, ic="aic")
lag_order = max(int(model.k_ar), 1)
if model.k_ar < 1:
    model = VAR(var_data.iloc[:split]).fit(1)
    lag_order = 1
steps = min(30, len(var_data) - split)
forecast = model.forecast(var_data.iloc[split-lag_order:split].values, steps=steps)
actual = var_data.iloc[split:split+len(forecast)]["d_grain"].values
pred = forecast[:, 1]
smape = float(np.mean(2.0 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-8)) * 100)
metrics = pd.DataFrame([
    {"Model": "Granger+VAR", "Feature Set": "d_wti,d_cong,d_grain", "Validation Approach": "chronological VAR holdout", "Metric 1": gc_tbl.loc[gc_tbl.direction=="wti->grain","p_value"].min(), "Metric 2": smape, "Key Takeaway": "Metric1=min Granger p (wti->grain); Metric2=sMAPE %"}
])
metrics.to_csv(PATHS["outputs"] / "metrics_rq4.csv", index=False)
metrics.to_csv(PATHS["tables"] / "model_performance_rq4.csv", index=False)
print("VAR lag order", lag_order)
display(gc_tbl.sort_values("p_value").head(10))
metrics
""",
        ),
        (
            "md",
            """### Figure 7. RQ4 lead-lag style view (energy vs grain changes)""",
        ),
        (
            "code",
            """
fig, ax = plt.subplots(figsize=(9, 4.5))
plot = work.dropna(subset=["d_wti", "d_grain"]).iloc[-400:]
ax.plot(plot["date"], plot["d_wti"].rolling(5).mean(), label="WTI % change (5d MA)", color="#1b263b")
ax.plot(plot["date"], plot["d_grain"].rolling(5).mean(), label="Grain spread % change (5d MA)", color="#e9c46a")
ax.set_title("Figure 7. Energy vs grain changes (lead-lag exploration)")
ax.set_xlabel("Date")
ax.set_ylabel("Change")
ax.legend()
fig.tight_layout()
fig.savefig(PATHS["figures"] / "fig07_rq4_leadlag.png", dpi=200)
plt.show()
""",
        ),
        (
            "md",
            """## Next steps after RQ notebooks

1. Review `notebooks/outputs/latest/metrics_rq*.csv` and figures.
2. Append `shared/CHANGELOG.md`.
3. Push `data/` + `notebooks/outputs/` to GitHub (no `.env`).
4. Use outputs to draft the Week4 interim PDF.""",
        ),
    ],
)

# feature table helper notebook note already covered
print("done")
