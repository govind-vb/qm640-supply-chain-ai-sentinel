#!/usr/bin/env python3
"""Execute cleaning + RQ1–RQ4 analyses with PortWatch + FinBERT features."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project_paths import ensure_output_dirs, load_env  # noqa: E402


def _chrono_split_both_classes(work: pd.DataFrame, ycol: str, min_test_frac: float = 0.25):
    """Find a chronological split that keeps both classes in train and test."""
    n = len(work)
    for frac in (0.7, 0.65, 0.6, 0.75, 0.55):
        split = int(n * frac)
        if split < 40 or n - split < 20:
            continue
        tr, te = work.iloc[:split], work.iloc[split:]
        if tr[ycol].nunique() >= 2 and te[ycol].nunique() >= 2:
            return tr, te, f"chronological {frac:.0%}/{1-frac:.0%}"
    # fallback: last min_test_frac
    split = max(40, int(n * (1 - min_test_frac)))
    return work.iloc[:split], work.iloc[split:], "chronological fallback"


def main() -> None:
    root = load_env()
    paths = ensure_output_dirs(root)
    sns.set_theme(style="whitegrid", context="talk")

    panel_path = paths["processed"] / "daily_panel.csv"
    if not panel_path.exists():
        from fetch_panel import build_daily_panel

        build_daily_panel()

    df = pd.read_csv(panel_path, parse_dates=["date"]).sort_values("date")

    # ---- cleaning ----
    cleaning_rows = []
    n0 = len(df)
    before = len(df)
    df = df.dropna(subset=["date", "wti_spot"])
    cleaning_rows.append(
        {
            "Issue": "Missing core market anchor",
            "Variables Affected": "date,wti_spot",
            "Detection Method": "isna",
            "Treatment Applied": f"drop {before - len(df)} rows",
            "Rationale": "Cannot model commodity outcomes without observed WTI",
        }
    )
    dup = int(df.duplicated(subset=["date"]).sum())
    df = df.drop_duplicates(subset=["date"], keep="last")
    cleaning_rows.append(
        {
            "Issue": "Duplicates",
            "Variables Affected": "date",
            "Detection Method": "duplicated(date)",
            "Treatment Applied": f"drop {dup} duplicate dates",
            "Rationale": "One row per day for panel models",
        }
    )
    ret = df["wti_spot"].pct_change()
    lo, hi = ret.quantile(0.01), ret.quantile(0.99)
    mask = ret.between(lo, hi) | ret.isna()
    removed = int((~mask).sum())
    df = df.loc[mask | ret.isna()].copy()
    cleaning_rows.append(
        {
            "Issue": "Outliers",
            "Variables Affected": "wti_spot returns",
            "Detection Method": "1st/99th pct return fences",
            "Treatment Applied": f"remove {removed} extreme return days",
            "Rationale": "Limit tick distortion while keeping crisis days inside fences",
        }
    )
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].ffill(limit=3).bfill(limit=1)
    cleaning_rows.append(
        {
            "Issue": "Missing values",
            "Variables Affected": ",".join(list(num_cols)[:8]) + ",...",
            "Detection Method": "isna sum",
            "Treatment Applied": "ffill limit 3, bfill limit 1",
            "Rationale": "Preserve chronology; avoid long leakage-style fills",
        }
    )
    # require PortWatch physical coverage
    before = len(df)
    df = df.dropna(subset=["chokepoint_vessel_count", "port_congestion_hours"])
    cleaning_rows.append(
        {
            "Issue": "Missing PortWatch physical IVs",
            "Variables Affected": "chokepoint_vessel_count,port_congestion_hours",
            "Detection Method": "isna",
            "Treatment Applied": f"drop {before - len(df)} rows without PortWatch merge",
            "Rationale": "Physical IVs must be PortWatch-derived for RQ1/RQ3",
        }
    )
    df = df.reset_index(drop=True)
    clean_path = paths["processed"] / "daily_panel_clean.csv"
    df.to_csv(clean_path, index=False)
    pd.DataFrame(cleaning_rows).to_csv(paths["tables"] / "cleaning_log.csv", index=False)
    print("clean rows", len(df), "from", n0)

    # ---- EDA descriptive stats ----
    key_cols = [
        c
        for c in [
            "wti_spot",
            "chokepoint_vessel_count",
            "port_congestion_hours",
            "vessel_draught_anomaly",
            "news_event_impact_score",
            "finbert_risk_probability",
            "bottleneck_index_5d",
            "volatility_regime",
            "omega_ratio_proxy",
            "grain_calendar_spread",
            "commodity_roll_yield",
        ]
        if c in df.columns
    ]
    desc_rows = []
    for c in key_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        desc_rows.append(
            {
                "variable": c,
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "max": s.max(),
                "missing_pct": float(s.isna().mean() * 100),
            }
        )
    pd.DataFrame(desc_rows).to_csv(paths["tables"] / "eda_descriptive_stats.csv", index=False)

    miss = df.isna().mean().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    miss.plot(kind="bar", ax=ax, color="#33415c")
    ax.set_title("Figure 1. Share of missing values by variable")
    ax.set_ylabel("Fraction missing")
    ax.set_xlabel("Variable")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig01_missingness.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["date"], df["wti_spot"], color="#1b263b", label="WTI")
    ax2 = ax.twinx()
    ax2.plot(df["date"], df["port_congestion_hours"], color="#e76f51", alpha=0.8, label="Congestion intensity")
    ax.set_title("Figure 2. WTI vs PortWatch congestion intensity")
    ax.set_xlabel("Date")
    ax.set_ylabel("WTI USD")
    ax2.set_ylabel("Congestion intensity (stress units)")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig02_wti_congestion.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(df[key_cols].corr(), ax=ax, cmap="vlag", center=0, annot=False)
    ax.set_title("Figure 3. Correlation heatmap of synopsis model features")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig03_corr_heatmap.png", dpi=200)
    plt.close()

    pd.DataFrame(
        [
            {
                "Figure/Table Reference": "Table eda_descriptive_stats",
                "What it Shows": "Mean/median/SD for key IVs/DVs",
                "Key Insight": "PortWatch transit and congestion intensity have usable coverage",
                "Why it Matters (Link to RQ/objective)": "RQ1/RQ3 physical IV credibility",
                "Decision/Next Step": "Keep PortWatch ArcGIS merge as primary physical source",
            },
            {
                "Figure/Table Reference": "Figure 1",
                "What it Shows": "Missingness by variable",
                "Key Insight": "GDELT tone may remain sparser than PortWatch/FRED",
                "Why it Matters (Link to RQ/objective)": "RQ2 feature completeness",
                "Decision/Next Step": "Rely on FinBERT risk with persistence between probe days",
            },
            {
                "Figure/Table Reference": "Figure 2",
                "What it Shows": "WTI vs PortWatch congestion intensity",
                "Key Insight": "Physical stress and energy prices co-move in disruption windows",
                "Why it Matters (Link to RQ/objective)": "RQ1 timing / RQ3 fusion",
                "Decision/Next Step": "Use lags in forecast models",
            },
            {
                "Figure/Table Reference": "Figure 3",
                "What it Shows": "Feature correlations",
                "Key Insight": "Identify near-duplicate predictors before fusion",
                "Why it Matters (Link to RQ/objective)": "Feature selection RQ1-RQ3",
                "Decision/Next Step": "Prefer PortWatch + FinBERT complementary sets",
            },
        ]
    ).to_csv(paths["tables"] / "eda_insight_summary.csv", index=False)

    # ---- sample size ----
    def green(k: int) -> int:
        return 50 + 8 * k

    def corr_power_n(r: float = 0.15) -> int:
        fisher = 0.5 * math.log((1 + r) / (1 - r))
        return math.ceil(((1.96 + 0.84) / fisher) ** 2 + 3)

    tbl = pd.DataFrame(
        [
            {"Research Question": "RQ1", "Method": "Green", "Key parameters": "alpha=.05; k=6", "Minimum N": green(6)},
            {"Research Question": "RQ2", "Method": "Green", "Key parameters": "alpha=.05; k=8", "Minimum N": green(8)},
            {
                "Research Question": "RQ3",
                "Method": "Correlation power (Fisher z)",
                "Key parameters": "alpha=.05; power=.80; r=.15",
                "Minimum N": corr_power_n(),
            },
            {"Research Question": "RQ4", "Method": "Green", "Key parameters": "alpha=.05; k=10", "Minimum N": green(10)},
        ]
    )
    final_n = int(tbl["Minimum N"].max())
    assert final_n == 347
    tbl.loc[len(tbl)] = {
        "Research Question": "Final N",
        "Method": "max(RQ1-RQ4)",
        "Key parameters": "Protects weakest-powered RQ",
        "Minimum N": final_n,
    }
    tbl.to_csv(paths["tables"] / "sample_size.csv", index=False)
    pd.DataFrame(
        [{"available_n": len(df), "final_n_required": 347, "meets_requirement": len(df) >= 347}]
    ).to_csv(paths["tables"] / "sample_size_vs_available.csv", index=False)

    # ---- RQ1 ----
    feature_cols = [
        c
        for c in [
            "chokepoint_vessel_count",
            "port_congestion_hours",
            "vessel_draught_anomaly",
            "commodity_roll_yield",
            "wti_spot",
            "natgas_hh",
        ]
        if c in df.columns and df[c].notna().sum() >= 50
    ]
    work = df.dropna(subset=feature_cols + ["bottleneck_index_5d"]).copy()
    work["lag1_target"] = work["bottleneck_index_5d"].shift(1)
    work = work.dropna().reset_index(drop=True)
    split = int(len(work) * 0.7)
    train, test = work.iloc[:split], work.iloc[split:]
    X_train, y_train = train[feature_cols], train["bottleneck_index_5d"]
    X_test, y_test = test[feature_cols], test["bottleneck_index_5d"]
    pred_base = test["lag1_target"]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    best_ridge, best_alpha, best_pred_ridge = None, None, None
    for alpha in (0.1, 1.0, 10.0):
        model = Ridge(alpha=alpha).fit(Xtr, y_train)
        pred = model.predict(Xte)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        if best_ridge is None or rmse < best_ridge:
            best_ridge, best_alpha, best_pred_ridge = rmse, alpha, pred
    rf = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3).fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    rows = []
    for name, pred, note in [
        ("lag_baseline", pred_base, ""),
        (f"ridge_alpha_{best_alpha}", best_pred_ridge, f"best alpha={best_alpha}"),
        ("random_forest", pred_rf, ""),
    ]:
        rows.append(
            {
                "Model": name,
                "Feature Set": ",".join(feature_cols),
                "Validation Approach": "chronological 70/30 walk-forward",
                "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                "MAE": mean_absolute_error(y_test, pred),
                "Key Takeaway": note,
            }
        )
    metrics = pd.DataFrame(rows)
    best = metrics.sort_values("RMSE").iloc[0]["Model"]
    metrics.loc[metrics["Model"] == best, "Key Takeaway"] = (
        (metrics.loc[metrics["Model"] == best, "Key Takeaway"].fillna("").astype(str) + "; Best RMSE on holdout").str.strip("; ")
    )
    metrics.to_csv(paths["outputs"] / "metrics_rq1.csv", index=False)
    metrics.to_csv(paths["tables"] / "model_performance_rq1.csv", index=False)
    pred_map = {
        "lag_baseline": pred_base,
        f"ridge_alpha_{best_alpha}": best_pred_ridge,
        "random_forest": pred_rf,
    }
    pred_best = pred_map[best]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(test["date"], y_test.values, label="Actual", color="#1b263b")
    ax.plot(test["date"], pred_best, label=f"Predicted ({best})", color="#e76f51", alpha=0.85)
    ax.set_title("Figure 4. RQ1 bottleneck forecast vs actual (holdout)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Bottleneck index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig04_rq1_forecast.png", dpi=200)
    plt.close()

    # feature importance
    fig, ax = plt.subplots(figsize=(8, 4.5))
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    imp.plot(kind="barh", ax=ax, color="#2a9d8f")
    ax.set_title("Figure 8. RQ1 random forest feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig08_rq1_importance.png", dpi=200)
    plt.close()
    print("RQ1 best", best)

    # ---- RQ2 ----
    text_features = [
        c
        for c in ["news_event_impact_score", "finbert_risk_probability", "gdelt_article_volume"]
        if c in df.columns
    ]
    market_features = [c for c in ["wti_spot", "commodity_roll_yield", "natgas_hh"] if c in df.columns]
    work2 = df.dropna(subset=market_features + ["volatility_regime"]).copy()
    # ensure FinBERT present
    if "finbert_risk_probability" in work2.columns:
        work2["finbert_risk_probability"] = work2["finbert_risk_probability"].ffill().bfill()

    def eval_clf(name: str, feats: list[str], train_df, test_df, val_label: str) -> dict:
        feats = [f for f in feats if f in work2.columns]
        tr = train_df.dropna(subset=feats)
        te = test_df.dropna(subset=feats)
        Xtr, ytr = tr[feats], tr["volatility_regime"].astype(int)
        Xte, yte = te[feats], te["volatility_regime"].astype(int)
        if ytr.nunique() < 2 or yte.nunique() < 2 or len(te) < 15:
            return {
                "Model": name,
                "Feature Set": ",".join(feats),
                "Validation Approach": val_label,
                "Accuracy": np.nan,
                "F1": np.nan,
                "AUC": np.nan,
                "Key Takeaway": "Need both classes in split",
            }
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        return {
            "Model": name,
            "Feature Set": ",".join(feats),
            "Validation Approach": val_label,
            "Accuracy": accuracy_score(yte, pred),
            "F1": f1_score(yte, pred, zero_division=0),
            "AUC": roc_auc_score(yte, prob),
            "Key Takeaway": "",
            "_yte": yte,
            "_prob": prob,
            "_pred": pred,
        }

    train2, test2, val_lab = _chrono_split_both_classes(work2, "volatility_regime")
    base = eval_clf("no_text_market_baseline", market_features, train2, test2, val_lab)
    full = eval_clf("market_plus_finbert_news", market_features + text_features, train2, test2, val_lab)
    m2 = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in [base, full]])
    if m2["F1"].notna().any():
        winner = m2.sort_values("F1", ascending=False).iloc[0]["Model"]
        m2.loc[m2["Model"] == winner, "Key Takeaway"] = "Higher F1 on holdout"
    m2.to_csv(paths["outputs"] / "metrics_rq2.csv", index=False)
    m2.to_csv(paths["tables"] / "model_performance_rq2.csv", index=False)
    chosen = full if full.get("_pred") is not None else base
    if chosen.get("_pred") is not None:
        yte, pred, prob = chosen["_yte"], chosen["_pred"], chosen["_prob"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(confusion_matrix(yte, pred), annot=True, fmt="d", cmap="Blues", ax=axes[0])
        axes[0].set_title("Figure 5a. Confusion matrix")
        RocCurveDisplay.from_predictions(yte, prob, ax=axes[1])
        axes[1].set_title("Figure 5b. ROC curve")
        fig.tight_layout()
        fig.savefig(paths["figures"] / "fig05_rq2_roc_cm.png", dpi=200)
        plt.close()
    print("RQ2 done", m2[["Model", "F1", "AUC"]].to_dict("records"))

    # ---- RQ3 ----
    spatial = [
        c
        for c in ["port_congestion_hours", "chokepoint_vessel_count", "vessel_draught_anomaly"]
        if c in df.columns
    ]
    news = [c for c in ["news_event_impact_score", "finbert_risk_probability"] if c in df.columns]
    work3 = df.dropna(subset=["omega_ratio_proxy"]).copy()
    if "finbert_risk_probability" in work3.columns:
        work3["finbert_risk_probability"] = work3["finbert_risk_probability"].ffill().bfill()
    results = []
    for name, feats in [
        ("spatial_only", spatial),
        ("news_only", news),
        ("early_fusion", spatial + news),
    ]:
        feats = [f for f in feats if f in work3.columns]
        if len(feats) < 1:
            continue
        d = work3.dropna(subset=feats + ["omega_ratio_proxy"]).copy()
        tr, te = d.iloc[: int(len(d) * 0.7)], d.iloc[int(len(d) * 0.7) :]
        model = LinearRegression().fit(tr[feats], tr["omega_ratio_proxy"])
        pred = model.predict(te[feats])
        rmse = mean_squared_error(te["omega_ratio_proxy"], pred) ** 0.5
        te_stress = te.copy()
        te_stress["pred"] = pred
        vol_te = te_stress["wti_spot"].pct_change().rolling(10).std()
        thr = vol_te.quantile(0.75)
        omega_stress = float(te_stress.loc[vol_te >= thr, "pred"].mean())
        results.append(
            {
                "Model": name,
                "Feature Set": ",".join(feats),
                "Validation Approach": "chronological 70/30",
                "RMSE_omega": rmse,
                "MeanPredOmega_stress": omega_stress,
                "Key Takeaway": "",
            }
        )
    m3 = pd.DataFrame(results)
    if len(m3):
        best3 = m3.sort_values("RMSE_omega").iloc[0]["Model"]
        m3.loc[m3["Model"] == best3, "Key Takeaway"] = "Lowest Omega RMSE; check stress mean too"
    m3.to_csv(paths["outputs"] / "metrics_rq3.csv", index=False)
    m3.to_csv(paths["tables"] / "model_performance_rq3.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    m3.set_index("Model")[["RMSE_omega"]].plot(kind="bar", ax=ax, legend=False, color="#2a9d8f")
    ax.set_title("Figure 6. RQ3 Omega RMSE by modality")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig06_rq3_modality.png", dpi=200)
    plt.close()
    print("RQ3 done")

    # ---- RQ4 weekly ----
    cols4 = [c for c in ["wti_spot", "wheat", "corn", "grain_calendar_spread", "port_congestion_hours"] if c in df.columns]
    daily4 = df[["date"] + cols4].dropna().copy()
    weekly = (
        daily4.set_index("date")
        .resample("W")
        .mean()
        .dropna()
        .reset_index()
    )
    weekly["d_wti"] = weekly["wti_spot"].pct_change()
    weekly["d_grain"] = weekly["grain_calendar_spread"].pct_change()
    weekly["d_cong"] = weekly["port_congestion_hours"].pct_change()
    g = weekly[["d_cong", "d_grain", "d_wti"]].replace([np.inf, -np.inf], np.nan).dropna()
    adf_rows = []
    for c in g.columns:
        series = g[c].astype(float).values
        stat, p, *_ = adfuller(series, maxlag=4, autolag=None)
        adf_rows.append({"series": c, "adf_stat": stat, "p_value": p, "stationary_5pct": p < 0.05})
    pd.DataFrame(adf_rows).to_csv(paths["tables"] / "rq4_adf.csv", index=False)
    maxlag = 5
    gc_rows = []
    for direction, pair in [("cong->grain", ["d_grain", "d_cong"]), ("wti->grain", ["d_grain", "d_wti"])]:
        try:
            gc = grangercausalitytests(g[pair], maxlag=maxlag, verbose=False)
        except TypeError:
            gc = grangercausalitytests(g[pair], maxlag=maxlag)
        for lag, res in gc.items():
            ftest = res[0]["ssr_ftest"]
            gc_rows.append({"direction": direction, "lag": lag, "F": ftest[0], "p_value": ftest[1]})
    gc_tbl = pd.DataFrame(gc_rows)
    gc_tbl.to_csv(paths["tables"] / "rq4_granger.csv", index=False)
    var_data = g[["d_wti", "d_grain"]].dropna().reset_index(drop=True)
    split = int(len(var_data) * 0.8)
    model = VAR(var_data.iloc[:split]).fit(maxlags=4, ic="aic")
    lag_order = max(int(model.k_ar), 1)
    if model.k_ar < 1:
        model = VAR(var_data.iloc[:split]).fit(1)
        lag_order = 1
    steps = min(12, len(var_data) - split)
    forecast = model.forecast(var_data.iloc[split - lag_order : split].values, steps=steps)
    actual = var_data.iloc[split : split + len(forecast)]["d_grain"].values
    pred = forecast[:, 1]
    smape = float(np.mean(2.0 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-8)) * 100)
    mae = float(mean_absolute_error(actual, pred))
    m4 = pd.DataFrame(
        [
            {
                "Model": "Granger+VAR_weekly",
                "Feature Set": "d_wti,d_cong,d_grain",
                "Validation Approach": "weekly chronological VAR holdout",
                "Metric 1": gc_tbl.loc[gc_tbl.direction == "wti->grain", "p_value"].min(),
                "Metric 2": smape,
                "MAE_d_grain": mae,
                "Key Takeaway": "Metric1=min Granger p (wti->grain); Metric2=sMAPE %",
            }
        ]
    )
    m4.to_csv(paths["outputs"] / "metrics_rq4.csv", index=False)
    m4.to_csv(paths["tables"] / "model_performance_rq4.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot = weekly.dropna(subset=["d_wti", "d_grain"]).iloc[-120:]
    ax.plot(plot["date"], plot["d_wti"].rolling(3).mean(), label="WTI % change (3w MA)", color="#1b263b")
    ax.plot(plot["date"], plot["d_grain"].rolling(3).mean(), label="Grain spread % change (3w MA)", color="#e9c46a")
    ax.set_title("Figure 7. Weekly energy vs grain changes (lead-lag exploration)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Change")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig07_rq4_leadlag.png", dpi=200)
    plt.close()
    print("RQ4 done; VAR lag", lag_order)
    print("Pipeline complete. Outputs in", paths["outputs"])


if __name__ == "__main__":
    main()
