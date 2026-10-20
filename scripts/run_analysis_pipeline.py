#!/usr/bin/env python3
"""Execute cleaning + RQ1–RQ4 analyses (same outputs as notebooks 02–07)."""

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


def main() -> None:
    root = load_env()
    paths = ensure_output_dirs(root)
    sns.set_theme(style="whitegrid", context="talk")

    # ---- collect if needed ----
    panel_path = paths["processed"] / "daily_panel.csv"
    if not panel_path.exists():
        from fetch_panel import build_daily_panel

        build_daily_panel()

    df = pd.read_csv(panel_path, parse_dates=["date"]).sort_values("date")

    # ---- cleaning (notebook 02) ----
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
    dup = df.duplicated(subset=["date"]).sum()
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
    removed = (~mask).sum()
    df = df.loc[mask | ret.isna()].copy()
    cleaning_rows.append(
        {
            "Issue": "Outliers",
            "Variables Affected": "wti_spot returns",
            "Detection Method": "1st/99th pct return fences",
            "Treatment Applied": f"remove {removed} extreme return days",
            "Rationale": "Limit distortion from bad ticks while keeping real crisis days inside fences",
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
    df = df.reset_index(drop=True)
    clean_path = paths["processed"] / "daily_panel_clean.csv"
    df.to_csv(clean_path, index=False)
    pd.DataFrame(cleaning_rows).to_csv(paths["tables"] / "cleaning_log.csv", index=False)
    print("clean rows", len(df), "from", n0)

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
    ax.plot(df["date"], df["wti_spot"], label="WTI spot", color="#1b263b")
    ax2 = ax.twinx()
    ax2.plot(df["date"], df["port_congestion_hours"], color="#e76f51", alpha=0.8)
    ax.set_title("Figure 2. WTI vs congestion/stress indicator")
    ax.set_xlabel("Date")
    ax.set_ylabel("WTI USD")
    ax2.set_ylabel("Congestion / stress")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig02_wti_congestion.png", dpi=200)
    plt.close()

    cols = [
        c
        for c in [
            "wti_spot",
            "port_congestion_hours",
            "chokepoint_vessel_count",
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
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(df[cols].corr(), ax=ax, cmap="vlag", center=0, annot=False)
    ax.set_title("Figure 3. Correlation heatmap of synopsis model features")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig03_corr_heatmap.png", dpi=200)
    plt.close()

    pd.DataFrame(
        [
            {
                "Figure/Table Reference": "Figure 1",
                "What it Shows": "Missingness by variable",
                "Key Insight": "Identify incomplete PortWatch vs market coverage",
                "Why it Matters (Link to RQ/objective)": "RQ1 physical IV completeness",
                "Decision/Next Step": "Add PortWatch CSV if vessel fields sparse",
            },
            {
                "Figure/Table Reference": "Figure 2",
                "What it Shows": "WTI vs congestion/stress",
                "Key Insight": "Co-movement during stress windows",
                "Why it Matters (Link to RQ/objective)": "RQ1/RQ3 timing",
                "Decision/Next Step": "Use lags in forecast models",
            },
            {
                "Figure/Table Reference": "Figure 3",
                "What it Shows": "Feature correlations",
                "Key Insight": "Which IVs move with targets",
                "Why it Matters (Link to RQ/objective)": "Feature selection RQ1-RQ3",
                "Decision/Next Step": "Drop near-duplicate predictors",
            },
        ]
    ).to_csv(paths["tables"] / "eda_insight_summary.csv", index=False)

    # ---- sample size (03) ----
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
        [
            {
                "available_n": len(df),
                "final_n_required": 347,
                "meets_requirement": len(df) >= 347,
            }
        ]
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
    if not feature_cols:
        raise RuntimeError("No usable RQ1 features after coverage filter")
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
    pred_ridge = Ridge(alpha=1.0).fit(Xtr, y_train).predict(Xte)
    pred_rf = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3).fit(
        X_train, y_train
    ).predict(X_test)
    rows = []
    for name, pred in [("lag_baseline", pred_base), ("ridge", pred_ridge), ("random_forest", pred_rf)]:
        rows.append(
            {
                "Model": name,
                "Feature Set": ",".join(feature_cols),
                "Validation Approach": "chronological 70/30 walk-forward",
                "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                "MAE": mean_absolute_error(y_test, pred),
                "Key Takeaway": "",
            }
        )
    metrics = pd.DataFrame(rows)
    best = metrics.sort_values("RMSE").iloc[0]["Model"]
    metrics.loc[metrics["Model"] == best, "Key Takeaway"] = "Best RMSE on holdout window"
    metrics.to_csv(paths["outputs"] / "metrics_rq1.csv", index=False)
    metrics.to_csv(paths["tables"] / "model_performance_rq1.csv", index=False)
    pred_best = {"lag_baseline": pred_base, "ridge": pred_ridge, "random_forest": pred_rf}[best]
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
    print("RQ1 best", best)

    # ---- RQ2 ----
    text_features = [
        c for c in ["news_event_impact_score", "finbert_risk_probability", "gdelt_article_volume"] if c in df.columns
    ]
    market_features = [c for c in ["wti_spot", "commodity_roll_yield", "natgas_hh"] if c in df.columns]
    work2 = df.dropna(subset=market_features + ["volatility_regime"]).copy()
    split = int(len(work2) * 0.7)
    train, test = work2.iloc[:split], work2.iloc[split:]

    def eval_clf(name: str, feats: list[str]) -> dict:
        feats = [f for f in feats if f in work2.columns]
        tr = train.dropna(subset=feats)
        te = test.dropna(subset=feats)
        Xtr, ytr = tr[feats], tr["volatility_regime"].astype(int)
        Xte, yte = te[feats], te["volatility_regime"].astype(int)
        if ytr.nunique() < 2 or yte.nunique() < 2:
            return {
                "Model": name,
                "Feature Set": ",".join(feats),
                "Validation Approach": "chronological 70/30",
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
            "Validation Approach": "chronological 70/30",
            "Accuracy": accuracy_score(yte, pred),
            "F1": f1_score(yte, pred, zero_division=0),
            "AUC": roc_auc_score(yte, prob),
            "Key Takeaway": "",
            "_yte": yte,
            "_prob": prob,
            "_pred": pred,
        }

    base = eval_clf("no_text_market_baseline", market_features)
    full = eval_clf("market_plus_news", market_features + text_features)
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
    print("RQ2 done")

    # ---- RQ3 ----
    spatial = [c for c in ["port_congestion_hours", "chokepoint_vessel_count", "vessel_draught_anomaly"] if c in df.columns]
    news = [c for c in ["news_event_impact_score", "finbert_risk_probability"] if c in df.columns]
    work3 = df.dropna(subset=["omega_ratio_proxy"]).copy()
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
        omega_stress = te_stress.loc[vol_te >= thr, "pred"].mean()
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

    # ---- RQ4 ----
    cols4 = [c for c in ["wti_spot", "wheat", "corn", "grain_calendar_spread", "port_congestion_hours"] if c in df.columns]
    work4 = df[["date"] + cols4].dropna().copy()
    work4["d_wti"] = work4["wti_spot"].pct_change()
    work4["d_grain"] = work4["grain_calendar_spread"].pct_change()
    work4["d_cong"] = work4["port_congestion_hours"].pct_change()
    g = work4[["d_cong", "d_grain", "d_wti"]].replace([np.inf, -np.inf], np.nan).dropna()
    adf_rows = []
    for c in g.columns:
        series = g[c].astype(float).values
        stat, p, *_ = adfuller(series, maxlag=5, autolag=None)
        adf_rows.append({"series": c, "adf_stat": stat, "p_value": p, "stationary_5pct": p < 0.05})
    pd.DataFrame(adf_rows).to_csv(paths["tables"] / "rq4_adf.csv", index=False)
    maxlag = 7
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
    model = VAR(var_data.iloc[:split]).fit(maxlags=5, ic="aic")
    lag_order = max(int(model.k_ar), 1)
    if model.k_ar < 1:
        model = VAR(var_data.iloc[:split]).fit(1)
        lag_order = 1
    steps = min(30, len(var_data) - split)
    forecast = model.forecast(var_data.iloc[split - lag_order : split].values, steps=steps)
    actual = var_data.iloc[split : split + len(forecast)]["d_grain"].values
    pred = forecast[:, 1]
    # sMAPE avoids explosion when actual ≈ 0 (common for daily spread changes)
    smape = float(np.mean(2.0 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-8)) * 100)
    mae = float(mean_absolute_error(actual, pred))
    m4 = pd.DataFrame(
        [
            {
                "Model": "Granger+VAR",
                "Feature Set": "d_wti,d_cong,d_grain",
                "Validation Approach": "chronological VAR holdout",
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
    plot = work4.dropna(subset=["d_wti", "d_grain"]).iloc[-400:]
    ax.plot(plot["date"], plot["d_wti"].rolling(5).mean(), label="WTI % change (5d MA)", color="#1b263b")
    ax.plot(plot["date"], plot["d_grain"].rolling(5).mean(), label="Grain spread % change (5d MA)", color="#e9c46a")
    ax.set_title("Figure 7. Energy vs grain changes (lead-lag exploration)")
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
