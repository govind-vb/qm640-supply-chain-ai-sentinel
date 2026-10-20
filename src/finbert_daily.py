"""Score daily GDELT headlines with ProsusAI/finbert and cache risk probabilities.

Uses a Python 3.12 sidecar venv (.venv_finbert) because torch has no cp314 wheels.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from project_paths import ensure_output_dirs, load_env

# Force line-buffered logs when redirected
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

UA = {"User-Agent": "qm640-sentinel/1.0"}
QUERY = '(oil OR crude OR LNG OR tanker OR "strait of hormuz" OR suez OR panama OR wheat OR grain OR shipping)'


def _log(msg: str) -> None:
    print(msg, flush=True)


def fetch_gdelt_headlines(months: int = 36, max_per_call: int = 25) -> pd.DataFrame:
    """Pull GDELT DOC artlist titles for FinBERT (public API, no key).

    GDELT requires >=5s between requests. Sample one calendar day per month so
    FinBERT risk has dated coverage across the panel (not a single crawl day).
    """
    records: list[dict[str, Any]] = []
    quoted = requests.utils.quote(QUERY)
    end = datetime.now(timezone.utc).date()
    for m in range(months):
        day_dt = end - timedelta(days=30 * m)
        day = day_dt.strftime("%Y%m%d")
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={quoted}&mode=ArtList&maxrecords={max_per_call}"
            f"&format=json&startdatetime={day}000000&enddatetime={day}234500"
        )
        try:
            time.sleep(5.5)
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code != 200 or not r.text.strip().startswith("{"):
                _log(f"GDELT monthly probe {day} status={r.status_code}")
                continue
            arts = r.json().get("articles") or []
            for a in arts:
                title = (a.get("title") or "").strip()
                if not title:
                    continue
                seen = str(a.get("seendate") or "")
                # Prefer article seendate when it parses; else probe day
                art_day = seen[:8] if len(seen) >= 8 and seen[:8].isdigit() else day
                records.append({"date": art_day, "headline": title[:512]})
            _log(f"GDELT monthly probe {day} articles={len(arts)}")
        except Exception as exc:  # noqa: BLE001
            _log(f"GDELT monthly probe failed {day}: {exc}")

    _log(f"GDELT headline rows={len(records)}")
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No GDELT headlines retrieved for FinBERT")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "headline"]).drop_duplicates(subset=["date", "headline"])
    return df


def score_finbert(headlines: pd.DataFrame, batch_size: int = 8) -> pd.DataFrame:
    """Run ProsusAI/finbert; return daily mean P(negative) as finbert_risk_probability."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_id = "ProsusAI/finbert"
    _log(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    neg_idx = next((i for i, lab in id2label.items() if "neg" in lab), 1)
    _log(f"FinBERT labels={id2label} neg_idx={neg_idx}")

    texts = headlines["headline"].tolist()
    probs: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs.extend(p[:, neg_idx].tolist())
            if i % 32 == 0 or i + batch_size >= len(texts):
                _log(f"FinBERT scored {min(i + batch_size, len(texts))}/{len(texts)}")

    scored = headlines.copy()
    scored["finbert_neg_prob"] = probs
    daily = (
        scored.groupby("date", as_index=False)
        .agg(
            finbert_risk_probability=("finbert_neg_prob", "mean"),
            finbert_headline_n=("finbert_neg_prob", "count"),
        )
        .sort_values("date")
    )
    return daily


def build_finbert_scores(force: bool = False) -> pd.DataFrame:
    root = load_env()
    paths = ensure_output_dirs(root)
    cache = paths["processed"] / "finbert_daily_scores.csv"
    headlines_path = paths["raw"] / "gdelt_headlines_sample.csv"

    if cache.exists() and not force:
        _log(f"Using cached {cache}")
        return pd.read_csv(cache, parse_dates=["date"])

    headlines = fetch_gdelt_headlines()
    headlines.to_csv(headlines_path, index=False)
    _log(f"Wrote {headlines_path} rows={len(headlines)}")
    daily = score_finbert(headlines)
    daily.to_csv(cache, index=False)
    _log(f"Wrote {cache} rows={len(daily)}")
    return daily


def merge_finbert_into_panel(force: bool = False) -> pd.DataFrame:
    root = load_env()
    paths = ensure_output_dirs(root)
    panel_path = paths["processed"] / "daily_panel.csv"
    if not panel_path.exists():
        raise RuntimeError("daily_panel.csv missing; run fetch_panel first")
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    scores = build_finbert_scores(force=force)
    scores["date"] = pd.to_datetime(scores["date"])
    panel["date"] = pd.to_datetime(panel["date"])
    drop_cols = [c for c in ("finbert_risk_probability", "finbert_headline_n") if c in panel.columns]
    if drop_cols:
        panel = panel.drop(columns=drop_cols)
    merged = panel.merge(scores, on="date", how="left")
    # Monthly FinBERT probes: carry forward between observation days (honest persistence)
    merged["finbert_risk_probability"] = merged["finbert_risk_probability"].ffill().bfill()
    if "finbert_headline_n" in merged.columns:
        merged["finbert_headline_n"] = merged["finbert_headline_n"].fillna(0)
    merged.to_csv(panel_path, index=False)
    _log(
        f"Merged FinBERT into {panel_path}; "
        f"non-null={merged['finbert_risk_probability'].notna().sum()}; "
        f"score_days={scores['date'].nunique()}"
    )
    return merged


if __name__ == "__main__":
    force = "--force" in sys.argv or "-f" in sys.argv
    merge_finbert_into_panel(force=force)
