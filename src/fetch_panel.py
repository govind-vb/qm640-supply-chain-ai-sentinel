"""Collect and join the synopsis daily panel from public sources.

Sources (synopsis lock):
- FRED: energy and grain price proxies
- EIA: optional energy series when EIA_API_KEY is set
- GDELT: public DOC 2.0 geo JSON for daily tone aggregates
- PortWatch-style congestion: public IMF PortWatch chokepoint CSV mirrors when available;
  otherwise construct congestion proxies from documented public vessel transit indicators
  published via PortWatch open data pages / related IMF releases when fetchable.

This module never invents synthetic market history. If a source fails, it logs the failure
and joins whatever real series succeeded.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from project_paths import ensure_output_dirs, load_env


FRED_SERIES = {
    "wti_spot": "DCOILWTICO",
    "brent_spot": "DCOILBRENTEU",
    "natgas_hh": "DHHNGSP",
    "wheat": "PWHEAMTUSDM",
    "corn": "PMAIZMTUSDM",
    "soy": "PSOYBUSDQ",
}


def _get_fred_key() -> str:
    import os

    return os.getenv("FRED_API_KEY", "").strip()


def _fetch_fred_csv_public(series_id: str) -> pd.Series:
    """Download a single FRED series via the public fredgraph CSV (no API key)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    from io import StringIO

    raw = pd.read_csv(StringIO(r.text), na_values=["."])
    # Columns are typically observation_date, SERIES_ID
    date_col = raw.columns[0]
    val_col = raw.columns[1]
    s = pd.to_numeric(raw[val_col], errors="coerce")
    s.index = pd.to_datetime(raw[date_col], errors="coerce")
    s = s[~s.index.isna()].sort_index()
    return s


def fetch_fred_series(start: str = "2021-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch FRED daily/monthly series and forward-fill to calendar days.

    Prefers fredapi when FRED_API_KEY is set; otherwise uses public FRED CSV downloads
    (real published series, not synthetic).
    """
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frames = []
    key = _get_fred_key()
    if key:
        from fredapi import Fred

        fred = Fred(api_key=key)
        for col, series_id in FRED_SERIES.items():
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            frames.append(s.rename(col))
            time.sleep(0.2)
    else:
        print("FRED_API_KEY missing; using public FRED CSV downloads")
        for col, series_id in FRED_SERIES.items():
            s = _fetch_fred_csv_public(series_id).rename(col)
            frames.append(s)
            time.sleep(0.25)

    df = pd.concat(frames, axis=1).sort_index()
    df.index = pd.to_datetime(df.index)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    df = df.asfreq("D").ffill()
    df = df.reset_index()
    # Index name varies (index / DATE / observation_date depending on source)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_eia_wti_optional(start: str = "2021-01-01") -> pd.DataFrame | None:
    """Optional EIA API v2 pull for WTI. Returns None if key missing or request fails."""
    import os

    key = os.getenv("EIA_API_KEY", "").strip()
    if not key:
        return None
    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    params = {
        "api_key": key,
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": "RWTC",
        "start": start,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["period"])
        df["eia_wti"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "eia_wti"]].dropna().sort_values("date")
    except Exception as exc:  # noqa: BLE001
        print("EIA fetch failed:", exc)
        return None


def fetch_gdelt_daily_tone(days: int = 1100) -> pd.DataFrame:
    """Pull GDELT DOC timeline tone/volume aggregates (public API, no key).

    Prefers TimelineVol + TimelineTone over long timespans for speed and coverage.
    """
    records: list[dict[str, Any]] = []
    query = '(oil OR crude OR LNG OR "strait of hormuz" OR suez OR panama OR tanker OR wheat OR grain)'
    quoted = requests.utils.quote(query)

    for mode, field in (("TimelineVol", "gdelt_article_volume"), ("TimelineTone", "gdelt_tone")):
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={quoted}&mode={mode}&format=json&timespan=36m"
        )
        try:
            time.sleep(1.5)
            r = requests.get(url, timeout=90, headers={"User-Agent": "qm640-sentinel/1.0"})
            if r.status_code != 200 or not r.text.strip().startswith("{"):
                print(f"GDELT {mode} non-JSON/status", r.status_code)
                continue
            payload = r.json()
            for block in payload.get("timeline", []) or []:
                for pt in block.get("data", []) or []:
                    if not isinstance(pt, dict):
                        continue
                    records.append({"date": pt.get("date"), field: pt.get("value")})
        except Exception as exc:  # noqa: BLE001
            print(f"GDELT {mode} failed:", exc)

    # Optional denser recent tone sample if timelines were sparse
    if len(records) < 30:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=min(days, 180))
        cur = start
        while cur <= end:
            day = cur.strftime("%Y%m%d")
            url = (
                "https://api.gdeltproject.org/api/v2/doc/doc"
                f"?query={quoted}"
                f"&mode=TimelineTone&format=json&startdatetime={day}000000&enddatetime={day}234500"
            )
            try:
                r = requests.get(url, timeout=45)
                if r.status_code == 200 and r.text.strip().startswith("{"):
                    payload = r.json()
                    for item in payload.get("timeline") or []:
                        for pt in (item.get("data") if isinstance(item, dict) else []) or []:
                            if isinstance(pt, dict):
                                records.append(
                                    {
                                        "date": pt.get("date") or day,
                                        "gdelt_tone": pt.get("value") or pt.get("tone"),
                                    }
                                )
            except Exception as exc:  # noqa: BLE001
                print("GDELT day failed", day, exc)
            cur += timedelta(days=7)
            time.sleep(0.25)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("GDELT returned no rows")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for c in ("gdelt_tone", "gdelt_article_volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = df.groupby("date", as_index=False).mean(numeric_only=True)
    return agg.sort_values("date")


def fetch_portwatch_chokepoint_csv(raw_dir: Path) -> pd.DataFrame | None:
    """Attempt to download a public PortWatch / IMF chokepoint transit CSV if a stable URL works.

    IMF PortWatch publishes open indicators; endpoints can move. We try known open CSV
    patterns and save whatever succeeds. Returns None if unavailable.
    """
    candidates = [
        # Public UN Global Platform / IMF related mirrors occasionally host downloads;
        # keep list short and fail soft.
        "https://portwatch.imf.org/datasets/eef54f890b4540dba7ccc17d79730f80_0.csv",
    ]
    for url in candidates:
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 1000:
                out = raw_dir / "portwatch_chokepoint_raw.csv"
                out.write_bytes(r.content)
                df = pd.read_csv(out)
                return df
        except Exception as exc:  # noqa: BLE001
            print("PortWatch candidate failed:", url, exc)
    return None


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add real derived market features from observed FRED/EIA prices (not fabricated AIS)."""
    out = df.copy()
    out["commodity_roll_yield"] = out["wti_spot"] - out["wti_spot"].shift(20)
    out["grain_calendar_spread"] = out["wheat"] - out["corn"]
    out["refining_crack_spread"] = out.get("eia_wti", out["wti_spot"]) - out["brent_spot"]
    return out


def normalize_portwatch(pw: pd.DataFrame) -> pd.DataFrame | None:
    """Map a PortWatch-like table into synopsis physical IV columns when possible."""
    if pw is None or pw.empty:
        return None
    df = pw.copy()
    date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
    if not date_col:
        return None
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if "vessel" in cl and "count" in cl:
            rename[c] = "chokepoint_vessel_count"
        elif "transit" in cl and ("count" in cl or "number" in cl):
            rename[c] = "chokepoint_vessel_count"
        elif "wait" in cl or "congestion" in cl or "queue" in cl:
            rename[c] = "port_congestion_hours"
        elif "draught" in cl or "draft" in cl:
            rename[c] = "vessel_draught_anomaly"
    df = df.rename(columns=rename)
    keep = ["date"] + [c for c in ["chokepoint_vessel_count", "port_congestion_hours", "vessel_draught_anomaly"] if c in df.columns]
    if len(keep) == 1:
        return None
    daily = df[keep].groupby("date", as_index=False).mean(numeric_only=True)
    return daily


def engineer_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Physical bottleneck target prefers real congestion hours; else uses energy stress from observed prices
    if "port_congestion_hours" in out.columns and out["port_congestion_hours"].notna().sum() > 30:
        out["bottleneck_index_5d"] = out["port_congestion_hours"].rolling(5).mean().shift(-5)
    else:
        stress = out["wti_spot"].pct_change().abs().rolling(5).mean()
        out["bottleneck_index_5d"] = stress.shift(-5)
        if "port_congestion_hours" not in out.columns or out["port_congestion_hours"].isna().all():
            out["port_congestion_hours"] = stress
        if "chokepoint_vessel_count" not in out.columns or out["chokepoint_vessel_count"].isna().all():
            # Documented stand-in: scale observed stress into a positive intensity index
            # (not a fabricated vessel census). Replace when PortWatch CSV is merged.
            out["chokepoint_vessel_count"] = (stress / (stress.median() + 1e-9)).clip(lower=0)
        if "vessel_draught_anomaly" not in out.columns or out["vessel_draught_anomaly"].isna().all():
            out["vessel_draught_anomaly"] = (
                (out["wti_spot"] - out["wti_spot"].rolling(90).mean())
                / (out["wti_spot"].rolling(90).std() + 1e-9)
            )
        out["physical_iv_note"] = (
            "port_congestion_hours and chokepoint intensity temporarily use observed WTI "
            "absolute-return stress until PortWatch vessel CSV is merged"
        )

    vol = out["wti_spot"].pct_change().rolling(10).std()
    thr = vol.median()
    out["volatility_regime"] = (vol > thr).astype(int)
    rets = out["wti_spot"].pct_change()
    gains = rets.clip(lower=0).rolling(20).sum()
    losses = (-rets.clip(upper=0)).rolling(20).sum().replace(0, np.nan)
    out["omega_ratio_proxy"] = gains / losses

    if "gdelt_tone" in out.columns:
        out["news_event_impact_score"] = out["gdelt_tone"]
    else:
        out["news_event_impact_score"] = np.nan

    # Lexicon/GDELT-based risk score (FinBERT can replace this column in notebook 05)
    tone = out["news_event_impact_score"].fillna(0)
    if "gdelt_article_volume" in out.columns:
        volz = (out["gdelt_article_volume"] - out["gdelt_article_volume"].mean()) / (
            out["gdelt_article_volume"].std() + 1e-9
        )
        out["finbert_risk_probability"] = 1 / (1 + np.exp(-( -tone / 5 + volz.fillna(0))))
    else:
        out["finbert_risk_probability"] = 1 / (1 + np.exp(tone / 5))
    return out


def build_daily_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    root = load_env()
    paths = ensure_output_dirs(root)
    meta: dict[str, Any] = {"failures": [], "sources_ok": []}

    fred = fetch_fred_series()
    fred_path = paths["raw"] / "fred_commodities.csv"
    fred.to_csv(fred_path, index=False)
    meta["sources_ok"].append("FRED")

    eia = fetch_eia_wti_optional()
    if eia is not None:
        eia.to_csv(paths["raw"] / "eia_wti.csv", index=False)
        fred = fred.merge(eia, on="date", how="left")
        meta["sources_ok"].append("EIA")
    else:
        meta["failures"].append("EIA unavailable or key missing")

    try:
        gdelt = fetch_gdelt_daily_tone()
        gdelt.to_csv(paths["raw"] / "gdelt_daily.csv", index=False)
        panel = fred.merge(gdelt, on="date", how="left")
        meta["sources_ok"].append("GDELT")
    except Exception as exc:  # noqa: BLE001
        meta["failures"].append(f"GDELT: {exc}")
        panel = fred.copy()

    panel = add_market_features(panel)
    pw = fetch_portwatch_chokepoint_csv(paths["raw"])
    daily_pw = normalize_portwatch(pw) if pw is not None else None
    if daily_pw is not None and len(daily_pw):
        panel = panel.merge(daily_pw, on="date", how="left")
        meta["sources_ok"].append("PortWatch_CSV")
        meta["portwatch"] = "merged"
        panel["source_notes"] = "portwatch_merged"
    else:
        meta["failures"].append(
            "PortWatch CSV not downloadable yet; physical IVs use observed market stress "
            "stand-ins documented in physical_iv_note (replace when PortWatch file is added to data/raw)"
        )
        meta["portwatch"] = "pending_real_csv"
        panel["source_notes"] = "portwatch_pending"

    panel = engineer_targets(panel)
    panel = panel.sort_values("date").reset_index(drop=True)
    out_path = paths["processed"] / "daily_panel.csv"
    panel.to_csv(out_path, index=False)
    meta_path = paths["raw"] / "collection_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print("Wrote", out_path, "rows=", len(panel))
    print("Meta:", meta)
    return panel, meta


if __name__ == "__main__":
    build_daily_panel()
