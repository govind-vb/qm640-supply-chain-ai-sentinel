"""Download IMF PortWatch daily chokepoint data via public ArcGIS FeatureServer.

No API key required. Source:
https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

CHOKEPOINTS_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
    "/Daily_Chokepoints_Data/FeatureServer/0"
)

# Synopsis chokepoints: match PortWatch portname / portid strings (case-insensitive)
SYNOPSIS_CHOKEPOINT_PATTERNS = (
    "hormuz",
    "suez",
    "panama",
    "bab el-mandeb",
    "bab el mandeb",
    "bab-el-mandeb",
    "red sea",
    "cape of good hope",
    "malacca",
    "turkish straits",
    "bosphorus",
    "bosporus",
)

PAGE_SIZE = 1000
MAX_RETRIES = 4
RETRY_DELAY = 4.0
REQUEST_DELAY = 0.35
UA = {"User-Agent": "qm640-sentinel/1.0 (academic research; PortWatch open data)"}


def _fetch_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=120)
            r.raise_for_status()
            if not r.text.strip().startswith("{"):
                raise RuntimeError(f"Non-JSON response status={r.status_code}")
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"ArcGIS query failed after retries: {last_err}")


def download_chokepoints_raw(raw_dir: Path, force: bool = False) -> pd.DataFrame:
    """Paginate the full Daily_Chokepoints_Data layer to CSV."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "portwatch_chokepoints_daily.csv"
    if out.exists() and not force and out.stat().st_size > 100_000:
        print("Reusing cached", out)
        df = pd.read_csv(out)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    count_payload = _fetch_json(
        f"{CHOKEPOINTS_URL}/query",
        {"where": "1=1", "returnCountOnly": "true", "f": "json"},
    )
    total = int(count_payload.get("count") or 0)
    if total <= 0:
        raise RuntimeError("PortWatch ArcGIS returned zero record count")

    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < total:
        payload = _fetch_json(
            f"{CHOKEPOINTS_URL}/query",
            {
                "where": "1=1",
                "outFields": "*",
                "orderByFields": "ObjectId",
                "resultOffset": str(offset),
                "resultRecordCount": str(PAGE_SIZE),
                "f": "json",
            },
        )
        features = payload.get("features") or []
        if not features:
            break
        for feat in features:
            attrs = feat.get("attributes") or {}
            rows.append(attrs)
        offset += len(features)
        print(f"PortWatch ArcGIS downloaded {min(offset, total)}/{total}")
        time.sleep(REQUEST_DELAY)
        if len(features) < PAGE_SIZE:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("PortWatch ArcGIS returned no feature rows")

    # Normalize date
    if "date" in df.columns:
        # May be YYYY-MM-DD string or epoch-ms
        sample = df["date"].dropna().astype(str).head(1)
        if len(sample) and sample.iloc[0].isdigit() and len(sample.iloc[0]) >= 12:
            df["date"] = pd.to_datetime(pd.to_numeric(df["date"], errors="coerce"), unit="ms", utc=True).dt.tz_localize(
                None
            )
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise RuntimeError("PortWatch layer missing date field")

    df.to_csv(out, index=False)
    print("Wrote", out, "rows=", len(df))
    return df


def _is_synopsis_chokepoint(name: str, portid: str) -> bool:
    text = f"{name} {portid}".lower()
    return any(p in text for p in SYNOPSIS_CHOKEPOINT_PATTERNS)


def aggregate_synopsis_physical_ivs(raw: pd.DataFrame) -> pd.DataFrame:
    """Map PortWatch transit/capacity fields into synopsis physical IV columns."""
    df = raw.copy()
    df["portname"] = df.get("portname", "").astype(str)
    df["portid"] = df.get("portid", "").astype(str)
    mask = df.apply(lambda r: _is_synopsis_chokepoint(r["portname"], r["portid"]), axis=1)
    sub = df.loc[mask].copy()
    if sub.empty:
        # Fall back to all chokepoints if name filter is too strict
        print("Synopsis name filter matched 0 rows; using all PortWatch chokepoints")
        sub = df.copy()

    for col in ("n_total", "n_tanker", "capacity", "capacity_tanker"):
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")

    if "n_tanker" not in sub.columns:
        sub["n_tanker"] = sub["n_total"]
    if "capacity" not in sub.columns:
        sub["capacity"] = sub["n_total"]
    daily = (
        sub.groupby("date", as_index=False)
        .agg(
            chokepoint_vessel_count=("n_total", "sum"),
            tanker_transit_count=("n_tanker", "sum"),
            capacity_total=("capacity", "sum"),
        )
        .sort_values("date")
    )

    # Congestion intensity: depressed transit vs 90-day baseline (positive = more stress)
    roll_mean = daily["chokepoint_vessel_count"].rolling(90, min_periods=30).mean()
    roll_std = daily["chokepoint_vessel_count"].rolling(90, min_periods=30).std()
    z = (roll_mean - daily["chokepoint_vessel_count"]) / (roll_std + 1e-9)
    daily["port_congestion_hours"] = z.clip(lower=0)  # stress units (see data dictionary)

    # Capacity-per-call anomaly as loading/size intensity proxy (draught stand-in name)
    cpc = daily["capacity_total"] / daily["chokepoint_vessel_count"].clip(lower=1)
    cpc_mu = cpc.rolling(90, min_periods=30).mean()
    cpc_sd = cpc.rolling(90, min_periods=30).std()
    daily["vessel_draught_anomaly"] = (cpc - cpc_mu) / (cpc_sd + 1e-9)

    daily["physical_iv_note"] = (
        "PortWatch ArcGIS Daily_Chokepoints_Data: chokepoint_vessel_count=sum(n_total); "
        "port_congestion_hours=positive transit-depression z-score (stress units, not clock hours); "
        "vessel_draught_anomaly=capacity-per-call z-score proxy"
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None)
    return daily[
        [
            "date",
            "chokepoint_vessel_count",
            "port_congestion_hours",
            "vessel_draught_anomaly",
            "tanker_transit_count",
            "physical_iv_note",
        ]
    ]


def fetch_and_aggregate(raw_dir: Path) -> pd.DataFrame:
    raw = download_chokepoints_raw(raw_dir)
    daily = aggregate_synopsis_physical_ivs(raw)
    if len(daily) < 300:
        raise RuntimeError(f"PortWatch daily aggregate too short: {len(daily)} days")
    daily.to_csv(raw_dir / "portwatch_physical_ivs_daily.csv", index=False)
    return daily


if __name__ == "__main__":
    from project_paths import ensure_output_dirs, load_env

    root = load_env()
    paths = ensure_output_dirs(root)
    out = fetch_and_aggregate(paths["raw"])
    print(out.tail())
    print(out.shape)
