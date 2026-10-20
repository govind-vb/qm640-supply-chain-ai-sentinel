# Data Dictionary (Daily Panel)

Aligned to the submitted synopsis. Physical IVs are from IMF PortWatch ArcGIS Daily_Chokepoints_Data.

| Variable | Description | Source | Type | Unit | Role |
|---|---|---|---|---|---|
| date | Calendar date key | Derived | Date | YYYY-MM-DD | Index |
| wti_spot | WTI spot price | FRED DCOILWTICO | Continuous | USD | Feature / market |
| brent_spot | Brent spot price | FRED DCOILBRENTEU | Continuous | USD | Feature |
| natgas_hh | Henry Hub natural gas | FRED DHHNGSP | Continuous | USD | Feature |
| wheat | Wheat price index/proxy | FRED PWHEAMTUSDM | Continuous | Index/USD | Feature / RQ4 |
| corn | Corn price index/proxy | FRED PMAIZMTUSDM | Continuous | Index/USD | Feature / RQ4 |
| eia_wti | EIA WTI (optional) | EIA API v2 | Continuous | USD | Feature |
| gdelt_tone | GDELT DOC tone aggregate | GDELT | Continuous | Tone index | IV (RQ2/RQ3) |
| gdelt_article_volume | GDELT article volume | GDELT | Continuous | Count/index | IV |
| news_event_impact_score | News impact score | GDELT tone | Continuous | Index | IV |
| finbert_risk_probability | Mean FinBERT P(negative) on daily headlines | ProsusAI/finbert on GDELT titles | Continuous | 0–1 | IV RQ2 |
| chokepoint_vessel_count | Daily sum of PortWatch `n_total` transit calls across synopsis chokepoints | IMF PortWatch ArcGIS | Continuous | Transit calls/day | IV RQ1 |
| port_congestion_hours | PortWatch congestion **intensity** (positive z when transit is depressed vs 90-day baseline). Named for synopsis compatibility; units are stress units, not literal berth wait hours (PortWatch does not publish wait-hour fields). | Derived from PortWatch | Continuous | Stress units | IV RQ1 |
| vessel_draught_anomaly | Capacity-per-call z-score vs 90-day baseline (loading/size intensity proxy; not AIS draught meters) | Derived from PortWatch capacity/`n_total` | Continuous | Z-score | IV RQ1 |
| tanker_transit_count | Daily sum of PortWatch `n_tanker` | IMF PortWatch | Continuous | Calls/day | Feature |
| commodity_roll_yield | Near vs lagged WTI difference | Derived FRED | Continuous | USD | Feature |
| grain_calendar_spread | Wheat minus corn proxy | Derived FRED | Continuous | Spread | DV RQ4 |
| refining_crack_spread | WTI/Brent related spread | Derived | Continuous | USD | Feature |
| bottleneck_index_5d | 5-day forward mean of congestion intensity | Derived PortWatch | Continuous | Index | DV RQ1 |
| volatility_regime | Normal (0) vs high-risk (1) | Derived from WTI vol | Binary | Label | DV RQ2 |
| omega_ratio_proxy | Rolling gain/loss ratio proxy | Derived | Continuous | Ratio | DV RQ3 |
| source_notes | Provenance flag | Derived | Text | — | Metadata |
| physical_iv_note | PortWatch mapping disclosure | Derived | Text | — | Metadata |

## GitHub paths

- Raw: `data/raw/` (includes `portwatch_chokepoints_daily.csv`, `portwatch_physical_ivs_daily.csv`)
- Processed: `data/processed/daily_panel.csv`, `daily_panel_clean.csv`, `finbert_daily_scores.csv`
- Notebooks: `notebooks/`
- Outputs: `notebooks/outputs/latest/`
