# Data Dictionary (Daily Panel)

Aligned to the submitted synopsis. Update notes after PortWatch merge.

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
| finbert_risk_probability | News risk score (GDELT-based until FinBERT run) | Derived / FinBERT optional | Continuous | 0–1 | IV |
| chokepoint_vessel_count | Daily vessel/transit count | PortWatch CSV when available | Continuous | Count/day | IV RQ1 |
| port_congestion_hours | Congestion / wait indicator | PortWatch when available; otherwise documented stand-in from observed market stress | Continuous | Hours / stress units | IV RQ1 |
| vessel_draught_anomaly | Draught anomaly | PortWatch when available | Continuous | Z / meters | IV RQ1 |
| commodity_roll_yield | Near vs lagged WTI difference | Derived FRED | Continuous | USD | Feature |
| grain_calendar_spread | Wheat minus corn proxy | Derived FRED | Continuous | Spread | DV RQ4 |
| refining_crack_spread | WTI/Brent related spread | Derived | Continuous | USD | Feature |
| bottleneck_index_5d | 5-day forward bottleneck / stress target | Derived | Continuous | Index | DV RQ1 |
| volatility_regime | Normal (0) vs high-risk (1) | Derived from WTI vol | Binary | Label | DV RQ2 |
| omega_ratio_proxy | Rolling gain/loss ratio proxy | Derived | Continuous | Ratio | DV RQ3 |
| source_notes | Provenance flag for physical fields | Derived | Text | — | Metadata |
| physical_iv_note | Disclosure when PortWatch pending | Derived | Text | — | Metadata |

## GitHub paths (after collection runs)

- Raw: `data/raw/`
- Processed: `data/processed/daily_panel.csv`, `data/processed/daily_panel_clean.csv`
- Notebooks: `notebooks/`
- Outputs: `notebooks/outputs/latest/`
