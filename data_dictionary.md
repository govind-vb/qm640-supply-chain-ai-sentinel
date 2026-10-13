# Data Dictionary (Daily Panel)

| Variable | Description | Source | Type | Unit | Role |
|---|---|---|---|---|---|
| date | Calendar / trading date key | Derived | Date | YYYY-MM-DD | Index |
| chokepoint_id | Shipping chokepoint code (Hormuz, Suez, Panama, etc.) | IMF PortWatch | Categorical | Code | IV / grouping |
| chokepoint_vessel_count | Daily cargo/tanker transit count | IMF PortWatch | Integer | Count/day | IV |
| port_congestion_hours | Average waiting time before berth | NOAA / PortWatch | Continuous | Hours | IV |
| vessel_draught_anomaly | Draught deviation from 3-year baseline | IMF PortWatch | Continuous | Meters | IV |
| news_event_impact_score | GDELT conflict / tone intensity | GDELT | Continuous | Index | IV |
| finbert_risk_probability | FinBERT disruption risk score | Derived (GDELT text) | Continuous | Probability 0–1 | IV |
| commodity_roll_yield | Front vs deferred futures spread | EIA / FRED | Continuous | USD or cents | DV / feature |
| refining_crack_spread | Crude vs refined product margin | EIA | Continuous | USD/bbl | Feature |
| grain_calendar_spread | Front vs deferred grain spread | FRED / USDA | Continuous | Cents/bu | DV (RQ4) |
| bottleneck_index_5d | 5-day forward physical delay index | Derived | Continuous | Index | DV (RQ1) |
| volatility_regime | Normal vs high-risk OOD state | Derived | Binary/categorical | Label | DV (RQ2) |
| omega_ratio_proxy | Rolling risk-adjusted performance proxy | Derived | Continuous | Ratio | DV (RQ3) |

## Notes

- All variables are measurable from public series or derived transforms.
- Panel target length: approximately 750 daily observations (2021–2024), exceeding Final N from sample-size calculations.
