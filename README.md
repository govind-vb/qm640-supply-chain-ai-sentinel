# Global Supply Chain AI Sentinel (QM 640 Capstone)

Student: Govind V B  
Mentor: Sharath Srivatsa  
Course: QM640 Data Analytics Capstone, Walsh College

This repository holds the public data access materials, data dictionary, and analysis code for my capstone project on predicting commodity logistics bottlenecks with multi-modal signals (shipping congestion features and news risk scores).

## Data policy

- No Kaggle datasets are used.
- Sources are public APIs or open institutional databases.
- No personally identifiable information is collected.

## Repository layout

```text
.
├── README.md
├── data_dictionary.md
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
└── src/
    ├── fetch_portwatch.py
    ├── fetch_eia_fred.py
    ├── fetch_gdelt.py
    └── build_daily_panel.py
```

## Data sources

1. IMF PortWatch: https://portwatch.imf.org/
2. NOAA MarineCadastre AIS: https://marinecadastre.gov/ais/
3. GDELT Project: https://www.gdeltproject.org/
4. U.S. EIA API v2: https://www.eia.gov/opendata/
5. FRED: https://fred.stlouisfed.org/

## How to reproduce

1. Clone this repository.
2. Create a Python 3.10+ environment.
3. Set EIA and FRED API keys as environment variables if required for those clients.
4. Run the scripts in `src/` to pull source extracts, then `build_daily_panel.py` to join the daily panel.
5. Use notebooks under `notebooks/` for exploratory work and modeling.

See `data_dictionary.md` for variable definitions used in the analysis panel.
