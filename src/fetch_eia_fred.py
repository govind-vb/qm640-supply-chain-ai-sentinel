"""Helper for EIA API v2 and FRED futures / price series pulls."""

EIA_HOME = "https://www.eia.gov/opendata/"
FRED_HOME = "https://fred.stlouisfed.org/"

SERIES = [
    "WTI front-month and deferred contracts",
    "Brent",
    "Heating oil",
    "Natural gas",
    "Wheat",
    "Corn",
]


def main() -> None:
    print("EIA/FRED fetch helper")
    print("EIA:", EIA_HOME)
    print("FRED:", FRED_HOME)
    print("Series:", "; ".join(SERIES))


if __name__ == "__main__":
    main()
