"""Fetch public chokepoint congestion indicators from IMF PortWatch documentation/API endpoints.

This script is a helper for later milestones. It documents intended fields and does not
require credentials for public summary pages.
"""

PORTWATCH_HOME = "https://portwatch.imf.org/"

FIELDS = [
    "date",
    "chokepoint_id",
    "chokepoint_vessel_count",
    "port_congestion_hours",
    "vessel_draught_anomaly",
]


def main() -> None:
    print("PortWatch fetch helper")
    print("Home URL:", PORTWATCH_HOME)
    print("Expected fields:", ", ".join(FIELDS))


if __name__ == "__main__":
    main()
