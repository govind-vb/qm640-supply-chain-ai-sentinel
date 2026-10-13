"""Helper for GDELT event and tone pulls used for FinBERT input text."""

GDELT_HOME = "https://www.gdeltproject.org/"


def main() -> None:
    print("GDELT fetch helper")
    print("Home URL:", GDELT_HOME)
    print("Outputs: news_event_impact_score and raw article snippets for FinBERT")


if __name__ == "__main__":
    main()
