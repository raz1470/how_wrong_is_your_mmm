"""Regenerate docs/example-report.html.

Run from the repo root:  uv run python tools/generate_example_report.py

Scenario matches notebooks/01_scenario_walkthrough.ipynb (history 208w seed 0
from 2023-01-09 (ends Dec 2026), plan 52w seed 1 from 2027-01-04, correlation 0.7). Uses the
default sweep (levers unset) and default fit() sim counts. Runs in about a
minute.
"""

from pathlib import Path

from how_wrong_is_your_mmm import DiscoveryReport, simulate_spend

OUT = Path(__file__).resolve().parent.parent / "docs" / "example-report.html"

CHANNELS = ["tv", "meta", "search_generic", "tiktok"]


def main() -> None:
    history_df = simulate_spend(
        n_obs=208, correlation=0.7, channels=CHANNELS, seed=0, start_date="2023-01-09"
    )
    plan_df = simulate_spend(
        n_obs=52, correlation=0.7, channels=CHANNELS, seed=1, start_date="2027-01-04"
    )
    report = DiscoveryReport(
        history_df=history_df,
        plan_df=plan_df,
        true_marginal_returns={
            "tv": 0.5,
            "meta": 1.0,
            "search_generic": 1.5,
            "tiktok": 1.2,
        },
        saturation={"tv": 0.60, "meta": 0.75, "search_generic": 0.90, "tiktok": 0.70},
        adstock={"tv": 0.50, "meta": 0.30, "search_generic": 0.10, "tiktok": 0.20},
        client_name="Demo client",
        plan_year="2027",
    )
    report.fit()
    print(f"winner: {report.winner_}")
    report.to_html(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
