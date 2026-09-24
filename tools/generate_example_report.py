"""Regenerate docs/example-report.html.

Run from the repo root:  uv run python tools/generate_example_report.py

Scenario matches notebooks/01_scenario_walkthrough.ipynb: one 156-week
trend demand series (seed 0) drives the spend, split into 104 weeks of
history from 2025-01-06 (spend seed 0) and a 52-week plan from 2027-01-04
(spend seed 1), correlation 0.7. The report then builds its own demand
series from that spend at the package defaults (demand_spend_corr 0.65,
trend), the same path a real client's spend takes. Uses the default sweep
(levers unset) and default fit() sim counts. Runs in a few minutes.
"""

from pathlib import Path

from how_wrong_is_your_mmm import DiscoveryReport, simulate_demand, simulate_spend

OUT = Path(__file__).resolve().parent.parent / "docs" / "example-report.html"

CHANNELS = ["tv", "meta", "search_generic", "tiktok"]
N_HISTORY = 104  # 2 years of actuals
N_PLAN = 52  # 1 plan year


def main() -> None:
    # One trend series for history and plan, so the spend trends too.
    demand = simulate_demand(N_HISTORY + N_PLAN, process="trend", seed=0)
    history_df = simulate_spend(
        n_obs=N_HISTORY,
        correlation=0.7,
        channels=CHANNELS,
        seed=0,
        start_date="2025-01-06",
        demand=demand[:N_HISTORY],
    )
    plan_df = simulate_spend(
        n_obs=N_PLAN,
        correlation=0.7,
        channels=CHANNELS,
        seed=1,
        start_date="2027-01-04",
        demand=demand[N_HISTORY:],
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
