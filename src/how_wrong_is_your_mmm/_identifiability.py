"""Identifiability diagnostic: can this spend PATTERN pin down saturation and adstock?

CollinearityDiagnostic asks whether a spend design can identify a channel's
LINEAR marginal return. This asks a different question about the same
design: even with demand correctly controlled for (no omitted-variable
bias at all), can the spend pattern itself pin down the two parameters
that make the response curve nonlinear -- the saturation exponent b and
the adstock decay lambda?

Both parameters enter nonlinearly and everything else is linear given
them, so the method profiles: grid over (b, lambda), transform the spend
columns, fit the linear model, keep the residual sum of squares, take the
argmin -- the maximum-likelihood (b, lambda) under Gaussian noise.

The recovered point is not really the answer. The answer is the SPREAD of
the recovered value across draws (b_sd/lam_sd) and the RSS valley's width:
a flat valley means many curvatures fit about equally well, which is what
"b is unmeasurable" actually means in practice.

Per-channel, not shared (reversed from the original session 44/45 design):
each channel gets its OWN plausible saturation/adstock and its OWN
recovered value -- a channel's spend pattern (how much it varies week to
week, its own autocorrelation) determines how well ITS curvature can be
pinned down, and that differs by channel even when every channel shares
the same true DGP curvature. A joint grid search over every channel's
(b, lambda) simultaneously is combinatorially infeasible (grid_size^(2K)
for K channels), so each channel is profiled on its OWN grid while every
OTHER channel is held at its own true (plausible) curvature -- the spend
pattern that channel actually has, not a hypothetical shared one.

Fits always include the true demand series as a control. That is
deliberate -- the question here is whether the SPEND PATTERN identifies
the response shape, which is separate from omitted-variable bias (see
CollinearityDiagnostic). Mixing the two would make a null result
unattributable, so unlike CollinearityDiagnostic's `controls` this is not
optional.

Promoted from tools/grid_sweep/profile_grid.py (session 45's item 5).
Per-channel profiling added later the same session, at Ryan's request.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from how_wrong_is_your_mmm._dgp import (
    _DEFAULT_MARGINAL_RETURNS,
    apply_adstock,
    simulate_sales,
)
from how_wrong_is_your_mmm._diagnostic import _validate_spend_data

# Deliberately wider than any plausible real range: when a design cannot
# identify b/lambda, estimates pile on the grid boundaries, which CAPS the
# measured spread and flatters the result. A wide grid lets a genuinely
# uninformative design look as uninformative as it actually is. See
# profile_grid.py's own comment -- ported unchanged.
_DEFAULT_B_CANDIDATES = np.round(np.linspace(0.20, 1.00, 33), 4)
_DEFAULT_LAM_CANDIDATES = np.round(np.linspace(0.00, 0.90, 31), 4)


def _per_channel_map(
    value: dict[str, float] | float,
    channels: list[str],
    name: str,
    lo: float,
    hi: float,
    lo_inclusive: bool,
    hi_inclusive: bool,
) -> dict[str, float]:
    """Normalize a float-or-per-channel-dict input to a per-channel dict,
    validating every value against (lo, hi) with the requested edge
    inclusivity. A bare float broadcasts to every channel -- the
    backward-compatible case, same convention as simulate_sales's own
    saturation/adstock params.
    """
    if isinstance(value, dict):
        missing = set(channels) - set(value)
        if missing:
            raise KeyError(f"{name} is missing channel(s): {sorted(missing)}")
        mapping = {ch: float(value[ch]) for ch in channels}
    else:
        mapping = {ch: float(value) for ch in channels}

    bound = (
        f"{'(' if not lo_inclusive else '['}{lo}, {hi}{']' if hi_inclusive else ')'}"
    )
    for ch, v in mapping.items():
        lo_ok = v >= lo if lo_inclusive else v > lo
        hi_ok = v <= hi if hi_inclusive else v < hi
        if not (lo_ok and hi_ok):
            raise ValueError(f"{name}[{ch!r}] must be in {bound}, got {v}")
    return mapping


def _design(
    spend_arr: np.ndarray,
    target_idx: int,
    b: float,
    lam: float,
    other_b: np.ndarray,
    other_lam: np.ndarray,
    demand_arr: np.ndarray,
) -> np.ndarray:
    """[1, adstock(x, lam)**b per channel, demand]. Every channel except
    `target_idx` uses ITS OWN true (b, lambda) from other_b/other_lam --
    only the profiled channel's column uses the grid candidate (b, lam)."""
    n, k = spend_arr.shape
    x = np.empty((n, k + 2))
    x[:, 0] = 1.0
    for j in range(k):
        bj, lamj = (b, lam) if j == target_idx else (other_b[j], other_lam[j])
        x[:, 1 + j] = apply_adstock(spend_arr[:, j], lamj) ** bj
    x[:, -1] = demand_arr
    return x


class IdentifiabilityDiagnostic:
    """Quantify whether a spend design identifies each channel's own
    saturation and adstock.

    Unlike CollinearityDiagnostic, this class does not generate its own
    synthetic spend -- it always takes a given design (spend_df), typically
    one already built or phased elsewhere (a BudgetPhaser recommendation,
    your own real spend, or synthetic spend you constructed yourself),
    plus the demand series that drove it. Composing this with
    CollinearityDiagnostic/BudgetPhaser's own `demand_` after fit() is the
    intended pattern -- reuse the one demand draw across all three
    diagnostics rather than drawing a second, inconsistent one.

    Parameters
    ----------
    spend_df:
        The spend design to evaluate. One column per channel.
    demand:
        The TRUE demand series that generated spend_df (length must match
        spend_df). Always included as a control in every fit -- see the
        module docstring for why this isn't optional the way
        CollinearityDiagnostic's `controls` is: the question here is
        whether the spend PATTERN identifies the curve, which only isolates
        cleanly from omitted-variable bias if demand is controlled for
        throughout.
    true_marginal_returns:
        Dict mapping channel name to true marginal return, used to simulate
        sales. Defaults to {"tv": 0.5, "meta": 1.0, "search": 1.5}, as with
        CollinearityDiagnostic -- there is no safe universal default.
    true_saturation:
        The saturation exponent b in (0, 1] to simulate against and try to
        recover, PER CHANNEL -- either a dict (must cover every channel in
        spend_df) or a single float that broadcasts to every channel (the
        old shared-value behaviour). 1.0 (default) is linear.
    true_adstock:
        The adstock decay lambda in [0, 1) to simulate against and try to
        recover, per channel (dict or broadcasting float, same convention
        as true_saturation). 0.0 (default) is no carryover.
    demand_coef:
        Coefficient on demand in the sales equation. 0.0 (default) means
        demand doesn't bias sales at all -- demand is still controlled for
        in every fit regardless (see above), so this only controls whether
        there's real omitted-variable bias for the control to remove; it
        does not affect the identifiability question this class answers.
    base_sales, revenue_noise_std:
        Forwarded to simulate_sales, same meaning as on
        CollinearityDiagnostic.
    reference_spend:
        Forwarded to simulate_sales. Defaults to spend_df's own per-channel
        mean, same as simulate_sales's own default. Fix this explicitly
        when comparing several designs (e.g. several phasing levers) so
        every design is calibrated against the SAME response curve -- see
        BudgetPhaser's own reference_spend docstring for the same point.
    b_candidates, lam_candidates:
        Grid of candidate values to profile EACH channel's own curvature
        against. Default to a 33x31 grid over b in [0.20, 1.00] and lambda
        in [0.00, 0.90] -- deliberately wide, see the module-level comment.
        Shrink these for a fast structural smoke test; the real grid is
        what any number you'd actually cite should use. The same grid is
        used for every channel.
    """

    def __init__(
        self,
        spend_df: pd.DataFrame,
        demand: np.ndarray | pd.Series,
        true_marginal_returns: dict[str, float] | None = None,
        true_saturation: dict[str, float] | float = 1.0,
        true_adstock: dict[str, float] | float = 0.0,
        demand_coef: float = 0.0,
        base_sales: float = 1_000.0,
        revenue_noise_std: float = 26_000.0,
        reference_spend: dict[str, float] | None = None,
        b_candidates: np.ndarray | None = None,
        lam_candidates: np.ndarray | None = None,
    ) -> None:
        _validate_spend_data(spend_df)
        demand_arr = np.asarray(demand, dtype=float)
        if demand_arr.shape != (len(spend_df),):
            raise ValueError(
                f"demand must be a 1-D series of length len(spend_df)="
                f"{len(spend_df)}, got shape {demand_arr.shape}"
            )

        self.spend_df = spend_df
        self.demand = demand_arr
        self.true_marginal_returns = (
            true_marginal_returns
            if true_marginal_returns is not None
            else _DEFAULT_MARGINAL_RETURNS
        )
        self.channels_ = list(spend_df.columns)
        self.true_saturation = _per_channel_map(
            true_saturation,
            self.channels_,
            "true_saturation",
            lo=0.0,
            hi=1.0,
            lo_inclusive=False,
            hi_inclusive=True,
        )
        self.true_adstock = _per_channel_map(
            true_adstock,
            self.channels_,
            "true_adstock",
            lo=0.0,
            hi=1.0,
            lo_inclusive=True,
            hi_inclusive=False,
        )
        self.demand_coef = demand_coef
        self.base_sales = base_sales
        self.revenue_noise_std = revenue_noise_std
        self.reference_spend = reference_spend
        self.b_candidates = (
            b_candidates if b_candidates is not None else _DEFAULT_B_CANDIDATES
        )
        self.lam_candidates = (
            lam_candidates if lam_candidates is not None else _DEFAULT_LAM_CANDIDATES
        )

        self.results_: dict[str, pd.DataFrame] | None = None
        self.rss_surface_: dict[str, np.ndarray] | None = None

    def fit(
        self,
        n_sims: int = 50,
        fast_mode: bool = False,
        noise_seed_offset: int = 0,
    ) -> IdentifiabilityDiagnostic:
        """Simulate n_sims sales draws at each channel's own
        (true_saturation, true_adstock), then profile EACH channel's own
        (b, lambda) grid in turn -- holding every other channel at its own
        true curvature -- and store the recovered values and RSS surface
        per channel.

        One simulate_sales() draw per sim is shared across every channel's
        profile (the true curvature doesn't change across the channel
        loop, only which channel's column the grid candidate is applied
        to), so this costs roughly `len(channels)` times one shared-grid
        fit, not `len(channels)` independent simulations.

        Parameters
        ----------
        n_sims:
            Number of noise draws. Solved against the whole grid in one
            vectorised lstsq per grid point (all n_sims sales columns at
            once), so this is cheap relative to growing the grid.
        fast_mode:
            If True, overrides n_sims=10 for quick iteration. Does NOT
            shrink the (b, lambda) grid -- pass smaller b_candidates/
            lam_candidates to __init__ for that.
        noise_seed_offset:
            Shift applied to every noise seed, same meaning as
            CollinearityDiagnostic.fit's own parameter.

        Returns
        -------
        self
        """
        if fast_mode:
            n_sims = 10

        spend_arr = self.spend_df[self.channels_].to_numpy()
        other_b = np.array([self.true_saturation[ch] for ch in self.channels_])
        other_lam = np.array([self.true_adstock[ch] for ch in self.channels_])

        sales_cols = [
            simulate_sales(
                self.spend_df,
                self.true_marginal_returns,
                base_sales=self.base_sales,
                revenue_noise_std=self.revenue_noise_std,
                seed=noise_seed_offset + sim,
                demand=self.demand,
                demand_coef=self.demand_coef,
                saturation=self.true_saturation,
                adstock=self.true_adstock,
                reference_spend=self.reference_spend,
            ).to_numpy()
            for sim in range(n_sims)
        ]
        sales_matrix = np.column_stack(sales_cols)

        results: dict[str, pd.DataFrame] = {}
        surfaces: dict[str, np.ndarray] = {}

        for idx, ch in enumerate(self.channels_):
            best_rss = np.full(n_sims, np.inf)
            best_b = np.zeros(n_sims)
            best_lam = np.zeros(n_sims)
            surface = np.zeros((len(self.b_candidates), len(self.lam_candidates)))

            for i, b in enumerate(self.b_candidates):
                for j, lam in enumerate(self.lam_candidates):
                    x = _design(spend_arr, idx, b, lam, other_b, other_lam, self.demand)
                    beta, _res, _rank, _sv = np.linalg.lstsq(
                        x, sales_matrix, rcond=None
                    )
                    resid = sales_matrix - x @ beta
                    rss = (resid**2).sum(axis=0)
                    surface[i, j] = rss.mean()
                    better = rss < best_rss
                    best_rss = np.where(better, rss, best_rss)
                    best_b = np.where(better, b, best_b)
                    best_lam = np.where(better, lam, best_lam)

            surfaces[ch] = surface
            results[ch] = pd.DataFrame(
                {
                    "sim": range(n_sims),
                    "recovered_b": best_b,
                    "recovered_lam": best_lam,
                }
            )

        self.rss_surface_ = surfaces
        self.results_ = results
        return self

    def valley_width(self, tol: float = 0.01) -> dict[str, float]:
        """Per channel: fraction of that channel's (b, lambda) grid within
        `tol` of its own minimum RSS.

        A direct reading of identification: 1.0 means every candidate in
        the grid fits about as well as the best one, so nothing is pinned
        down for that channel; near 0 means only points close to the true
        curve fit well.

        Parameters
        ----------
        tol:
            Relative tolerance above the minimum RSS counted as "tied".
            Default 0.01 (within 1%), matching profile_grid.py.
        """
        if self.rss_surface_ is None:
            raise RuntimeError("Call fit() first (to populate rss_surface_).")
        widths = {}
        for ch, surface in self.rss_surface_.items():
            lo = surface.min()
            widths[ch] = float((surface <= lo * (1.0 + tol)).mean())
        return widths

    def summary(self, tol: float = 0.01) -> pd.DataFrame:
        """Per-channel summary: recovered mean/sd/bias for b and lambda,
        and the RSS valley width at `tol` (see valley_width) -- one row
        per channel, same shape convention as
        CollinearityDiagnostic.summary().

        Returns
        -------
        pd.DataFrame with columns [channel, b_mean, b_sd, b_bias, lam_mean,
        lam_sd, lam_bias, valley_pct].
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() before summary().")

        widths = self.valley_width(tol=tol)
        rows = []
        for ch in self.channels_:
            b = self.results_[ch]["recovered_b"]
            lam = self.results_[ch]["recovered_lam"]
            rows.append(
                {
                    "channel": ch,
                    "b_mean": b.mean(),
                    "b_sd": b.std(),
                    "b_bias": b.mean() - self.true_saturation[ch],
                    "lam_mean": lam.mean(),
                    "lam_sd": lam.std(),
                    "lam_bias": lam.mean() - self.true_adstock[ch],
                    "valley_pct": 100 * widths[ch],
                }
            )
        return pd.DataFrame(rows).round(4)
