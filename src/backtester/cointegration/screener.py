"""Cointegration screening: scan every pair in a universe for a statistically
significant long-run equilibrium relationship (Engle-Granger), on the
IN-SAMPLE window only.

Correctness note: the Engle-Granger test is direction-asymmetric —
`coint(a, b)` and `coint(b, a)` generally give different p-values, because
the test regresses one series on the other and checks the residual for a
unit root, and OLS regression isn't symmetric. This module runs both
directions per pair and keeps whichever orientation is more significant.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True, slots=True)
class PairResult:
    dependent: str
    independent: str
    p_value: float
    hedge_ratio: float
    spread: pd.Series


def _ols_hedge_ratio(dependent: pd.Series, independent: pd.Series) -> float:
    x = sm.add_constant(independent)
    model = sm.OLS(dependent, x).fit()
    return float(model.params[independent.name])


def _best_direction(series_a: pd.Series, series_b: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    """Return (dependent, independent, p_value) for whichever regression
    direction yields the smaller (more significant) Engle-Granger p-value."""
    _, p_ab, _ = coint(series_a, series_b)
    _, p_ba, _ = coint(series_b, series_a)
    if p_ab <= p_ba:
        return series_a, series_b, float(p_ab)
    return series_b, series_a, float(p_ba)


def scan_universe(
    prices: pd.DataFrame,
    universe: list[str] | None = None,
    min_observations: int = 252,
) -> list[PairResult]:
    """`prices` must be an IN-SAMPLE-ONLY DataFrame indexed by date with one
    column per ticker (close prices). Scans every C(n,2) pair, runs the
    Engle-Granger test in both directions, computes the OLS hedge ratio for
    the winning orientation, and returns results sorted by p-value
    ascending (most-cointegrated first).
    """
    symbols = universe if universe is not None else list(prices.columns)
    results: list[PairResult] = []
    for a, b in combinations(symbols, 2):
        aligned = prices[[a, b]].dropna()
        if len(aligned) < min_observations:
            continue
        dependent, independent, p_value = _best_direction(aligned[a], aligned[b])
        hedge_ratio = _ols_hedge_ratio(dependent, independent)
        spread = dependent - hedge_ratio * independent
        results.append(
            PairResult(
                dependent=str(dependent.name),
                independent=str(independent.name),
                p_value=p_value,
                hedge_ratio=hedge_ratio,
                spread=spread,
            )
        )
    return sorted(results, key=lambda r: r.p_value)


def select_pairs(
    results: list[PairResult], p_threshold: float = 0.05, top_n: int = 1
) -> list[PairResult]:
    """Filter to statistically significant pairs (p < threshold) and return
    the top_n most significant."""
    return [r for r in results if r.p_value < p_threshold][:top_n]
