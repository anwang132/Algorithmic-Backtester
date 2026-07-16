"""Deterministic synthetic price series for testing the cointegration
screener and strategy against ground truth, without any network dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_cointegrated_pair(
    n: int = 1000,
    hedge_ratio: float = 1.5,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """series_a = hedge_ratio * series_b + stationary AR(1) noise, where
    series_b is a random walk. By construction, series_a and series_b are
    cointegrated with the known hedge ratio `hedge_ratio`."""
    rng = np.random.default_rng(seed)
    series_b = 100.0 + np.cumsum(rng.normal(0, 1, n))

    ar_phi = 0.5
    noise = np.zeros(n)
    innovations = rng.normal(0, noise_std, n)
    for i in range(1, n):
        noise[i] = ar_phi * noise[i - 1] + innovations[i]

    series_a = hedge_ratio * series_b + noise
    dates = pd.bdate_range("2018-01-01", periods=n)
    return (
        pd.Series(series_a, index=dates, name="STOCK_A"),
        pd.Series(series_b, index=dates, name="STOCK_B"),
    )


def make_independent_pair(n: int = 1000, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    """Two independent random walks — not cointegrated (with overwhelming
    probability) since they share no common stochastic trend."""
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 10_000)
    series_a = 100.0 + np.cumsum(rng_a.normal(0, 1, n))
    series_b = 100.0 + np.cumsum(rng_b.normal(0, 1, n))
    dates = pd.bdate_range("2018-01-01", periods=n)
    return (
        pd.Series(series_a, index=dates, name="INDEP_A"),
        pd.Series(series_b, index=dates, name="INDEP_B"),
    )
