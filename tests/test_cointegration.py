import pandas as pd
import pytest

from backtester.cointegration.screener import scan_universe, select_pairs
from tests.fixtures.synthetic_pairs import make_cointegrated_pair, make_independent_pair


def test_cointegrated_pair_is_detected_with_recovered_hedge_ratio():
    series_a, series_b = make_cointegrated_pair(n=1000, hedge_ratio=1.5, noise_std=0.5, seed=42)
    prices = pd.DataFrame({series_a.name: series_a, series_b.name: series_b})

    results = scan_universe(prices, min_observations=100)

    assert len(results) == 1
    result = results[0]
    assert result.p_value < 0.05

    # Recovered hedge ratio should be close to the ground-truth generating
    # coefficient, regardless of which direction the Engle-Granger test picked.
    if result.dependent == series_a.name:
        assert result.hedge_ratio == pytest.approx(1.5, rel=0.15)
    else:
        assert result.hedge_ratio == pytest.approx(1 / 1.5, rel=0.15)


def test_independent_pair_is_not_flagged_cointegrated():
    series_a, series_b = make_independent_pair(n=1000, seed=7)
    prices = pd.DataFrame({series_a.name: series_a, series_b.name: series_b})

    results = scan_universe(prices, min_observations=100)

    assert len(results) == 1
    assert results[0].p_value > 0.05


def test_select_pairs_filters_by_threshold_and_top_n():
    series_a, series_b = make_cointegrated_pair(seed=1)
    prices = pd.DataFrame({series_a.name: series_a, series_b.name: series_b})
    results = scan_universe(prices, min_observations=100)

    selected = select_pairs(results, p_threshold=0.05, top_n=1)
    assert len(selected) == 1

    selected_none = select_pairs(results, p_threshold=1e-30, top_n=1)
    assert selected_none == []


def test_scan_universe_skips_pairs_below_min_observations():
    series_a, series_b = make_cointegrated_pair(n=50, seed=1)
    prices = pd.DataFrame({series_a.name: series_a, series_b.name: series_b})

    results = scan_universe(prices, min_observations=100)

    assert results == []
