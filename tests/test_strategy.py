from datetime import date, timedelta

import numpy as np

from backtester.events import Direction, MarketEvent
from backtester.strategy.pairs_mean_reversion import PairsMeanReversionStrategy


def _make_strategy(**overrides) -> PairsMeanReversionStrategy:
    params = dict(
        symbol_a="A",
        symbol_b="B",
        hedge_ratio=1.0,
        z_window=10,
        entry_z=1.5,
        exit_z=0.5,
        stop_z=4.0,
    )
    params.update(overrides)
    return PairsMeanReversionStrategy(**params)


def _push_day(strategy, day, price_a, price_b):
    """Push both legs' bars for one day. hedge_ratio=1.0 and price_b held
    constant across a test lets us treat price_a as the spread directly."""
    event_a = MarketEvent(day, strategy.symbol_a, price_a, price_a, price_a, price_a, 1000)
    event_b = MarketEvent(day, strategy.symbol_b, price_b, price_b, price_b, price_b, 1000)
    first = strategy.on_market_event(event_a)
    assert first is None  # never emitted before BOTH legs report for the day
    return strategy.on_market_event(event_b)


def test_no_signal_until_window_is_full():
    strategy = _make_strategy(z_window=5)
    day0 = date(2020, 1, 1)
    for i in range(4):
        signal = _push_day(strategy, day0 + timedelta(days=i), 10.0 + i, 0.0)
        assert signal is None
    assert len(strategy.history) == 4
    assert all(obs.z_score is None for obs in strategy.history)


def test_unrelated_symbol_is_ignored():
    strategy = _make_strategy()
    event = MarketEvent(date(2020, 1, 1), "UNRELATED", 1.0, 1.0, 1.0, 1.0, 100)
    assert strategy.on_market_event(event) is None
    assert strategy._price_a is None
    assert strategy._price_b is None


def test_entries_and_exits_strictly_alternate_no_double_entry():
    """Feed a mean-reverting spread series designed to cross the entry
    threshold repeatedly. Regardless of the exact z-score numbers, emitted
    entry/exit signals must strictly alternate ENTRY, EXIT, ENTRY, EXIT, ...
    — an entry can never follow another entry without an intervening exit."""
    strategy = _make_strategy(z_window=10, entry_z=1.0, exit_z=0.3, stop_z=6.0)
    rng = np.random.default_rng(42)
    day0 = date(2020, 1, 1)
    signals = []
    spread = 0.0
    for i in range(300):
        spread = 0.7 * spread + rng.normal(0, 3.0)  # mean-reverting with noisy excursions
        signal = _push_day(strategy, day0 + timedelta(days=i), spread, 0.0)
        if signal is not None:
            signals.append(signal)

    directions = [s.direction for s in signals]
    assert Direction.LONG in directions or Direction.SHORT in directions, (
        "scenario produced no entries"
    )
    assert Direction.EXIT in directions, "scenario produced no exits"

    state = "FLAT"
    for d in directions:
        if d in (Direction.LONG, Direction.SHORT):
            assert state == "FLAT", "entry signal fired while already in a position (double-entry)"
            state = "IN_POSITION"
        elif d == Direction.EXIT:
            assert state == "IN_POSITION", "exit signal fired while already flat"
            state = "FLAT"


def test_signal_direction_matches_z_score_sign():
    strategy = _make_strategy(z_window=5, entry_z=1.0, exit_z=0.3, stop_z=10.0)
    day0 = date(2020, 1, 1)
    signal = None
    for i, spread in enumerate([0.0, 0.0, 0.0, 0.0, 50.0]):
        signal = _push_day(strategy, day0 + timedelta(days=i), spread, 0.0)

    assert signal is not None
    # Spread jumped far ABOVE its recent mean -> z is strongly positive ->
    # short the spread (sell the expensive leg A, buy leg B).
    assert signal.direction == Direction.SHORT
    assert strategy.position_state.name == "SHORT_SPREAD"


def test_signal_carries_paired_leg_and_hedge_ratio():
    strategy = _make_strategy(hedge_ratio=1.7, z_window=5, entry_z=1.0, exit_z=0.3, stop_z=10.0)
    day0 = date(2020, 1, 1)
    signal = None
    for i, spread in enumerate([0.0, 0.0, 0.0, 0.0, -50.0]):
        signal = _push_day(strategy, day0 + timedelta(days=i), spread, 0.0)

    assert signal is not None
    assert signal.direction == Direction.LONG
    assert signal.paired_symbol == "B"
    assert signal.hedge_ratio == 1.7
    assert signal.symbol == "A"
