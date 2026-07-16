"""Cointegration-based pairs mean-reversion strategy.

Trades the spread `price_a - hedge_ratio * price_b` between two cointegrated
symbols. Enters when the rolling z-score of the spread diverges beyond
`entry_z`, exits on reversion toward the mean (|z| < exit_z) or a stop-loss
if divergence keeps widening (|z| > stop_z). `z_window`, `entry_z`, `exit_z`
are intended to be grid-searched on the in-sample window only and then
frozen for the out-of-sample evaluation (see cli.py orchestration).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date
from enum import Enum

import numpy as np

from backtester.events import Direction, MarketEvent, SignalEvent
from backtester.strategy.base import Strategy


class PositionState(Enum):
    FLAT = "FLAT"
    LONG_SPREAD = "LONG_SPREAD"
    SHORT_SPREAD = "SHORT_SPREAD"


@dataclass(frozen=True, slots=True)
class SpreadObservation:
    """One day's (spread, z-score) pair, recorded for later visualization.
    `z_score` is None until the rolling window has enough history."""

    timestamp: date
    spread: float
    z_score: float | None


class PairsMeanReversionStrategy(Strategy):
    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        hedge_ratio: float,
        z_window: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        strategy_id: str = "pairs_mean_reversion",
    ) -> None:
        if not (0 < exit_z < entry_z < stop_z):
            raise ValueError("must satisfy 0 < exit_z < entry_z < stop_z")
        if z_window < 2:
            raise ValueError("z_window must be >= 2")
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.hedge_ratio = hedge_ratio
        self.z_window = z_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.strategy_id = strategy_id

        self._price_a: float | None = None
        self._price_b: float | None = None
        self._date_a: date | None = None
        self._date_b: date | None = None
        self._spread_history: deque[float] = deque(maxlen=z_window)
        self.position_state: PositionState = PositionState.FLAT
        self.history: list[SpreadObservation] = []
        self.signal_log: list[SignalEvent] = []

    def on_market_event(self, event: MarketEvent) -> SignalEvent | None:
        if event.symbol == self.symbol_a:
            self._price_a, self._date_a = event.close, event.timestamp
        elif event.symbol == self.symbol_b:
            self._price_b, self._date_b = event.close, event.timestamp
        else:
            return None  # not one of this pair's two legs

        # Only evaluate once BOTH legs have reported for the SAME day —
        # required for a valid spread, and prevents acting on a stale price
        # for the other leg.
        if self._price_a is None or self._price_b is None or self._date_a != self._date_b:
            return None

        spread = self._price_a - self.hedge_ratio * self._price_b
        self._spread_history.append(spread)

        z_score: float | None = None
        if len(self._spread_history) >= self.z_window:
            mean = float(np.mean(self._spread_history))
            std = float(np.std(self._spread_history, ddof=1))
            if std != 0:
                z_score = (spread - mean) / std

        self.history.append(SpreadObservation(event.timestamp, spread, z_score))

        if z_score is None:
            return None
        return self._evaluate_thresholds(z_score, event.timestamp)

    def _evaluate_thresholds(self, z: float, ts: date) -> SignalEvent | None:
        if self.position_state == PositionState.FLAT:
            if z > self.entry_z:
                self.position_state = PositionState.SHORT_SPREAD
                return self._signal(Direction.SHORT, z, ts)
            if z < -self.entry_z:
                self.position_state = PositionState.LONG_SPREAD
                return self._signal(Direction.LONG, z, ts)
            return None

        # Already in a position: exit on reversion toward the mean, or on
        # further divergence past the stop-loss threshold.
        if abs(z) < self.exit_z or abs(z) > self.stop_z:
            self.position_state = PositionState.FLAT
            return self._signal(Direction.EXIT, z, ts)
        return None

    def _signal(self, direction: Direction, z: float, ts: date) -> SignalEvent:
        signal = SignalEvent(
            timestamp=ts,
            symbol=self.symbol_a,
            direction=direction,
            strategy_id=self.strategy_id,
            strength=abs(z),
            paired_symbol=self.symbol_b,
            hedge_ratio=self.hedge_ratio,
        )
        # Recorded for visualization: unlike inferring entry/exit from fill
        # BUY/SELL direction (ambiguous — a SELL fill on leg A is a SHORT
        # entry OR a LONG-spread exit), this is the ground truth of what the
        # strategy actually decided, on the day it decided it.
        self.signal_log.append(signal)
        return signal
