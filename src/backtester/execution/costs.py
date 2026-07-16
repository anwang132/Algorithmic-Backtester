"""Transaction cost models: commission and slippage.

Fixed-bps slippage (rather than a volume/spread-aware model) is the right
level of fidelity for daily OHLCV bars on a liquid large-cap universe — a
volume-participation model would need intraday depth data this project
doesn't have, and would be false precision built on a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from backtester.events import Direction


@dataclass(frozen=True, slots=True)
class CommissionModel:
    fixed_per_trade: float = 1.0
    per_share: float = 0.005

    def compute(self, quantity: int) -> float:
        return self.fixed_per_trade + abs(quantity) * self.per_share


@dataclass(frozen=True, slots=True)
class SlippageModel:
    bps: float = 5.0

    def apply(self, price: float, direction: Direction) -> float:
        """Buys fill worse (higher) than quoted price; sells fill worse
        (lower) — slippage always works against the trader, never for."""
        adjustment = price * (self.bps / 10_000)
        if direction == Direction.BUY:
            return price + adjustment
        return price - adjustment
