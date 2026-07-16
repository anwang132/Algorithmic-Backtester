"""Performance metrics computed from an equity curve and trade log:
Sharpe ratio, max drawdown, CAGR, win rate, and per-trade P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from backtester.events import FillEvent


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    sharpe_ratio: float
    max_drawdown: float  # negative fraction, e.g. -0.193 for a 19.3% drawdown
    cagr: float
    total_return: float
    win_rate: float
    num_trades: int
    avg_trade_pnl: float


def compute_metrics(
    equity_curve: list[tuple[date, float]],
    trade_log: list[FillEvent],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    if len(equity_curve) < 2:
        raise ValueError("need at least 2 equity points to compute performance metrics")

    dates, values = zip(*equity_curve, strict=True)
    equity = pd.Series(values, index=pd.to_datetime(list(dates)))
    returns = equity.pct_change().dropna()

    win_rate, avg_pnl, num_trades = _trade_stats(trade_log)

    return PerformanceMetrics(
        sharpe_ratio=_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        max_drawdown=_max_drawdown(equity),
        cagr=_cagr(equity, periods_per_year),
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1),
        win_rate=win_rate,
        num_trades=num_trades,
        avg_trade_pnl=avg_pnl,
    )


def _sharpe_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int) -> float:
    if returns.empty or returns.std(ddof=1) == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1))


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def _cagr(equity: pd.Series, periods_per_year: int) -> float:
    n_periods = len(equity) - 1
    if n_periods <= 0 or equity.iloc[0] <= 0:
        return 0.0
    n_years = n_periods / periods_per_year
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)


def _trade_stats(trade_log: list[FillEvent]) -> tuple[float, float, int]:
    """FIFO-style running position/cost-basis tracking per symbol. A
    'closing fill' is one that reduces (fully or partially) or flips the
    magnitude of an open position — that's where realized P&L exists.
    Returns (win_rate, avg_realized_pnl_per_closing_fill, num_closing_fills).
    """
    position: dict[str, int] = {}
    avg_cost: dict[str, float] = {}
    realized_pnls: list[float] = []

    for fill in trade_log:
        sym = fill.symbol
        qty = fill.quantity
        pos = position.get(sym, 0)
        cost = avg_cost.get(sym, 0.0)

        same_direction_or_flat = pos == 0 or (pos > 0) == (qty > 0)
        if same_direction_or_flat:
            new_pos = pos + qty
            if new_pos != 0:
                avg_cost[sym] = (
                    fill.fill_price if pos == 0 else (cost * pos + fill.fill_price * qty) / new_pos
                )
            position[sym] = new_pos
            continue

        # Reducing or flipping an open position: realize P&L on the closed portion.
        closing_qty = min(abs(qty), abs(pos))
        direction_sign = 1 if pos > 0 else -1
        pnl = closing_qty * direction_sign * (fill.fill_price - cost) - fill.commission
        realized_pnls.append(pnl)

        new_pos = pos + qty
        position[sym] = new_pos
        if new_pos == 0:
            avg_cost[sym] = 0.0
        elif abs(qty) > abs(pos):
            # Flipped through zero to the opposite side — new cost basis is this fill's price.
            avg_cost[sym] = fill.fill_price

    if not realized_pnls:
        return 0.0, 0.0, 0
    wins = sum(1 for p in realized_pnls if p > 0)
    return wins / len(realized_pnls), sum(realized_pnls) / len(realized_pnls), len(realized_pnls)
