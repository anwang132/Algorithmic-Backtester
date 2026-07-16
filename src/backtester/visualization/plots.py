"""Matplotlib charts: equity curve, drawdown underwater chart, and the
spread/z-score chart with trade markers. Colors follow the validated
reference palette (see the dataviz skill) — a single blue hue for the
equity/spread magnitude, a diverging red for the "bad" drawdown fill, fixed
categorical slots for entry/exit direction, never a cycled/rainbow palette.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from backtester.events import Direction, SignalEvent
from backtester.strategy.pairs_mean_reversion import SpreadObservation

# --- Reference palette (see dataviz skill / references/palette.md) ---
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"  # sequential/categorical slot 1 — equity, spread
GREEN = "#008300"  # categorical slot 2 — long-spread entries (good)
RED = "#e34948"  # categorical slot 8 / diverging red pole — drawdown, short entries
ORANGE = "#eb6834"  # categorical slot 6 — z-score line
VIOLET = "#4a3aa7"  # categorical slot 7 — exits


def _new_axes(figsize: tuple[float, float]) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    return fig, ax


def plot_equity_curve(
    equity_curve: list[tuple[date, float]],
    out_of_sample_start: date | None = None,
    in_sample_sharpe: float | None = None,
    out_of_sample_sharpe: float | None = None,
) -> Figure:
    dates, values = zip(*equity_curve, strict=True)
    fig, ax = _new_axes((10, 5))
    ax.plot(dates, values, color=BLUE, linewidth=2, zorder=3)
    ax.set_title("Equity Curve", fontsize=13, fontweight="bold", loc="left")
    ax.set_ylabel("Portfolio Value ($)")

    if out_of_sample_start is not None:
        ax.axvline(out_of_sample_start, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=2)
        y_top = ax.get_ylim()[1]
        ax.text(
            out_of_sample_start,
            y_top,
            "  Out-of-sample →",
            color=INK_SECONDARY,
            fontsize=9,
            va="top",
        )

    annotation_parts = []
    if in_sample_sharpe is not None:
        annotation_parts.append(f"In-sample Sharpe: {in_sample_sharpe:.2f}")
    if out_of_sample_sharpe is not None:
        annotation_parts.append(f"Out-of-sample Sharpe: {out_of_sample_sharpe:.2f}")
    if annotation_parts:
        ax.text(
            0.01,
            0.02,
            "   |   ".join(annotation_parts),
            transform=ax.transAxes,
            fontsize=9,
            color=INK_SECONDARY,
        )

    fig.tight_layout()
    return fig


def plot_drawdown(equity_curve: list[tuple[date, float]]) -> Figure:
    dates, values = zip(*equity_curve, strict=True)
    equity = pd.Series(values, index=pd.to_datetime(list(dates)))
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100

    fig, ax = _new_axes((10, 3.5))
    ax.fill_between(drawdown.index, drawdown.values, 0, color=RED, alpha=0.35, zorder=2)
    ax.plot(drawdown.index, drawdown.values, color=RED, linewidth=1.5, zorder=3)
    ax.axhline(0, color=BASELINE, linewidth=1)
    max_dd = float(drawdown.min())
    ax.set_title(f"Drawdown  (max: {max_dd:.1f}%)", fontsize=13, fontweight="bold", loc="left")
    ax.set_ylabel("Drawdown (%)")
    fig.tight_layout()
    return fig


def plot_spread_and_zscore(
    history: list[SpreadObservation],
    signal_log: list[SignalEvent],
    symbol_a: str,
    symbol_b: str,
    entry_z: float,
    exit_z: float,
) -> Figure:
    fig, (ax_spread, ax_z) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, facecolor=SURFACE, height_ratios=[1.2, 1]
    )
    for ax in (ax_spread, ax_z):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.tick_params(colors=INK_MUTED, labelsize=9)

    dates = [obs.timestamp for obs in history]
    spreads = [obs.spread for obs in history]
    ax_spread.plot(dates, spreads, color=BLUE, linewidth=1.4, zorder=3)
    ax_spread.set_title(
        f"Spread & Z-Score: {symbol_a} vs {symbol_b}", fontsize=13, fontweight="bold", loc="left"
    )
    ax_spread.set_ylabel("Spread")

    z_dates = [obs.timestamp for obs in history if obs.z_score is not None]
    z_values = [obs.z_score for obs in history if obs.z_score is not None]
    ax_z.plot(z_dates, z_values, color=ORANGE, linewidth=1.2, zorder=3)
    for level, style in ((entry_z, "-"), (-entry_z, "-"), (exit_z, ":"), (-exit_z, ":")):
        ax_z.axhline(level, color=INK_MUTED, linewidth=0.9, linestyle=style)
    ax_z.axhline(0, color=BASELINE, linewidth=1)
    ax_z.set_ylabel("Z-Score")

    # Trade markers from the strategy's own signal log — NOT inferred from
    # fill BUY/SELL direction, which is ambiguous (a SELL fill on leg A is
    # either a fresh SHORT entry or an exit closing a LONG-spread position).
    entries_long = [s for s in signal_log if s.direction == Direction.LONG]
    entries_short = [s for s in signal_log if s.direction == Direction.SHORT]
    exits = [s for s in signal_log if s.direction == Direction.EXIT]
    spread_by_date = dict(zip(dates, spreads, strict=True))
    for signals, color, marker, label in (
        (entries_long, GREEN, "^", "Long entry"),
        (entries_short, RED, "v", "Short entry"),
        (exits, VIOLET, "o", "Exit"),
    ):
        xs = [s.timestamp for s in signals if s.timestamp in spread_by_date]
        ys = [spread_by_date[x] for x in xs]
        if xs:
            ax_spread.scatter(xs, ys, color=color, marker=marker, s=45, zorder=4, label=label)

    if entries_long or entries_short or exits:
        legend = ax_spread.legend(loc="upper left", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    fig.tight_layout()
    return fig


def save_all_charts(
    equity_curve: list[tuple[date, float]],
    signal_log: list[SignalEvent],
    history: list[SpreadObservation],
    symbol_a: str,
    symbol_b: str,
    entry_z: float,
    exit_z: float,
    charts_dir: str | Path,
    out_of_sample_start: date | None = None,
    in_sample_sharpe: float | None = None,
    out_of_sample_sharpe: float | None = None,
) -> dict[str, Path]:
    """Render all three charts and save them as PNGs. Returns a dict of
    chart-name -> saved file path."""
    out_dir = Path(charts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    figures = {
        "equity_curve.png": plot_equity_curve(
            equity_curve, out_of_sample_start, in_sample_sharpe, out_of_sample_sharpe
        ),
        "drawdown.png": plot_drawdown(equity_curve),
        "spread_zscore.png": plot_spread_and_zscore(
            history, signal_log, symbol_a, symbol_b, entry_z, exit_z
        ),
    }
    for filename, fig in figures.items():
        path = out_dir / filename
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        saved[filename] = path
    return saved
