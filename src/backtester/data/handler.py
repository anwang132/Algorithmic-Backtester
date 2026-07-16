"""Market data layer: downloads daily OHLCV via yfinance, caches to local
parquet, and exposes a chronologically-ordered trading calendar plus O(1)
per-symbol/per-day bar lookups to the engine.

Design choices (see plan for rationale):
  * Cache key is the ticker only, not the requested date range — the cache
    always holds the maximal available history so re-running with a
    different in-sample/out-of-sample split never re-hits the network.
  * `auto_adjust=True` bakes splits/dividends into OHLC so a stock split
    doesn't fabricate a fake price jump that could look like a trade signal
    or corrupt a cointegration test.
  * `get_bar` returns None for a symbol with no data on a given day rather
    than forward-filling — silently forward-filling would let a strategy
    "trade" on stale data without any signal that it happened. The trading
    calendar is the UNION of all symbols' trading days, not the
    intersection, so one symbol's data gap doesn't drop that day for the
    rest of the universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

EARLIEST_CACHE_START = date(2005, 1, 1)
CACHE_FRESHNESS_DAYS = 5


@dataclass(frozen=True, slots=True)
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: int


class DataHandler:
    def __init__(
        self,
        universe: list[str],
        start: date,
        end: date,
        cache_dir: str | Path = "data/cache",
    ) -> None:
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")
        self.universe = list(universe)
        self.start = start
        self.end = end
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._bars: dict[str, dict[date, Bar]] = {}
        self._calendar: list[date] | None = None

    def load_or_download(self) -> None:
        """Populate per-symbol bar dicts for [self.start, self.end], using
        the local parquet cache where it's fresh enough, downloading via
        yfinance otherwise."""
        for symbol in self.universe:
            full_history = self._load_or_download_symbol(symbol)
            sliced = full_history.loc[
                (full_history.index.date >= self.start) & (full_history.index.date <= self.end)
            ]
            if sliced.empty:
                raise ValueError(
                    f"No trading data for {symbol} in [{self.start}, {self.end}] "
                    "after loading/downloading"
                )
            self._bars[symbol] = {
                idx.date(): Bar(
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                )
                for idx, row in sliced.iterrows()
            }
        self._calendar = None  # invalidate cached calendar

    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}.parquet"

    def _load_or_download_symbol(self, symbol: str) -> pd.DataFrame:
        path = self._cache_path(symbol)
        if path.exists():
            cached = pd.read_parquet(path)
            if not cached.empty:
                cached_max = cached.index.max().date()
                fresh_enough = cached_max >= self.end or cached_max >= (
                    date.today() - timedelta(days=CACHE_FRESHNESS_DAYS)
                )
                if fresh_enough:
                    return cached

        df = yf.download(
            symbol,
            start=EARLIEST_CACHE_START,
            end=date.today() + timedelta(days=1),
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"yfinance returned no data for symbol {symbol!r}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.to_parquet(path)
        return df

    @property
    def trading_calendar(self) -> list[date]:
        """Sorted union of every symbol's trading dates within [start, end]."""
        if self._calendar is None:
            if not self._bars:
                raise RuntimeError("call load_or_download() before trading_calendar")
            all_dates: set[date] = set()
            for bars in self._bars.values():
                all_dates.update(bars.keys())
            self._calendar = sorted(all_dates)
        return self._calendar

    def get_bar(self, symbol: str, dt: date) -> Bar | None:
        """Return symbol's bar for dt, or None if it didn't trade that day."""
        return self._bars.get(symbol, {}).get(dt)

    def close_price_frame(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """Wide DataFrame of close prices (index=date, columns=symbol) for
        use by the cointegration screener. Optionally sliced further within
        the already-loaded [self.start, self.end] range."""
        if not self._bars:
            raise RuntimeError("call load_or_download() before close_price_frame")
        frame = pd.DataFrame(
            {
                symbol: {dt: bar.close for dt, bar in bars.items()}
                for symbol, bars in self._bars.items()
            }
        ).sort_index()
        frame.index = pd.to_datetime(frame.index)
        if start is not None:
            frame = frame.loc[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame.index <= pd.Timestamp(end)]
        return frame
