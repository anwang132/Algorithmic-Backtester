"""Abstract Strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from backtester.events import MarketEvent, SignalEvent


class Strategy(ABC):
    @abstractmethod
    def on_market_event(self, event: MarketEvent) -> SignalEvent | None:
        """Called once per MarketEvent as the engine streams a trading day's
        bars. Implementations must only use `event` and internal state built
        from PAST MarketEvents — never look ahead to future bars."""
        raise NotImplementedError
