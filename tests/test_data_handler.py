from datetime import date

import pandas as pd
import pytest

import backtester.data.handler as handler_module
from backtester.data.handler import Bar, DataHandler


def _handler_with_bars(
    bars: dict[str, dict[date, Bar]], start: date, end: date, tmp_path
) -> DataHandler:
    handler = DataHandler(list(bars.keys()), start, end, cache_dir=tmp_path)
    handler._bars = bars  # bypass load_or_download for a deterministic, network-free test
    return handler


def test_rejects_start_after_end():
    with pytest.raises(ValueError):
        DataHandler(["AAA"], date(2020, 1, 5), date(2020, 1, 1))


def test_get_bar_returns_none_for_missing_day(tmp_path):
    bars = {"AAA": {date(2020, 1, 2): Bar(1, 1, 1, 1, 100)}}
    handler = _handler_with_bars(bars, date(2020, 1, 1), date(2020, 1, 3), tmp_path)
    assert handler.get_bar("AAA", date(2020, 1, 2)) is not None
    assert handler.get_bar("AAA", date(2020, 1, 3)) is None
    assert handler.get_bar("BBB", date(2020, 1, 2)) is None


def test_trading_calendar_is_union_sorted_deduplicated(tmp_path):
    bars = {
        "AAA": {date(2020, 1, 2): Bar(1, 1, 1, 1, 1), date(2020, 1, 3): Bar(1, 1, 1, 1, 1)},
        "BBB": {date(2020, 1, 2): Bar(1, 1, 1, 1, 1), date(2020, 1, 6): Bar(1, 1, 1, 1, 1)},
    }
    handler = _handler_with_bars(bars, date(2020, 1, 1), date(2020, 1, 7), tmp_path)
    assert handler.trading_calendar == [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]


def test_close_price_frame_shape_and_slicing(tmp_path):
    bars = {
        "AAA": {date(2020, 1, 2): Bar(1, 1, 1, 10.0, 1), date(2020, 1, 3): Bar(1, 1, 1, 11.0, 1)},
        "BBB": {date(2020, 1, 2): Bar(1, 1, 1, 20.0, 1), date(2020, 1, 3): Bar(1, 1, 1, 21.0, 1)},
    }
    handler = _handler_with_bars(bars, date(2020, 1, 1), date(2020, 1, 7), tmp_path)
    frame = handler.close_price_frame()
    assert list(frame.columns) == ["AAA", "BBB"]
    assert frame.loc[pd.Timestamp(date(2020, 1, 2)), "AAA"] == 10.0

    sliced = handler.close_price_frame(start=date(2020, 1, 3))
    assert len(sliced) == 1


def test_load_or_download_uses_cache_after_first_download(tmp_path, monkeypatch):
    call_count = {"n": 0}

    def fake_download(symbol, start, end, auto_adjust, progress):  # noqa: ARG001
        call_count["n"] += 1
        idx = pd.bdate_range(date(2020, 1, 1), date(2020, 1, 10))
        return pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100}, index=idx
        )

    monkeypatch.setattr(handler_module.yf, "download", fake_download)

    h1 = DataHandler(["AAA"], date(2020, 1, 2), date(2020, 1, 8), cache_dir=tmp_path)
    h1.load_or_download()
    assert call_count["n"] == 1
    assert h1.get_bar("AAA", date(2020, 1, 3)) is not None

    # Second handler over a range fully covered by the cached data (cached
    # max 2020-01-10 >= requested end 2020-01-08) must NOT hit the network again.
    h2 = DataHandler(["AAA"], date(2020, 1, 2), date(2020, 1, 8), cache_dir=tmp_path)
    h2.load_or_download()
    assert call_count["n"] == 1
