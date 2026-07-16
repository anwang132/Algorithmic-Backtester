"""Typed, validated configuration loader for the backtester.

Loads config/default.yaml (or any override path) into a pydantic model tree.
Validation catches nonsensical parameter combinations (e.g. exit_z >= entry_z)
before a multi-minute backtest run wastes time on a config that was never
going to produce a sane strategy.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, model_validator


class DateConfig(BaseModel):
    in_sample_start: date
    in_sample_end: date
    out_of_sample_start: date
    out_of_sample_end: date

    @model_validator(mode="after")
    def check_ordering(self) -> DateConfig:
        if not (
            self.in_sample_start
            < self.in_sample_end
            <= self.out_of_sample_start
            < self.out_of_sample_end
        ):
            raise ValueError(
                "dates must satisfy in_sample_start < in_sample_end <= "
                "out_of_sample_start < out_of_sample_end"
            )
        return self


class CapitalConfig(BaseModel):
    initial: float
    risk_pct_per_leg: float

    @field_validator("initial")
    @classmethod
    def positive_capital(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("capital.initial must be positive")
        return v

    @field_validator("risk_pct_per_leg")
    @classmethod
    def sane_risk_pct(cls, v: float) -> float:
        if not (0 < v <= 1):
            raise ValueError("capital.risk_pct_per_leg must be in (0, 1]")
        return v


class CostConfig(BaseModel):
    commission_fixed: float
    commission_per_share: float
    slippage_bps: float


class CointegrationConfig(BaseModel):
    p_value_threshold: float
    min_observations: int

    @field_validator("p_value_threshold")
    @classmethod
    def sane_p_value(cls, v: float) -> float:
        if not (0 < v < 1):
            raise ValueError("cointegration.p_value_threshold must be in (0, 1)")
        return v


class StrategyConfig(BaseModel):
    z_window: int
    entry_z: float
    exit_z: float
    stop_z: float
    param_grid: dict[str, list[float | int]]

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> StrategyConfig:
        if not (0 < self.exit_z < self.entry_z < self.stop_z):
            raise ValueError("strategy thresholds must satisfy 0 < exit_z < entry_z < stop_z")
        return self


class OutputConfig(BaseModel):
    results_dir: str
    charts_dir: str


class BacktesterConfig(BaseModel):
    universe: list[str]
    dates: DateConfig
    capital: CapitalConfig
    costs: CostConfig
    cointegration: CointegrationConfig
    strategy: StrategyConfig
    output: OutputConfig

    @field_validator("universe")
    @classmethod
    def universe_has_enough_symbols(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("universe must contain at least 2 symbols to form a pair")
        if len(set(v)) != len(v):
            raise ValueError("universe contains duplicate symbols")
        return v


def load_config(path: str | Path = "config/default.yaml") -> BacktesterConfig:
    """Read and validate a YAML config file into a BacktesterConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    return BacktesterConfig(**raw)
