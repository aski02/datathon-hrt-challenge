from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .types import SplitInput


@runtime_checkable
class Strategy(Protocol):
    """Minimal strategy protocol.

    A strategy returns one position per session for the provided split.
    `fit(...)` is optional.
    """

    name: str

    def predict(self, split: SplitInput) -> pd.Series | pd.DataFrame | np.ndarray:
        ...


def coerce_positions(
    output: pd.Series | pd.DataFrame | np.ndarray | list[float],
    sessions: pd.Index,
) -> pd.Series:
    """Normalize strategy output into a Series indexed by session."""

    if isinstance(output, pd.DataFrame):
        if {"session", "target_position"} <= set(output.columns):
            series = output.set_index("session")["target_position"]
        elif "target_position" in output.columns and len(output) == len(sessions):
            series = pd.Series(output["target_position"].to_numpy(), index=sessions)
        else:
            raise ValueError("DataFrame output must contain `session,target_position` or a target_position column.")
    elif isinstance(output, pd.Series):
        if output.index.name == "session" or output.index.isin(sessions).all():
            series = output
        elif len(output) == len(sessions):
            series = pd.Series(output.to_numpy(), index=sessions)
        else:
            raise ValueError("Series output must be indexed by session or match session count.")
    else:
        arr = np.asarray(output, dtype=float)
        if arr.ndim != 1 or len(arr) != len(sessions):
            raise ValueError("Array/list output must be 1D with one value per session.")
        series = pd.Series(arr, index=sessions)

    series = series.astype(float)
    series = series.reindex(sessions)
    if series.isna().any():
        missing = int(series.isna().sum())
        raise ValueError(f"Strategy did not provide positions for all sessions (missing={missing}).")
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("Strategy output contains non-finite values (NaN/inf).")
    return series


def sharpe_from_positions(positions: pd.Series, returns: pd.Series) -> float:
    pnl = positions.to_numpy(dtype=float) * returns.to_numpy(dtype=float)
    pnl_std = pnl.std(ddof=0)
    if pnl_std == 0:
        return 0.0
    return float(pnl.mean() / pnl_std * 16.0)
