from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.types import SplitInput


class AlwaysLongStrategy:
    """Long-only strategy with conservative position sizing.

    Core idea:
    - Keep positions always positive (never short).
    - Downweight sessions with high seen-half volatility.
    - Add small mean-reversion tilt from seen-half return/trend.

    This is intentionally simple and robust to reduce overfitting.
    """

    name = "always-long-vol-adjusted"

    def __init__(self) -> None:
        self.base_position = 1.0
        self.w_vol = 0.15
        self.w_ret = 0.08
        self.w_trend = 0.03
        self.min_pos = 0.25
        self.max_pos = 1.75

        self._mu_vol: float | None = None
        self._sd_vol: float | None = None
        self._mu_ret: float | None = None
        self._sd_ret: float | None = None
        self._mu_trend: float | None = None
        self._sd_trend: float | None = None

    @staticmethod
    def _safe_std(x: pd.Series) -> float:
        std = float(x.std(ddof=0))
        return std if std > 1e-12 else 1.0

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        _ = train_target_return
        f = train_split.features

        vol = f["close_std_seen"].fillna(f["close_std_seen"].median())
        ret = f["ret_full_seen"].fillna(0.0)
        trend = f["trend_slope"].fillna(0.0)

        self._mu_vol = float(vol.mean())
        self._sd_vol = self._safe_std(vol)
        self._mu_ret = float(ret.mean())
        self._sd_ret = self._safe_std(ret)
        self._mu_trend = float(trend.mean())
        self._sd_trend = self._safe_std(trend)

    def predict(self, split: SplitInput) -> pd.Series:
        if any(
            value is None
            for value in (
                self._mu_vol,
                self._sd_vol,
                self._mu_ret,
                self._sd_ret,
                self._mu_trend,
                self._sd_trend,
            )
        ):
            raise RuntimeError("Strategy must be fit before predict.")

        f = split.features
        vol = f["close_std_seen"].fillna(self._mu_vol).to_numpy(dtype=float)
        ret = f["ret_full_seen"].fillna(0.0).to_numpy(dtype=float)
        trend = f["trend_slope"].fillna(0.0).to_numpy(dtype=float)

        z_vol = (vol - self._mu_vol) / self._sd_vol
        z_ret = (ret - self._mu_ret) / self._sd_ret
        z_trend = (trend - self._mu_trend) / self._sd_trend

        raw = self.base_position - self.w_vol * z_vol - self.w_ret * z_ret - self.w_trend * z_trend
        positions = np.clip(raw, self.min_pos, self.max_pos)
        return pd.Series(positions, index=split.sessions, name="target_position")


def build_strategy() -> AlwaysLongStrategy:
    return AlwaysLongStrategy()
