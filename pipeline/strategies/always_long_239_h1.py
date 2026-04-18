from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.types import SplitInput

"""
Based on always_long_239
+ small headline risk penalty: counts of "risk" patterns in headline text, with a negative weight.
The headline risk patterns are:
- "class action"
- "loses key contract"
- "scheduled maintenance"
- "decline" (as a whole word, to avoid false positives like "declined" or "declining")
"""


class AlwaysLong239HeadlineV1Strategy:
    """always_long_239 + conservative headline risk integration.

    Keeps the original long-only risk-adjusted sizing and adds a very small
    headline-based penalty for sessions with many "risk" headline patterns.
    """

    name = "always-long-239-h1"

    def __init__(self) -> None:
        # Base from always_long_239.
        self.base_position = 1.0
        self.w_vol = 0.15
        self.w_ret = 0.08
        self.w_trend = 0.03

        # New: small penalty on headline risk z-score (negative weight).
        self.w_headline_risk = -0.03

        self.min_pos = 0.25
        self.max_pos = 1.75

        self._mu_vol: float | None = None
        self._sd_vol: float | None = None
        self._mu_ret: float | None = None
        self._sd_ret: float | None = None
        self._mu_trend: float | None = None
        self._sd_trend: float | None = None
        self._mu_headline_risk: float | None = None
        self._sd_headline_risk: float | None = None

    @staticmethod
    def _safe_std(x: pd.Series) -> float:
        std = float(x.std(ddof=0))
        return std if std > 1e-12 else 1.0

    @staticmethod
    def _headline_risk_score(headline_text: pd.Series) -> pd.Series:
        low = headline_text.fillna("").str.lower()
        return (
            low.str.count(r"class action")
            + low.str.count(r"loses key contract")
            + low.str.count(r"scheduled maintenance")
            + low.str.count(r"\bdecline\b")
        ).astype(float)

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        _ = train_target_return
        f = train_split.features

        vol = f["close_std_seen"].fillna(f["close_std_seen"].median())
        ret = f["ret_full_seen"].fillna(0.0)
        trend = f["trend_slope"].fillna(0.0)
        headline_risk = self._headline_risk_score(f["headline_text"])

        self._mu_vol = float(vol.mean())
        self._sd_vol = self._safe_std(vol)
        self._mu_ret = float(ret.mean())
        self._sd_ret = self._safe_std(ret)
        self._mu_trend = float(trend.mean())
        self._sd_trend = self._safe_std(trend)
        self._mu_headline_risk = float(headline_risk.mean())
        self._sd_headline_risk = self._safe_std(headline_risk)

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
                self._mu_headline_risk,
                self._sd_headline_risk,
            )
        ):
            raise RuntimeError("Strategy must be fit before predict.")

        f = split.features
        vol = f["close_std_seen"].fillna(self._mu_vol).to_numpy(dtype=float)
        ret = f["ret_full_seen"].fillna(0.0).to_numpy(dtype=float)
        trend = f["trend_slope"].fillna(0.0).to_numpy(dtype=float)
        headline_risk = self._headline_risk_score(f["headline_text"]).to_numpy(dtype=float)

        z_vol = (vol - self._mu_vol) / self._sd_vol
        z_ret = (ret - self._mu_ret) / self._sd_ret
        z_trend = (trend - self._mu_trend) / self._sd_trend
        z_headline_risk = (headline_risk - self._mu_headline_risk) / self._sd_headline_risk

        raw = (
            self.base_position
            - self.w_vol * z_vol
            - self.w_ret * z_ret
            - self.w_trend * z_trend
            + self.w_headline_risk * z_headline_risk
        )
        positions = np.clip(raw, self.min_pos, self.max_pos)
        return pd.Series(positions, index=split.sessions, name="target_position")


def build_strategy() -> AlwaysLong239HeadlineV1Strategy:
    return AlwaysLong239HeadlineV1Strategy()
