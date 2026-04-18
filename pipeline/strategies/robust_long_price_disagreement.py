from __future__ import annotations

import re

import numpy as np
import pandas as pd

from pipeline.types import SplitInput


class RobustLongPriceDisagreementStrategy:
    """Long-only strategy centered on robust de-risking.

    Design goals:
    - Respect the strong unconditional long bias in the training labels.
    - Use only a few stable seen-half price features.
    - Use headline information only as a disagreement / noise filter.
    - Avoid any dependency on external template catalogs.
    """

    name = "robust-long-price-disagreement"

    def __init__(self) -> None:
        self.base_position = 1.0
        self.min_position = 0.25
        self.max_position = 1.75

        # Fixed, low-complexity sizing weights selected from repeated CV.
        self.w_ret = 0.12
        self.w_close_pos = 0.08
        self.w_trend = 0.02
        self.w_late_n = 0.05
        self.w_abs_vol = 0.08
        self.w_late_spread = 0.08

        # Empirical body-score shrinkage toward the global long bias.
        self.body_prior_count = 25.0

        self._feature_means: dict[str, float] = {}
        self._feature_stds: dict[str, float] = {}
        self._body_score: pd.Series | None = None
        self._global_prior: float | None = None

    @staticmethod
    def _safe_std(values: pd.Series) -> float:
        std = float(values.std(ddof=0))
        return std if std > 1e-12 else 1.0

    @staticmethod
    def _normalize_body(text: str) -> str:
        parts = str(text).split()
        body = " ".join(parts[2:]) if len(parts) >= 3 else str(text)
        body = body.lower().replace("year-over-year", "yoy")
        body = re.sub(r"\s+", " ", body).strip()
        return body

    def _fit_body_score_table(self, split: SplitInput, train_target_return: pd.Series) -> None:
        rows = split.headlines[["session", "headline"]].copy()
        rows["body"] = rows["headline"].astype(str).map(self._normalize_body)
        rows = rows.merge(train_target_return.rename("target_return"), left_on="session", right_index=True, how="left")

        global_prior = float(train_target_return.mean())
        stats = rows.groupby("body")["target_return"].agg(["mean", "count"])
        stats["score"] = (
            stats["mean"] * stats["count"] + global_prior * self.body_prior_count
        ) / (stats["count"] + self.body_prior_count)

        self._body_score = stats["score"]
        self._global_prior = global_prior

    def _headline_features(self, split: SplitInput) -> pd.DataFrame:
        if self._body_score is None or self._global_prior is None:
            raise RuntimeError("Strategy must be fit before predict.")

        rows = split.headlines[["session", "bar_ix", "headline"]].copy()
        if rows.empty:
            return pd.DataFrame(
                {
                    "late_n": pd.Series(0.0, index=split.sessions),
                    "late_spread": pd.Series(0.0, index=split.sessions),
                }
            )

        rows["body"] = rows["headline"].astype(str).map(self._normalize_body)
        rows["score"] = rows["body"].map(self._body_score).fillna(self._global_prior)
        late = rows[rows["bar_ix"] >= 40]

        features = pd.DataFrame(index=split.sessions)
        features["late_n"] = late.groupby("session").size().reindex(split.sessions, fill_value=0).astype(float)
        features["late_spread"] = (
            late.groupby("session")["score"].std(ddof=0).reindex(split.sessions, fill_value=0.0).astype(float)
        )
        return features

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        target = train_target_return.reindex(train_split.sessions).fillna(0.0).astype(float)
        self._fit_body_score_table(train_split, target)

        headline_features = self._headline_features(train_split)
        train_frame = pd.DataFrame(index=train_split.sessions)
        train_frame["ret_full_seen"] = train_split.features["ret_full_seen"].fillna(0.0)
        train_frame["close_pos"] = train_split.features["close_position_in_seen_range"].fillna(0.5)
        train_frame["trend_slope"] = train_split.features["trend_slope"].fillna(0.0)
        train_frame["close_std_seen"] = train_split.features["close_std_seen"].fillna(
            train_split.features["close_std_seen"].median()
        )
        train_frame["late_n"] = headline_features["late_n"]
        train_frame["late_spread"] = headline_features["late_spread"]

        self._feature_means = {column: float(train_frame[column].mean()) for column in train_frame.columns}
        self._feature_stds = {column: self._safe_std(train_frame[column]) for column in train_frame.columns}

    def _zscore(self, values: pd.Series, column: str) -> np.ndarray:
        mean = self._feature_means[column]
        std = self._feature_stds[column]
        return ((values.astype(float) - mean) / std).to_numpy(dtype=float)

    def predict(self, split: SplitInput) -> pd.Series:
        if not self._feature_means or not self._feature_stds or self._body_score is None or self._global_prior is None:
            raise RuntimeError("Strategy must be fit before predict.")

        headline_features = self._headline_features(split)
        frame = pd.DataFrame(index=split.sessions)
        frame["ret_full_seen"] = split.features["ret_full_seen"].fillna(self._feature_means["ret_full_seen"])
        frame["close_pos"] = split.features["close_position_in_seen_range"].fillna(self._feature_means["close_pos"])
        frame["trend_slope"] = split.features["trend_slope"].fillna(self._feature_means["trend_slope"])
        frame["close_std_seen"] = split.features["close_std_seen"].fillna(self._feature_means["close_std_seen"])
        frame["late_n"] = headline_features["late_n"].fillna(self._feature_means["late_n"])
        frame["late_spread"] = headline_features["late_spread"].fillna(self._feature_means["late_spread"])

        z_ret = self._zscore(frame["ret_full_seen"], "ret_full_seen")
        z_close_pos = self._zscore(frame["close_pos"], "close_pos")
        z_trend = self._zscore(frame["trend_slope"], "trend_slope")
        z_vol = self._zscore(frame["close_std_seen"], "close_std_seen")
        z_late_n = self._zscore(frame["late_n"], "late_n")
        z_late_spread = self._zscore(frame["late_spread"], "late_spread")

        raw = (
            self.base_position
            - self.w_ret * z_ret
            - self.w_close_pos * z_close_pos
            - self.w_trend * z_trend
            - self.w_late_n * z_late_n
            - self.w_abs_vol * np.abs(z_vol)
            - self.w_late_spread * z_late_spread
        )
        positions = np.clip(raw, self.min_position, self.max_position)
        return pd.Series(positions, index=split.sessions, name="target_position")


def build_strategy() -> RobustLongPriceDisagreementStrategy:
    return RobustLongPriceDisagreementStrategy()
