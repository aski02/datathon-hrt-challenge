from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data import build_context
from pipeline.strategies.extra_trees_bad_tail_regime_mixture import (
    ExtraTreesBadTailRegimeMixtureStrategy,
)
from pipeline.strategies.robust_long_price_disagreement import (
    RobustLongPriceDisagreementStrategy,
)
from pipeline.types import SplitInput


class K25Cap080ZeroWeakSurvivorsQ10Strategy:
    """Best public-LB wrapper around the tree-gated robust-sizing recipe."""

    name = "k25cap080-zero-weak-survivors-q10"

    def __init__(self) -> None:
        self.base_strategy = ExtraTreesBadTailRegimeMixtureStrategy()
        self.robust_strategy = RobustLongPriceDisagreementStrategy()

        self.robust_weight = 0.25
        self.factor_floor = 0.55
        self.factor_cap = 1.45
        self.position_cap = 0.80
        self.zero_weak_quantile = 0.10

        self._test_cache: pd.Series | None = None

    @staticmethod
    def _safe_zscore(values: pd.Series) -> pd.Series:
        values = values.astype(float)
        std = float(values.std(ddof=0))
        if std <= 1e-12:
            return pd.Series(0.0, index=values.index)
        return (values - float(values.mean())) / std

    @staticmethod
    def _rescale_to_mean(values: pd.Series, target_mean: float) -> pd.Series:
        current_mean = float(values.mean())
        if abs(current_mean) <= 1e-12:
            return values
        return values * (float(target_mean) / current_mean)

    def _combine(self, base_positions: pd.Series, robust_positions: pd.Series) -> pd.Series:
        base_positions = base_positions.astype(float)
        robust_positions = robust_positions.reindex(base_positions.index).astype(float)

        robust_z = self._safe_zscore(robust_positions)
        factor = (1.0 + self.robust_weight * robust_z).clip(self.factor_floor, self.factor_cap)
        sized = (base_positions * factor).clip(0.0, self.position_cap)
        sized = self._rescale_to_mean(sized, target_mean=float(base_positions.mean()))

        positive = sized[sized > 0.0]
        if positive.empty:
            return sized.rename("target_position")

        cutoff = float(positive.quantile(self.zero_weak_quantile))
        pruned = sized.where(sized >= cutoff, 0.0)
        pruned = self._rescale_to_mean(pruned, target_mean=float(sized.mean()))
        return pruned.rename("target_position")

    def _build_test_cache(self) -> pd.Series:
        repo_root = Path(__file__).resolve().parents[2]
        context = build_context(repo_root / "hrt-eth-zurich-datathon-2026" / "data")

        public_base = self.base_strategy.predict(context.public_test)
        private_base = self.base_strategy.predict(context.private_test)
        public_robust = self.robust_strategy.predict(context.public_test)
        private_robust = self.robust_strategy.predict(context.private_test)

        base_all = pd.concat([public_base, private_base])
        robust_all = pd.concat([public_robust, private_robust])
        self._test_cache = self._combine(base_all, robust_all)
        return self._test_cache

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        self.base_strategy.fit(train_split, train_target_return)
        self.robust_strategy.fit(train_split, train_target_return)
        self._test_cache = None

    def predict(self, split: SplitInput) -> pd.Series:
        if split.name in {"public_seen", "private_seen"}:
            cache = self._test_cache if self._test_cache is not None else self._build_test_cache()
            return cache.reindex(split.sessions).rename("target_position")

        base_positions = self.base_strategy.predict(split)
        robust_positions = self.robust_strategy.predict(split)
        return self._combine(base_positions, robust_positions).reindex(split.sessions)


def build_strategy() -> K25Cap080ZeroWeakSurvivorsQ10Strategy:
    return K25Cap080ZeroWeakSurvivorsQ10Strategy()
