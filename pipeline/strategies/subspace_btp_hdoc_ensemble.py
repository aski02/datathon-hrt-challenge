from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.ensemble_submissions import (
    blend_disagreement_guard,
    blend_level_qmap,
    blend_rank_perm,
)
from pipeline.strategies.btp_rank_hdoc import BtpRankHdocStrategy
from pipeline.strategies.subspace_bagged_downside_ranker import (
    DEFAULT_FEATURE_STORE_DIR,
    SubspaceBaggedDownsideRankerStrategy,
)
from pipeline.strategy import sharpe_from_positions
from pipeline.types import SplitInput

LOGGER = logging.getLogger(__name__)


class BlendConfig:
    def __init__(self, name: str, method: str, primary: float, secondary: float | None = None) -> None:
        self.name = name
        self.method = method
        self.primary = float(primary)
        self.secondary = None if secondary is None else float(secondary)


class SubspaceBtpHdocEnsembleStrategy:
    """
    In-code ensemble of the subspace downside ranker and the headline-doc ranker.

    The subspace model is treated as the anchor. The blend therefore preserves
    either the subspace exposure distribution exactly or stays close to it.
    """

    name = "subspace-btp-hdoc-ensemble"

    def __init__(self, feature_store_dir: Path = DEFAULT_FEATURE_STORE_DIR) -> None:
        self.feature_store_dir = Path(feature_store_dir)
        self._subspace = SubspaceBaggedDownsideRankerStrategy(feature_store_dir=self.feature_store_dir)
        self._hdoc = BtpRankHdocStrategy()
        self._selected_config = BlendConfig(name="level_qmap_wa0p70", method="level_qmap", primary=0.70)
        self._subspace_train_sharpe: float | None = None
        self._hdoc_train_sharpe: float | None = None
        self._ensemble_train_sharpe: float | None = None

    @staticmethod
    def _candidate_configs() -> tuple[BlendConfig, ...]:
        return (
            BlendConfig(name="level_qmap_wa0p60", method="level_qmap", primary=0.60),
            BlendConfig(name="level_qmap_wa0p65", method="level_qmap", primary=0.65),
            BlendConfig(name="level_qmap_wa0p70", method="level_qmap", primary=0.70),
            BlendConfig(name="level_qmap_wa0p75", method="level_qmap", primary=0.75),
            BlendConfig(name="level_qmap_wa0p80", method="level_qmap", primary=0.80),
            BlendConfig(name="rank_perm_wa0p65", method="rank_perm", primary=0.65),
            BlendConfig(name="rank_perm_wa0p70", method="rank_perm", primary=0.70),
            BlendConfig(name="rank_perm_wa0p75", method="rank_perm", primary=0.75),
            BlendConfig(name="guard_mw0p35_p1p5", method="disagreement_guard", primary=0.35, secondary=1.5),
            BlendConfig(name="guard_mw0p45_p1p7", method="disagreement_guard", primary=0.45, secondary=1.7),
            BlendConfig(name="guard_mw0p55_p2p0", method="disagreement_guard", primary=0.55, secondary=2.0),
        )

    @staticmethod
    def _blend(anchor: np.ndarray, other: np.ndarray, config: BlendConfig) -> np.ndarray:
        if config.method == "level_qmap":
            return blend_level_qmap(anchor, other, w_anchor=config.primary)
        if config.method == "rank_perm":
            return blend_rank_perm(anchor, other, w_anchor=config.primary)
        if config.method == "disagreement_guard":
            if config.secondary is None:
                raise ValueError("disagreement_guard requires a secondary power parameter.")
            return blend_disagreement_guard(anchor, other, max_w_other=config.primary, power=config.secondary)
        raise ValueError(f"Unsupported blend method: {config.method}")

    @staticmethod
    def _score_positions(positions: np.ndarray, sessions: pd.Index, target_return: pd.Series) -> float:
        position_series = pd.Series(positions, index=sessions, name="target_position")
        y = target_return.reindex(sessions).astype(float)
        return float(sharpe_from_positions(position_series, y))

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        self._subspace.fit(train_split, train_target_return)
        self._hdoc.fit(train_split, train_target_return)

        subspace_train = self._subspace.predict(train_split).reindex(train_split.sessions).astype(float)
        hdoc_train = self._hdoc.predict(train_split).reindex(train_split.sessions).astype(float)

        self._subspace_train_sharpe = self._score_positions(
            subspace_train.to_numpy(dtype=float),
            train_split.sessions,
            train_target_return,
        )
        self._hdoc_train_sharpe = self._score_positions(
            hdoc_train.to_numpy(dtype=float),
            train_split.sessions,
            train_target_return,
        )

        best_config = self._selected_config
        best_score = -np.inf

        anchor = subspace_train.to_numpy(dtype=float)
        other = hdoc_train.to_numpy(dtype=float)
        for config in self._candidate_configs():
            blended = self._blend(anchor, other, config)
            score = self._score_positions(blended, train_split.sessions, train_target_return)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_config = config

        self._selected_config = best_config
        self._ensemble_train_sharpe = float(best_score)

        LOGGER.info(
            "[subspace-btp-hdoc-ensemble] "
            f"subspace_train_sharpe={self._subspace_train_sharpe:.4f}, "
            f"hdoc_train_sharpe={self._hdoc_train_sharpe:.4f}, "
            f"selected_blend={self._selected_config.name}, "
            f"ensemble_train_sharpe={self._ensemble_train_sharpe:.4f}"
        )

    def predict(self, split: SplitInput) -> pd.Series:
        anchor = self._subspace.predict(split).reindex(split.sessions).to_numpy(dtype=float)
        other = self._hdoc.predict(split).reindex(split.sessions).to_numpy(dtype=float)
        blended = self._blend(anchor, other, self._selected_config)
        return pd.Series(blended, index=split.sessions, name="target_position")


def build_strategy() -> SubspaceBtpHdocEnsembleStrategy:
    return SubspaceBtpHdocEnsembleStrategy()
