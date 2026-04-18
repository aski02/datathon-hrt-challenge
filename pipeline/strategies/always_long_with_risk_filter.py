from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.strategy import sharpe_from_positions
from pipeline.types import SplitInput


class AlwaysLongWithRiskFilterStrategy:
    """Always long baseline with a simple model-based risk-off switch."""

    name = "always-long-with-risk-filter"

    def __init__(self) -> None:
        self.alpha: float = 10.0
        self.z_cut: float = -1.0
        self.model: Pipeline | None = None
        self.feature_cols: list[str] = []
        self.pred_mean: float = 0.0
        self.pred_std: float = 1.0

    def _build_X(self, split: SplitInput) -> pd.DataFrame:
        X = split.features.copy()
        if not self.feature_cols:
            self.feature_cols = [c for c in X.columns if c not in ("session", "headline_text")]
        return X.reindex(columns=self.feature_cols)

    def _positions_from_pred(self, pred: np.ndarray) -> pd.Series:
        z = (pred - self.pred_mean) / max(self.pred_std, 1e-9)
        positions = np.where(z < self.z_cut, 0.0, 1.0)
        return pd.Series(positions, name="target_position")

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        y = train_target_return.reindex(train_split.sessions).to_numpy(dtype=float)
        X_train = self._build_X(train_split)

        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self.model.fit(X_train, y)
        pred = self.model.predict(X_train)
        self.pred_mean = float(pred.mean())
        self.pred_std = float(pred.std(ddof=0)) if float(pred.std(ddof=0)) > 0 else 1.0

        # Small incremental step: only tune a tiny set of risk-off cutoffs.
        candidate_cuts = [-1.5, -1.0, -0.5, 0.0]
        best_cut = self.z_cut
        best_score = -np.inf
        for cut in candidate_cuts:
            self.z_cut = cut
            positions = self._positions_from_pred(pred)
            score = sharpe_from_positions(positions, pd.Series(y))
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_cut = cut
        self.z_cut = best_cut

    def predict(self, split: SplitInput) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted. `fit(...)` must run before `predict(...)`.")
        X = self._build_X(split)
        pred = self.model.predict(X)
        positions = self._positions_from_pred(pred)
        positions.index = split.sessions
        return positions


def build_strategy() -> AlwaysLongWithRiskFilterStrategy:
    return AlwaysLongWithRiskFilterStrategy()
