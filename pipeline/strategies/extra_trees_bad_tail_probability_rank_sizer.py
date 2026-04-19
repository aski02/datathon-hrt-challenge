from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from pipeline.strategies.model_risk_utils import BaseClassifierSizedLongOnlyStrategy
from pipeline.strategy import sharpe_from_positions


class ExtraTreesBadTailProbabilityRankSizerStrategy(BaseClassifierSizedLongOnlyStrategy):
    """Classifier-only risk gate with survivor-relative probability ranking for sizing."""

    name = "extra-trees-bad-tail-probability-rank-sizer"

    def __init__(self) -> None:
        super().__init__()
        self.min_survivor_position: float = 0.35
        self.max_survivor_position: float = 1.50

    def candidate_models(self) -> list[Pipeline]:
        configs = [
            (300, 4, 15),
            (400, 5, 10),
            (500, 6, 8),
        ]
        return [
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "etc",
                        ExtraTreesClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            max_features="sqrt",
                            random_state=9792048,
                            n_jobs=1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            )
            for n_estimators, max_depth, min_samples_leaf in configs
        ]

    def candidate_position_ranges(self) -> list[tuple[float, float]]:
        return [
            (0.35, 1.25),
            (0.35, 1.50),
            (0.35, 2.00),
            (0.50, 2.00),
        ]

    def fit(self, train_split, train_target_return) -> None:
        X = self._build_X(train_split)
        y = train_target_return.reindex(train_split.sessions).astype(float)
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)

        best_score = -np.inf
        best_model = None
        best_bad_tail_quantile = self.bad_tail_quantile
        best_flat_quantile = self.flat_quantile
        best_range = (self.min_survivor_position, self.max_survivor_position)

        for bad_tail_quantile in self.candidate_bad_tail_quantiles():
            cutoff_y = float(np.quantile(y, bad_tail_quantile))
            labels = (y <= cutoff_y).astype(int)
            if labels.nunique() < 2:
                continue

            for model in self.candidate_models():
                for min_position, max_position in self.candidate_position_ranges():
                    quantile_scores = {q: [] for q in self.candidate_flat_quantiles()}
                    for train_idx, valid_idx in splitter.split(X):
                        X_train = X.iloc[train_idx]
                        X_valid = X.iloc[valid_idx]
                        y_valid = y.iloc[valid_idx]
                        labels_train = labels.iloc[train_idx]

                        fitted = clone(model)
                        fitted.fit(X_train, labels_train)

                        risk_train = self._positive_score(fitted, X_train)
                        risk_valid = self._positive_score(fitted, X_valid)
                        for flat_quantile in self.candidate_flat_quantiles():
                            risk_cutoff = float(np.quantile(risk_train, 1.0 - flat_quantile))
                            positions = self._positions_from_risk_with_bounds(
                                risk_valid,
                                risk_cutoff,
                                min_position=min_position,
                                max_position=max_position,
                            )
                            quantile_scores[flat_quantile].append(
                                sharpe_from_positions(pd.Series(positions, index=y_valid.index), y_valid)
                            )

                    for flat_quantile, scores in quantile_scores.items():
                        score = float(np.mean(scores))
                        if np.isfinite(score) and score > best_score:
                            best_score = score
                            best_model = clone(model)
                            best_bad_tail_quantile = bad_tail_quantile
                            best_flat_quantile = flat_quantile
                            best_range = (min_position, max_position)

        if best_model is None:
            raise RuntimeError("No valid classifier-sized candidate found during Sharpe tuning.")

        self.bad_tail_quantile = best_bad_tail_quantile
        self.flat_quantile = best_flat_quantile
        self.min_survivor_position, self.max_survivor_position = best_range

        cutoff_y = float(np.quantile(y, self.bad_tail_quantile))
        labels = (y <= cutoff_y).astype(int)
        self.model = best_model.fit(X, labels)
        risk_train = self._positive_score(self.model, X)
        self.risk_cutoff = float(np.quantile(risk_train, 1.0 - self.flat_quantile))

    def _positions_from_risk(self, risk: np.ndarray, risk_cutoff: float) -> np.ndarray:
        return self._positions_from_risk_with_bounds(
            risk,
            risk_cutoff,
            min_position=self.min_survivor_position,
            max_position=self.max_survivor_position,
        )

    @staticmethod
    def _positions_from_risk_with_bounds(
        risk: np.ndarray,
        risk_cutoff: float,
        min_position: float,
        max_position: float,
    ) -> np.ndarray:
        positions = np.zeros(len(risk), dtype=float)
        survivors = risk < risk_cutoff
        if not np.any(survivors):
            return positions

        survivor_risk = risk[survivors]
        if len(survivor_risk) == 1:
            positions[survivors] = max_position
            return positions

        lo = float(survivor_risk.min())
        hi = float(survivor_risk.max())
        if hi <= lo + 1e-12:
            positions[survivors] = 0.5 * (min_position + max_position)
            return positions

        # Lower risk means higher conviction. Scale within survivors, not against the cutoff.
        score = 1.0 - (survivor_risk - lo) / (hi - lo)
        score = np.clip(score, 0.0, 1.0)
        positions[survivors] = min_position + (max_position - min_position) * score
        return positions


def build_strategy() -> ExtraTreesBadTailProbabilityRankSizerStrategy:
    return ExtraTreesBadTailProbabilityRankSizerStrategy()
