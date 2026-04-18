from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pipeline.strategies.model_risk_utils import BaseClassifierSizedLongOnlyStrategy


class ExtraTreesBadTailProbabilitySizerStrategy(BaseClassifierSizedLongOnlyStrategy):
    name = "extra-trees-bad-tail-probability-sizer"

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
                            random_state=42,
                            n_jobs=1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            )
            for n_estimators, max_depth, min_samples_leaf in configs
        ]

    def _positions_from_risk(self, risk: np.ndarray, risk_cutoff: float) -> np.ndarray:
        positions = np.zeros(len(risk), dtype=float)
        survivors = risk < risk_cutoff
        if not np.any(survivors):
            return positions

        score = 1.0 - risk[survivors] / max(risk_cutoff, 1e-9)
        score = np.clip(score, 0.0, 1.0)
        positions[survivors] = 0.35 + 0.90 * score
        return positions


def build_strategy() -> ExtraTreesBadTailProbabilitySizerStrategy:
    return ExtraTreesBadTailProbabilitySizerStrategy()
