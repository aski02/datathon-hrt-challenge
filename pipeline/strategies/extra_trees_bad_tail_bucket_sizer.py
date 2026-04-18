from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pipeline.strategies.model_risk_utils import BaseClassifierSizedLongOnlyStrategy


class ExtraTreesBadTailBucketSizerStrategy(BaseClassifierSizedLongOnlyStrategy):
    name = "extra-trees-bad-tail-bucket-sizer"

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

        survivor_risk = risk[survivors]
        q1 = float(np.quantile(survivor_risk, 1.0 / 3.0))
        q2 = float(np.quantile(survivor_risk, 2.0 / 3.0))

        positions[survivors] = 0.60
        positions[survivors & (risk <= q1)] = 1.20
        positions[survivors & (risk > q1) & (risk <= q2)] = 0.90
        positions[survivors & (risk > q2)] = 0.60
        return positions


def build_strategy() -> ExtraTreesBadTailBucketSizerStrategy:
    return ExtraTreesBadTailBucketSizerStrategy()
