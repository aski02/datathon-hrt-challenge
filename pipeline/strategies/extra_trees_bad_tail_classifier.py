from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pipeline.strategies.model_risk_utils import BaseClassifierRiskFilterStrategy


class ExtraTreesBadTailClassifierStrategy(BaseClassifierRiskFilterStrategy):
    name = "extra-trees-bad-tail-classifier"

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
                            n_jobs=-1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            )
            for n_estimators, max_depth, min_samples_leaf in configs
        ]


def build_strategy() -> ExtraTreesBadTailClassifierStrategy:
    return ExtraTreesBadTailClassifierStrategy()
