from __future__ import annotations

import pandas as pd

from pipeline.types import SplitInput


class TemplateStrategy:
    """Copy this file, rename the class, and implement your logic in `predict`."""

    name = "template"

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        # Optional: train_split + train_target_return are available here for fitting.
        _ = train_split, train_target_return

    def predict(self, split: SplitInput) -> pd.Series:
        # Example: use one feature and keep positions bounded.
        momentum = split.features["ret_full_seen"].fillna(0.0)
        positions = (-50.0 * momentum).clip(-2.0, 2.0)
        return positions.rename("target_position")


def build_strategy() -> TemplateStrategy:
    return TemplateStrategy()
