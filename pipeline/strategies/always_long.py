from __future__ import annotations

import pandas as pd

from pipeline.types import SplitInput


class AlwaysLongStrategy:
    name = "always-long"

    def predict(self, split: SplitInput) -> pd.Series:
        return pd.Series(1.0, index=split.sessions, name="target_position")


def build_strategy() -> AlwaysLongStrategy:
    return AlwaysLongStrategy()
