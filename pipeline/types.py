from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitInput:
    """All seen-half inputs available for one split."""

    name: str
    bars: pd.DataFrame
    headlines: pd.DataFrame
    sessions: pd.Index
    features: pd.DataFrame


@dataclass(frozen=True)
class PipelineContext:
    """Runner-side data bundle for strategy training/inference."""

    train: SplitInput
    public_test: SplitInput
    private_test: SplitInput
    train_target_return: pd.Series

    @property
    def test_sessions(self) -> pd.Index:
        return self.public_test.sessions.append(self.private_test.sessions)


# Backward-compatible alias; strategies should not use global context in predict().
StrategyContext = PipelineContext
