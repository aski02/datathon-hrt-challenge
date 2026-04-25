from __future__ import annotations

import unittest

import pandas as pd

from pipeline.runner import _build_submission_frame
from pipeline.types import PipelineContext, SplitInput


def _split(name: str, sessions: list[int]) -> SplitInput:
    index = pd.Index(sessions, name="session")
    return SplitInput(
        name=name,
        bars=pd.DataFrame({"session": sessions}),
        headlines=pd.DataFrame({"session": sessions}),
        sessions=index,
        features=pd.DataFrame(index=index),
    )


class TestRunnerSubmissionFrame(unittest.TestCase):
    def setUp(self) -> None:
        self.context = PipelineContext(
            train=_split("train_seen", [1, 2]),
            public_test=_split("public_seen", [10, 20]),
            private_test=_split("private_seen", [30, 40]),
            train_target_return=pd.Series([0.1, -0.1], index=pd.Index([1, 2], name="session")),
        )

    def test_builds_sorted_submission_frame(self) -> None:
        frame = _build_submission_frame(
            self.context,
            pd.Series([1.0, 2.0], index=self.context.public_test.sessions),
            pd.Series([3.0, 4.0], index=self.context.private_test.sessions),
        )

        self.assertEqual(frame.columns.tolist(), ["session", "target_position"])
        self.assertEqual(frame["session"].tolist(), [10, 20, 30, 40])
        self.assertEqual(frame["target_position"].tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_rejects_duplicate_sessions_across_splits(self) -> None:
        context = PipelineContext(
            train=self.context.train,
            public_test=_split("public_seen", [10, 20]),
            private_test=_split("private_seen", [20, 30]),
            train_target_return=self.context.train_target_return,
        )

        with self.assertRaisesRegex(ValueError, "duplicated"):
            _build_submission_frame(
                context,
                pd.Series([1.0, 2.0], index=context.public_test.sessions),
                pd.Series([3.0, 4.0], index=context.private_test.sessions),
            )


if __name__ == "__main__":
    unittest.main()
