from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.ensemble_submissions import (
    SubmissionData,
    _align_submissions,
    _load_submission,
    blend_disagreement_guard,
    blend_level_qmap,
    blend_rank_perm,
)


class TestSubmissionEnsembling(unittest.TestCase):
    def test_blends_preserve_shape_and_anchor_distribution_when_expected(self) -> None:
        anchor = np.array([10.0, 20.0, 30.0, 40.0])
        other = np.array([4.0, 3.0, 2.0, 1.0])

        level = blend_level_qmap(anchor, other, w_anchor=0.75)
        rank = blend_rank_perm(anchor, other, w_anchor=0.75)
        guarded = blend_disagreement_guard(anchor, other, max_w_other=0.25, power=1.5)

        self.assertEqual(level.shape, anchor.shape)
        self.assertEqual(rank.shape, anchor.shape)
        self.assertEqual(guarded.shape, anchor.shape)
        np.testing.assert_allclose(np.sort(rank), np.sort(anchor))
        np.testing.assert_allclose(np.sort(guarded), np.sort(anchor))

    def test_align_rejects_different_session_sets(self) -> None:
        first = SubmissionData("a", np.array([1, 2]), np.array([0.1, 0.2]), Path("a.csv"))
        second = SubmissionData("b", np.array([1, 3]), np.array([0.1, 0.2]), Path("b.csv"))

        with self.assertRaisesRegex(ValueError, "identical session"):
            _align_submissions(first, second)

    def test_load_submission_validates_duplicate_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            pd.DataFrame(
                {
                    "session": [1, 1],
                    "target_position": [0.5, 0.6],
                }
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "duplicated"):
                _load_submission(path)


if __name__ == "__main__":
    unittest.main()
