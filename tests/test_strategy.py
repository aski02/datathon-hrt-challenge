from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.strategy import coerce_positions, sharpe_from_positions


class TestCoercePositions(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = pd.Index([10, 20, 30], name="session")

    def test_accepts_session_indexed_series(self) -> None:
        output = pd.Series([1.5, 0.0, -2.0], index=self.sessions, name="target_position")

        result = coerce_positions(output, self.sessions)

        pd.testing.assert_series_equal(result, output.astype(float))

    def test_accepts_submission_frame(self) -> None:
        output = pd.DataFrame(
            {
                "session": [30, 10, 20],
                "target_position": [-2.0, 1.5, 0.0],
            }
        )

        result = coerce_positions(output, self.sessions)

        expected = pd.Series([1.5, 0.0, -2.0], index=self.sessions, name="target_position")
        pd.testing.assert_series_equal(result, expected)

    def test_rejects_missing_session(self) -> None:
        output = pd.Series([1.0, 2.0], index=pd.Index([10, 20], name="session"))

        with self.assertRaisesRegex(ValueError, "missing=1"):
            coerce_positions(output, self.sessions)

    def test_rejects_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            coerce_positions([1.0, np.inf, 2.0], self.sessions)


class TestSharpeFromPositions(unittest.TestCase):
    def test_zero_variance_returns_zero(self) -> None:
        positions = pd.Series([1.0, 1.0, 1.0])
        returns = pd.Series([0.0, 0.0, 0.0])

        self.assertEqual(sharpe_from_positions(positions, returns), 0.0)


if __name__ == "__main__":
    unittest.main()
