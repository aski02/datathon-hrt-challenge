from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features_price import (
    build_price_features,
    build_test_set,
    build_train_set,
    build_train_target,
)


def _make_seen_bars(sessions: tuple[int, ...], n_bars: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    for session in sessions:
        drift = rng.normal(0.03, 0.01)
        noise = rng.normal(0.0, 0.25, size=n_bars)
        close = 100.0 + session * 0.1 + np.cumsum(drift + noise)
        open_ = close + rng.normal(0.0, 0.08, size=n_bars)
        high = np.maximum(open_, close) + np.abs(rng.normal(0.05, 0.02, size=n_bars))
        low = np.minimum(open_, close) - np.abs(rng.normal(0.05, 0.02, size=n_bars))

        for bar_ix in range(n_bars):
            rows.append(
                {
                    "session": session,
                    "bar_ix": bar_ix,
                    "open": float(open_[bar_ix]),
                    "high": float(high[bar_ix]),
                    "low": float(low[bar_ix]),
                    "close": float(close[bar_ix]),
                }
            )

    return pd.DataFrame(rows)


def _make_unseen_train_from_seen(seen: pd.DataFrame, n_bars: int = 50, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    last_seen = seen.sort_values(["session", "bar_ix"]).groupby("session")["close"].last()

    for session, close0 in last_seen.items():
        noise = rng.normal(0.0, 0.2, size=n_bars)
        drift = rng.normal(0.01, 0.01)
        close = float(close0) + np.cumsum(drift + noise)
        open_ = close + rng.normal(0.0, 0.08, size=n_bars)
        high = np.maximum(open_, close) + np.abs(rng.normal(0.05, 0.02, size=n_bars))
        low = np.minimum(open_, close) - np.abs(rng.normal(0.05, 0.02, size=n_bars))

        for j in range(n_bars):
            rows.append(
                {
                    "session": int(session),
                    "bar_ix": 50 + j,
                    "open": float(open_[j]),
                    "high": float(high[j]),
                    "low": float(low[j]),
                    "close": float(close[j]),
                }
            )

    return pd.DataFrame(rows)


class TestFeaturesPrice(unittest.TestCase):
    def test_one_row_per_session_and_sorted_index(self) -> None:
        seen = _make_seen_bars((8, 2, 5), n_bars=50, seed=11)
        seen = seen.sample(frac=1.0, random_state=123).reset_index(drop=True)

        x = build_price_features(seen)

        self.assertEqual(x.shape[0], 3)
        self.assertEqual(x.index.name, "session")
        self.assertEqual(x.index.tolist(), [2, 5, 8])
        self.assertTrue(x.index.is_unique)

    def test_deterministic_columns_and_values_under_input_shuffle(self) -> None:
        seen = _make_seen_bars((1, 2), n_bars=50, seed=4)

        x1 = build_price_features(seen)
        x2 = build_price_features(seen.sample(frac=1.0, random_state=42).reset_index(drop=True))

        self.assertEqual(x1.columns.tolist(), x2.columns.tolist())
        pd.testing.assert_frame_equal(x1, x2)

    def test_train_test_column_consistency(self) -> None:
        seen_train = _make_seen_bars((1, 2, 3), n_bars=50, seed=1)
        unseen_train = _make_unseen_train_from_seen(seen_train, n_bars=50, seed=2)
        seen_test = _make_seen_bars((100, 101), n_bars=50, seed=3)

        x_train, y_train = build_train_set(seen_train, unseen_train)
        x_test = build_test_set(seen_test)

        self.assertEqual(x_train.columns.tolist(), x_test.columns.tolist())
        self.assertTrue(y_train.index.equals(x_train.index))

    def test_target_uses_last_seen_close_and_final_unseen_close(self) -> None:
        seen_rows = []
        for i in range(50):
            close = 100.0 + i
            seen_rows.append({"session": 1, "bar_ix": i, "open": close, "high": close, "low": close, "close": close})
        seen = pd.DataFrame(seen_rows)

        unseen_rows = []
        for j in range(50):
            close = 110.0 + j
            unseen_rows.append(
                {"session": 1, "bar_ix": 50 + j, "open": close, "high": close, "low": close, "close": close}
            )
        unseen = pd.DataFrame(unseen_rows)

        y = build_train_target(seen, unseen)
        expected = (159.0 / 149.0) - 1.0
        self.assertAlmostEqual(float(y.loc[1]), expected, places=12)

        unseen_modified = unseen.copy()
        unseen_modified.loc[unseen_modified["bar_ix"] < 99, "close"] = 1e9
        y2 = build_train_target(seen, unseen_modified)
        self.assertAlmostEqual(float(y2.loc[1]), expected, places=12)

    def test_features_do_not_depend_on_unseen_train_bars(self) -> None:
        seen_train = _make_seen_bars((4, 5), n_bars=50, seed=9)
        unseen_a = _make_unseen_train_from_seen(seen_train, n_bars=50, seed=10)
        unseen_b = unseen_a.copy()
        unseen_b["close"] = unseen_b["close"] * 25.0
        unseen_b["open"] = unseen_b["open"] * 25.0
        unseen_b["high"] = unseen_b["high"] * 25.0
        unseen_b["low"] = unseen_b["low"] * 25.0

        x_a, _ = build_train_set(seen_train, unseen_a)
        x_b, _ = build_train_set(seen_train, unseen_b)

        pd.testing.assert_frame_equal(x_a, x_b)

    def test_degenerate_bars_are_safe_and_finite(self) -> None:
        rows = []
        for session in (1, 2):
            for bar_ix in range(50):
                value = 0.0 if session == 1 else 1.0
                rows.append(
                    {
                        "session": session,
                        "bar_ix": bar_ix,
                        "open": value,
                        "high": value,
                        "low": value,
                        "close": value,
                    }
                )

        seen = pd.DataFrame(rows)
        x = build_price_features(seen)

        self.assertTrue(np.isfinite(x.to_numpy()).all())
        self.assertEqual(x.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
