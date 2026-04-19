# Memory

## Scope
- Implemented a reusable price-only feature layer under `agents/features` only.
- Ignored all headline/text data by design.

## Decisions
- Enforced deterministic preprocessing by sorting input bars with `session, bar_ix`.
- Used only seen bars for features; unseen train bars are used only in `build_train_target`.
- Chose fixed-width features based on `expected_bars=50` (configurable argument).
- Included both shape helpers:
  - raw normalized close path (`close_norm_*`) for later train-only PCA usage.
  - DCT-II coefficients (`dct_close_*`) implemented without scipy.
- Added robust numerical guards with `eps=1e-12` and finite coercion to avoid NaN/inf in degenerate bars.

## Assumptions
- Challenge schema columns are exactly: `session, bar_ix, open, high, low, close`.
- Session-bar pairs are unique; duplicates are treated as invalid and raise.
- Train seen and unseen session sets should match; mismatches raise explicit errors.

## Validation
- Added tests for:
  - one row per session
  - deterministic columns/order under row shuffle
  - train/test feature column consistency
  - target construction formula
  - no feature dependency on unseen train bars
  - degenerate bars produce finite outputs
- Local test run:
  - `../../.venv/bin/python -m unittest tests/test_features_price.py`
  - Result: `Ran 6 tests ... OK`
- CLI smoke run on challenge parquet files:
  - `../../.venv/bin/python -m src.build_feature_store`
  - Output shapes: `X_train (1000, 426)`, `y_train (1000,)`, `X_public_test (10000, 426)`, `X_private_test (10000, 426)`
