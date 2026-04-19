# Price-Only Feature Pipeline

This folder provides a reusable, leakage-safe, session-level feature builder for the Zurich Datathon 2026 challenge.

## Files

- `src/features_price.py`: core feature engineering API.
- `src/build_feature_store.py`: CLI to build and persist train/public/private feature tables.
- `tests/test_features_price.py`: focused tests for shape, leakage safety, and target correctness.
- `memory.md`: assumptions and implementation decisions.

## Public API

```python
from src.features_price import build_price_features, build_train_set, build_test_set, build_train_target

X = build_price_features(seen_bars)
X_train, y_train = build_train_set(seen_train, unseen_train)
X_test = build_test_set(seen_test)
y_train = build_train_target(seen_train, unseen_train)
```

Outputs are indexed by `session` and deterministic in column order.

## Implemented Feature Groups

- Raw path blocks (fixed width):
  - `ret_cc_*` (close-close log returns)
  - `ret_oc_*` (open-close log returns)
  - `range_hl_*` (high-low log ranges)
  - `gap_prev_close_*` (open vs previous close log gap)
  - `candle_loc_*` (close location in bar range)
  - `body_norm_*` (normalized candle body)
- Summary window features over windows `3, 5, 10, 20, 50`:
  - cumulative log return, mean return, volatility
  - slope of log-close vs time
  - last-k arithmetic return
  - max drawdown, max run-up
  - fraction positive returns
  - mean range, mean absolute body
  - lag-1 autocorrelation, skew, kurtosis
- Shape-compression helpers:
  - `close_norm_*` normalized log-close path block (PCA-ready)
  - `dct_close_*` orthonormal DCT-II coefficients
- Global seen-half features:
  - `last_close_zscore`
  - `last_close_pos_in_seen_range`

## Build Feature Store

Run from `/Users/maxkromer/Development/datathon-eth/datathon-hrt-challenge/agents/features`:

```bash
python -m src.build_feature_store
```

This reads default parquet inputs from:

`/Users/maxkromer/Development/datathon-eth/datathon-hrt-challenge/hrt-eth-zurich-datathon-2026/data`

and writes to:

`/Users/maxkromer/Development/datathon-eth/datathon-hrt-challenge/agents/features/feature_store`

### Optional arguments

```bash
python -m src.build_feature_store \
  --seen-train /path/to/bars_seen_train.parquet \
  --unseen-train /path/to/bars_unseen_train.parquet \
  --seen-public-test /path/to/bars_seen_public_test.parquet \
  --seen-private-test /path/to/bars_seen_private_test.parquet \
  --output-dir /path/to/output \
  --output-format parquet \
  --expected-bars 50 \
  --dct-coeffs 10
```

## Run Tests

From the same folder:

```bash
python -m unittest tests/test_features_price.py
```
