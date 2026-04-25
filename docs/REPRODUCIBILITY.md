# Reproducibility

This document describes how to reproduce the selected submission from a clean checkout.

## Environment

```bash
python3 -m venv venv
source venv/bin/activate
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```

## Data

The official challenge parquet files are expected at:

```text
hrt-eth-zurich-datathon-2026/data/
```

Generated feature-store and submission files are intentionally not tracked.

## Build the Feature Store

```bash
cd agents/features
../../venv/bin/python -m src.build_feature_store
cd ../..
```

Expected outputs:

```text
agents/features/feature_store/X_train.parquet
agents/features/feature_store/y_train.parquet
agents/features/feature_store/X_public_test.parquet
agents/features/feature_store/X_private_test.parquet
```

## Run the Final Strategy

```bash
./venv/bin/python -m pipeline.runner \
  --strategy-file pipeline/strategies/subspace_btp_hdoc_ensemble.py \
  --output-name subspace-btp-hdoc-ensemble_all_test.csv
```

Expected output:

```text
submission/subspace-btp-hdoc-ensemble_all_test.csv
```

The CSV should contain exactly two columns, `session,target_position`, and 20,000 rows.

## Tests

```bash
./venv/bin/python -m unittest discover -s tests
cd agents/features
../../venv/bin/python -m unittest tests/test_features_price.py
```
