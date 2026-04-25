# HRT Datathon 2026

This repository contains a reproducible Python pipeline for the HRT simulated
market-close prediction challenge from the 2026 Datathon by Analytics Club at ETH.

The public project focus is the final submitted strategy:

```text
pipeline/strategies/subspace_btp_hdoc_ensemble.py
```

It blends a price-path subspace downside ranker with a headline-document ranker,
then writes a standard `session,target_position` submission CSV.

## Repository Layout

```text
pipeline/                         Main strategy runner, data loading, and supported strategies
agents/features/                  Reusable price-only feature-store builder
hrt-eth-zurich-datathon-2026/      Challenge data and challenge reference notes
docs/                             Reproducibility notes and presentation material
side_challenges/                  Optional side-challenge experiments
tests/                            Unit tests for pipeline contracts
submission/                       Local generated CSV outputs, ignored by git
```

Generated feature-store files and submissions are intentionally ignored. Official
challenge parquet files remain in `hrt-eth-zurich-datathon-2026/data/`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```

## Build the Feature Store

The final strategy needs the price feature store used by the subspace ranker.

```bash
cd agents/features
../../venv/bin/python -m src.build_feature_store
cd ../..
```

This creates:

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

Output:

```text
submission/subspace-btp-hdoc-ensemble_all_test.csv
```

Use `--verbose` to show strategy-level model-selection diagnostics.

## Tests

```bash
./venv/bin/python -m unittest discover -s tests
cd agents/features
../../venv/bin/python -m unittest tests/test_features_price.py
```

## Notes

- The supported strategy interface is `fit(train_split, train_target_return)` plus
  `predict(split)`.
- `pipeline/strategies/` is a runnable strategy catalog: the final ensemble is
  highlighted, and alternate hackathon strategies remain executable.
- The side-challenge Bedrock/Claude scripts are optional experiments and require separate AWS
  credentials and dependencies.
- Full reproduction instructions are in `docs/REPRODUCIBILITY.md`.
