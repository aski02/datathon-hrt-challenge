# Datathon 2026: HRT Challenge

This repo contains our Datathon pipeline for the HRT Challenge and the model we chose to submit.

## Chosen Submission Model

Current submission model:

- `pipeline/strategies/subspace_btp_hdoc_ensemble.py`

It ensembles:

- `subspace_bagged_downside_ranker`
- `btp-rank-hdoc`

The subspace model is used as the anchor, and the ensemble selects a small
blend configuration on train before producing the final test submission.

## Setup

From repo root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Build Required Feature Store

The chosen model depends on the price feature store used by the subspace model.

From repo root:

```bash
cd agents/features
../../venv/bin/python -m src.build_feature_store
cd ../..
```

This writes:

- `agents/features/feature_store/X_train.parquet`
- `agents/features/feature_store/X_public_test.parquet`
- `agents/features/feature_store/X_private_test.parquet`
- `agents/features/feature_store/y_train.parquet`

## Run The Chosen Model

From repo root:

```bash
./venv/bin/python -m pipeline.runner \
  --strategy-file pipeline/strategies/subspace_btp_hdoc_ensemble.py \
  --output-name subspace-btp-hdoc-ensemble_all_test.csv
```

Output:

- `submission/subspace-btp-hdoc-ensemble_all_test.csv`

## Other Docs

- Pipeline details: `pipeline/README.md`
- Challenge brief: `hrt-eth-zurich-datathon-2026/README.md`
