# Strategy Catalog

This directory contains the runnable strategy catalog from the hackathon.
The final submitted model is highlighted, but the alternate strategies remain
available for comparison and follow-up experimentation.

## Final Submission

```bash
./venv/bin/python -m pipeline.runner \
  --strategy-file pipeline/strategies/subspace_btp_hdoc_ensemble.py \
  --output-name subspace-btp-hdoc-ensemble_all_test.csv
```

## Main Model Families

- `subspace_btp_hdoc_ensemble.py`: final blend of price subspace ranking and headline-document ranking.
- `subspace_bagged_downside_ranker.py`: price-only feature-store ranker used by the final blend.
- `btp_rank_hdoc.py` and `btp_rank_tpl.py`: headline document/template ranking models.
- `extra_trees_bad_tail_*.py`: tree-based downside-risk gates, rankers, and sizers.
- `always_long*.py`: long-biased baselines and headline/prior ablations.
- `robust_long_price_disagreement.py`: price/headline disagreement baseline.
- `template_strategy.py`: minimal skeleton for adding a new strategy.

## Running Any Strategy

Every strategy module in this folder, except utility modules such as
`model_risk_utils.py`, exposes `build_strategy()` and can be run with:

```bash
./venv/bin/python -m pipeline.runner --strategy-file pipeline/strategies/<strategy_file>.py
```

Strategies that depend on the generated price feature store require:

```bash
cd agents/features
../../venv/bin/python -m src.build_feature_store
cd ../..
```
