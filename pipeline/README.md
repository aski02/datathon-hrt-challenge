# Strategy Pipeline

`pipeline/` is the supported package for loading challenge data, fitting one
strategy, validating positions, and writing a submission CSV.

## Final Strategy

```bash
./venv/bin/python -m pipeline.runner \
  --strategy-file pipeline/strategies/subspace_btp_hdoc_ensemble.py \
  --output-name subspace-btp-hdoc-ensemble_all_test.csv
```

If the feature store is missing, build it first:

```bash
cd agents/features
../../venv/bin/python -m src.build_feature_store
cd ../..
```

## Strategy Contract

A strategy file may expose any of these entrypoints:

- `build_strategy()` returning an object with `predict(split)`
- a module-level `strategy` object with `predict(split)`
- a module-level `predict(split)` function
- a custom symbol selected with `--entrypoint`

The supported object API is:

```python
def fit(self, train_split, train_target_return) -> None:
    ...

def predict(self, split) -> pd.Series | pd.DataFrame | np.ndarray:
    ...
```

`fit` is optional. The runner calls it once before test prediction.

`split` contains:

- `name`: `train_seen`, `public_seen`, or `private_seen`
- `sessions`: target session index
- `bars`: split-local seen OHLC bars
- `headlines`: split-local seen headlines
- `features`: deterministic session-level features built from seen data only

`predict` may return:

- a `pd.Series` indexed by session
- a `pd.DataFrame` with `session,target_position`
- a one-dimensional array/list with one value per split session

The runner validates missing sessions, duplicate sessions, and non-finite target
positions before writing output.

## Strategy Catalog

`pipeline/strategies/` intentionally contains both the final submission model
and alternate hackathon strategies. Any strategy file exposing `build_strategy`
can be run through the same runner:

```bash
./venv/bin/python -m pipeline.runner --strategy-file pipeline/strategies/<strategy_file>.py
```

Key files:

- `subspace_btp_hdoc_ensemble.py`: final submitted ensemble.
- `subspace_bagged_downside_ranker.py`: price feature-store ranker used by the final ensemble.
- `btp_rank_hdoc.py` and `btp_rank_tpl.py`: headline-template/document rankers.
- `extra_trees_bad_tail_*`: tree-based downside-risk variants.
- `always_long*.py` and `robust_long_price_disagreement.py`: simpler baselines and ablations.
- `template_strategy.py`: copyable strategy skeleton.

See `pipeline/strategies/README.md` for the catalog.

## Submission Ensembling

To blend two existing submission CSVs:

```bash
./venv/bin/python -m pipeline.ensemble_submissions --help
```

See `pipeline/README_ENSEMBLING.md` for supported blend modes.
