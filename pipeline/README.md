# Strategy Pipeline

This folder provides a clean workflow for testing and exporting strategies.

## Chosen Submission Model

The current submission model is:

- `pipeline/strategies/subspace_btp_hdoc_ensemble.py`

It ensembles:

- `subspace_bagged_downside_ranker`
- `btp-rank-hdoc`

To run it from repo root:

```bash
./venv/bin/python -m pipeline.runner \
  --strategy-file pipeline/strategies/subspace_btp_hdoc_ensemble.py \
  --output-name subspace-btp-hdoc-ensemble_all_test.csv
```

If the subspace feature store is missing, build it first:

```bash
cd agents/features
../../venv/bin/python -m src.build_feature_store
cd ../..
```

## Goal

For a new strategy, you only need **one Python file** with a `predict(split)` implementation.
Then run one command to generate the final submission CSV.

## Setup (Generic)

Run this from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

1. Copy `pipeline/strategies/template_strategy.py` and rename it, e.g. `pipeline/strategies/my_strategy.py`.
2. Edit your logic in the `predict` method.
3. Run from repo root:

```bash
python -m pipeline.runner --strategy-file pipeline/strategies/my_strategy.py
```

This writes:
- `submission/<strategy_name>_all_test.csv`

The file is validated to contain:
- columns: `session,target_position`
- one unique row per test session
- total rows = public + private test sessions (currently 20,000)
- finite numeric `target_position` values only (no `NaN`/`inf`)

## Strategy API

Supported strategy entry styles:
- `build_strategy()` (default entrypoint) returning an object with `predict(...)`
- module-level `predict(split)` function (no class required)
- custom symbol via `--entrypoint <symbol>`

All styles resolve to:

```python
def predict(self, split) -> pd.Series | pd.DataFrame | np.ndarray
```

Optional training hook:

```python
def fit(self, train_split, train_target_return) -> None
```

Where:
- `split.name` is one of `train_seen`, `public_seen`, `private_seen`
- `split.sessions` is the target session index for that split
- `split.features` contains precomputed session-level features:
  - numeric bar features (`ret_*`, `range_full_seen`, etc.)
  - `headline_text` concatenation for text models
- `fit(...)` receives `train_target_return` derived from `unseen_train` for supervised fitting

Leakage guardrails:
- Runner calls `fit(...)` exactly once before any test prediction.
- `predict(...)` receives only split-local data, not global train/public/private context.

Output options from `predict`:
- `pd.Series` indexed by session
- `pd.DataFrame` with `session,target_position`
- array/list with one value per split session

## CLI Options

```bash
python -m pipeline.runner --help
```

Useful flags:
- `--entrypoint build_strategy` (default)
- `--output-name custom.csv`
- `--write-split-files` to also emit `*_public.csv` and `*_private.csv`

## Included Examples

- `pipeline/strategies/always_long.py`
- `pipeline/strategies/template_strategy.py`

## Submission Ensembling

To blend existing submission CSV files (for example model A + model B), use:

```bash
python -m pipeline.ensemble_submissions --help
```

Detailed instructions and examples are in:

- `pipeline/README_ENSEMBLING.md`
