# Submission Ensembling CLI

`pipeline.ensemble_submissions` blends two existing `session,target_position`
CSV files. It is useful for local experimentation after two complete submissions
have already been generated.

Generated ensemble CSVs and diagnostics are ignored by git.

## Input Contract

Each input CSV must contain:

- `session`
- `target_position`

Both files must contain the same session set. The loader sorts by `session`,
rejects duplicate sessions, and rejects non-finite positions.

## Methods

- `level_qmap`: maps the second model to the anchor model's exposure
  distribution by rank, then applies a convex level blend.
- `rank_perm`: blends ranks, then remaps to the anchor exposure distribution.
- `disagreement_guard`: rank-blends with a dynamic secondary weight that shrinks
  when the two models disagree.

## Single Blend Example

```bash
./venv/bin/python -m pipeline.ensemble_submissions \
  --inputs submission/model_a_all_test.csv submission/model_b_all_test.csv \
  --anchor-index 0 \
  --mode single \
  --method level_qmap \
  --w-anchor 0.70 \
  --output submission/model_a_model_b_level_qmap_all_test.csv
```

## Auto Sweep Example

```bash
./venv/bin/python -m pipeline.ensemble_submissions \
  --inputs submission/model_a_all_test.csv submission/model_b_all_test.csv \
  --anchor-index 0 \
  --mode auto \
  --prefix model_a_model_b \
  --output-dir submission
```

This writes multiple candidate CSVs plus one diagnostics table:

```text
submission/model_a_model_b_auto_diagnostics.csv
```

## Useful Flags

- `--w-anchor`: anchor weight for `level_qmap` and `rank_perm`.
- `--max-w-other`, `--power`: parameters for `disagreement_guard`.
- `--auto-level-weights`: comma-separated sweep list for `level_qmap`.
- `--auto-rank-weights`: comma-separated sweep list for `rank_perm`.
- `--auto-guard-maxw`, `--auto-guard-power`: comma-separated sweep lists for
  disagreement guard.

Use diagnostics for sanity checks on distribution, correlation, and average
absolute deviation before deciding whether a generated ensemble is worth testing.
