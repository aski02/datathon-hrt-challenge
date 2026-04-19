# Submission Ensembling CLI

This document explains how to blend existing submission CSV files with
`pipeline/ensemble_submissions.py`.

The tool is useful when you already have two `session,target_position` outputs
and want to combine them into stronger ensemble candidates quickly.

## Input Contract

Each input CSV must contain:

- `session`
- `target_position`

Both files must have the same session set (typically 20,000 test sessions).

## Methods

The script supports 3 methods:

- `level_qmap`
  - Maps the second model to the anchor model's exposure distribution via ranks.
  - Then applies a convex level blend:
    - `out = w_anchor * anchor + (1 - w_anchor) * other_qmapped`
- `rank_perm`
  - Blends anchor/other ranks, then remaps to the anchor distribution.
  - Keeps anchor exposure distribution exactly (changes ordering only).
- `disagreement_guard`
  - Uses a dynamic other weight that shrinks when models disagree:
    - `w_other = max_w_other * (1 - |rank_a - rank_b|)^power`
  - Then rank-blends and remaps to anchor distribution.

## Single Blend Example

From repo root:

```bash
python -m pipeline.ensemble_submissions \
  --inputs agents/analyse_submissions/k25cap080-zero-weak-survivors-q10_all_test.csv \
           agents/analyse_submissions/subspace-bagged-downside-ranker_all_test.csv \
  --anchor-index 0 \
  --mode single \
  --method level_qmap \
  --w-anchor 0.70 \
  --output submission/blend_level_qmap_70_30_all_test.csv
```

## Auto Sweep Example

Generate many candidates at once:

```bash
python -m pipeline.ensemble_submissions \
  --inputs agents/analyse_submissions/k25cap080-zero-weak-survivors-q10_all_test.csv \
           agents/analyse_submissions/subspace-bagged-downside-ranker_all_test.csv \
  --anchor-index 0 \
  --mode auto \
  --prefix k25_subspace_auto \
  --output-dir agents/analyse_submissions
```

This writes:

- multiple `*_all_test.csv` candidates
- one diagnostics table:
  - `<prefix>_auto_diagnostics.csv`

## Useful Flags

- `--w-anchor`: anchor weight for `level_qmap` and `rank_perm`.
- `--max-w-other`, `--power`: parameters for `disagreement_guard`.
- `--auto-level-weights`: sweep list for `level_qmap`.
- `--auto-rank-weights`: sweep list for `rank_perm`.
- `--auto-guard-maxw`, `--auto-guard-power`: sweep lists for disagreement guard.

All sweep lists are comma-separated (for example `0.65,0.70,0.75`).

## Practical Guidance

- Use the stronger model as `--anchor-index`.
- Start with `level_qmap` around `w_anchor=0.65..0.80`.
- Try `disagreement_guard` when the secondary model is noisy but complementary.
- Use diagnostics to select 3-6 candidates for leaderboard testing.

## Best Performing Settings (Shortlist)

Based on recent leaderboard results for
`k25cap080-zero-weak-survivors-q10_all_test.csv` (anchor) +
`subspace-bagged-downside-ranker_all_test.csv` (other):

1. `level_qmap`, `w_anchor=0.70` (70/30): **2.63103** (best)
2. `disagreement_guard`, `max_w_other=0.45`, `power=1.7`: **2.61913**
3. `sparse_topdecile_overlay`: **2.60290** (not implemented in this CLI)

For this CLI, the highest-priority settings to test are:

1. `level_qmap` with `w_anchor=0.70`
2. `disagreement_guard` with `max_w_other=0.45`, `power=1.7`
3. `level_qmap` with `w_anchor=0.72` (small perturbation around the best point)

### Copy/Paste: 3 Recommended Runs

Run these from repo root:

```bash
python -m pipeline.ensemble_submissions \
  --inputs agents/analyse_submissions/k25cap080-zero-weak-survivors-q10_all_test.csv \
           agents/analyse_submissions/subspace-bagged-downside-ranker_all_test.csv \
  --anchor-index 0 \
  --mode single \
  --method level_qmap \
  --w-anchor 0.70 \
  --output agents/analyse_submissions/shortlist_level_qmap_70_30_all_test.csv
```

```bash
python -m pipeline.ensemble_submissions \
  --inputs agents/analyse_submissions/k25cap080-zero-weak-survivors-q10_all_test.csv \
           agents/analyse_submissions/subspace-bagged-downside-ranker_all_test.csv \
  --anchor-index 0 \
  --mode single \
  --method disagreement_guard \
  --max-w-other 0.45 \
  --power 1.7 \
  --output agents/analyse_submissions/shortlist_disagreement_guard_all_test.csv
```

```bash
python -m pipeline.ensemble_submissions \
  --inputs agents/analyse_submissions/k25cap080-zero-weak-survivors-q10_all_test.csv \
           agents/analyse_submissions/subspace-bagged-downside-ranker_all_test.csv \
  --anchor-index 0 \
  --mode single \
  --method level_qmap \
  --w-anchor 0.72 \
  --output agents/analyse_submissions/shortlist_level_qmap_72_28_all_test.csv
```
