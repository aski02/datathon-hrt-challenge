from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "submission"


@dataclass(frozen=True)
class SubmissionData:
    name: str
    sessions: np.ndarray
    positions: np.ndarray
    source_path: Path


def _safe_float_token(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _parse_float_list(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected at least one float value.")
    return [float(part) for part in parts]


def _load_submission(path: Path) -> SubmissionData:
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")
    frame = pd.read_csv(path)
    required_columns = {"session", "target_position"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"{path} must contain columns: session,target_position")

    frame = frame.loc[:, ["session", "target_position"]].sort_values("session").reset_index(drop=True)
    if frame["session"].duplicated().any():
        duplicates = int(frame["session"].duplicated().sum())
        raise ValueError(f"{path} has duplicated sessions ({duplicates}).")
    if not np.isfinite(frame["target_position"].to_numpy()).all():
        raise ValueError(f"{path} contains NaN/inf target_position values.")

    return SubmissionData(
        name=path.stem,
        sessions=frame["session"].to_numpy(dtype=np.int64),
        positions=frame["target_position"].to_numpy(dtype=float),
        source_path=path,
    )


def _align_submissions(first: SubmissionData, second: SubmissionData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if first.sessions.shape != second.sessions.shape or not np.array_equal(first.sessions, second.sessions):
        raise ValueError("Input submissions do not contain identical session sets/order.")
    return first.sessions, first.positions, second.positions


def _rank01(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=float)
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    return (ranks - 1.0) / (len(values) - 1.0)


def _perm_map_to_reference(score: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    order = np.argsort(score, kind="mergesort")
    mapped = np.empty_like(reference_values, dtype=float)
    mapped[order] = np.sort(reference_values)
    return mapped


def _qmap_to_reference(raw_values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    return _perm_map_to_reference(_rank01(raw_values), reference_values)


def blend_level_qmap(anchor: np.ndarray, other: np.ndarray, w_anchor: float) -> np.ndarray:
    other_qmapped = _qmap_to_reference(other, anchor)
    return w_anchor * anchor + (1.0 - w_anchor) * other_qmapped


def blend_rank_perm(anchor: np.ndarray, other: np.ndarray, w_anchor: float) -> np.ndarray:
    score = w_anchor * _rank01(anchor) + (1.0 - w_anchor) * _rank01(other)
    return _perm_map_to_reference(score, anchor)


def blend_disagreement_guard(anchor: np.ndarray, other: np.ndarray, max_w_other: float, power: float) -> np.ndarray:
    rank_anchor = _rank01(anchor)
    rank_other = _rank01(other)
    disagreement = np.abs(rank_anchor - rank_other)
    w_other = max_w_other * np.power(1.0 - disagreement, power)
    score = (1.0 - w_other) * rank_anchor + w_other * rank_other
    return _perm_map_to_reference(score, anchor)


def _compute_diagnostics(
    name: str, positions: np.ndarray, anchor: np.ndarray, other: np.ndarray
) -> dict[str, float | str]:
    return {
        "name": name,
        "mean": float(np.mean(positions)),
        "std": float(np.std(positions, ddof=0)),
        "min": float(np.min(positions)),
        "max": float(np.max(positions)),
        "corr_to_anchor": float(np.corrcoef(positions, anchor)[0, 1]),
        "corr_to_other": float(np.corrcoef(positions, other)[0, 1]),
        "mae_to_anchor": float(np.mean(np.abs(positions - anchor))),
        "mae_to_other": float(np.mean(np.abs(positions - other))),
        "sign_disagree_vs_anchor": float(np.mean(np.sign(positions) != np.sign(anchor))),
    }


def _write_submission(output_path: Path, sessions: np.ndarray, positions: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"session": sessions, "target_position": positions}).to_csv(output_path, index=False)


def _validate_weights(w_anchor: float) -> None:
    if not 0.0 <= w_anchor <= 1.0:
        raise ValueError(f"w_anchor must be in [0,1], got {w_anchor}")


def _validate_guard_params(max_w_other: float, power: float) -> None:
    if not 0.0 <= max_w_other <= 1.0:
        raise ValueError(f"max_w_other must be in [0,1], got {max_w_other}")
    if power <= 0.0:
        raise ValueError(f"power must be > 0, got {power}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blend two existing submission CSV files into one or many ensembles.")
    parser.add_argument("--inputs", nargs=2, type=Path, required=True, help="Two input submission CSV paths.")
    parser.add_argument(
        "--anchor-index",
        type=int,
        choices=[0, 1],
        default=0,
        help="Which --inputs file is the anchor model (default: 0).",
    )
    parser.add_argument(
        "--mode", choices=["single", "auto"], default="single", help="single = one output, auto = sweep."
    )
    parser.add_argument(
        "--method",
        choices=["level_qmap", "rank_perm", "disagreement_guard"],
        default="level_qmap",
        help="Blend method for mode=single.",
    )
    parser.add_argument(
        "--w-anchor",
        type=float,
        default=0.70,
        help="Anchor weight for level_qmap / rank_perm in mode=single.",
    )
    parser.add_argument(
        "--max-w-other",
        type=float,
        default=0.45,
        help="Maximum dynamic other weight for disagreement_guard in mode=single.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=1.7,
        help="Disagreement exponent for disagreement_guard in mode=single.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output file for mode=single.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--prefix", default="ensemble", help="Filename prefix for generated outputs.")
    parser.add_argument(
        "--auto-level-weights",
        default="0.65,0.68,0.70,0.72,0.75",
        help="Comma-separated anchor weights for level_qmap in mode=auto.",
    )
    parser.add_argument(
        "--auto-rank-weights",
        default="0.62,0.66,0.70,0.74",
        help="Comma-separated anchor weights for rank_perm in mode=auto.",
    )
    parser.add_argument(
        "--auto-guard-maxw",
        default="0.35,0.45,0.55",
        help="Comma-separated max_w_other values for disagreement_guard in mode=auto.",
    )
    parser.add_argument(
        "--auto-guard-power",
        default="1.5,1.7,2.0",
        help="Comma-separated power values for disagreement_guard in mode=auto.",
    )
    return parser


def _resolve_anchor(inputs: list[SubmissionData], anchor_index: int) -> tuple[SubmissionData, SubmissionData]:
    anchor = inputs[anchor_index]
    other = inputs[1 - anchor_index]
    return anchor, other


def _single_mode(args: argparse.Namespace, sessions: np.ndarray, anchor: np.ndarray, other: np.ndarray) -> None:
    _validate_weights(args.w_anchor)
    _validate_guard_params(args.max_w_other, args.power)

    method_to_fn: dict[str, Callable[[], np.ndarray]] = {
        "level_qmap": lambda: blend_level_qmap(anchor, other, args.w_anchor),
        "rank_perm": lambda: blend_rank_perm(anchor, other, args.w_anchor),
        "disagreement_guard": lambda: blend_disagreement_guard(anchor, other, args.max_w_other, args.power),
    }
    positions = method_to_fn[args.method]()

    if args.output is None:
        method_suffix = args.method
        if args.method in {"level_qmap", "rank_perm"}:
            method_suffix += f"_wa{_safe_float_token(args.w_anchor)}"
        elif args.method == "disagreement_guard":
            method_suffix += f"_mw{_safe_float_token(args.max_w_other)}_p{_safe_float_token(args.power)}"
        output_path = args.output_dir / f"{args.prefix}_{method_suffix}_all_test.csv"
    else:
        output_path = args.output

    _write_submission(output_path, sessions, positions)
    diagnostics = pd.DataFrame([_compute_diagnostics(output_path.stem, positions, anchor, other)])
    diagnostics_path = output_path.with_name(f"{output_path.stem}_diagnostics.csv")
    diagnostics.to_csv(diagnostics_path, index=False)

    print(f"wrote: {output_path}")
    print(f"wrote: {diagnostics_path}")
    print(diagnostics.to_string(index=False))


def _auto_mode(args: argparse.Namespace, sessions: np.ndarray, anchor: np.ndarray, other: np.ndarray) -> None:
    level_weights = _parse_float_list(args.auto_level_weights)
    rank_weights = _parse_float_list(args.auto_rank_weights)
    guard_maxw = _parse_float_list(args.auto_guard_maxw)
    guard_power = _parse_float_list(args.auto_guard_power)

    for weight in level_weights + rank_weights:
        _validate_weights(weight)
    for max_w in guard_maxw:
        for power in guard_power:
            _validate_guard_params(max_w, power)

    outputs: list[tuple[str, Path, np.ndarray]] = []

    for w_anchor in level_weights:
        name = f"{args.prefix}_level_qmap_wa{_safe_float_token(w_anchor)}"
        positions = blend_level_qmap(anchor, other, w_anchor)
        path = args.output_dir / f"{name}_all_test.csv"
        _write_submission(path, sessions, positions)
        outputs.append((name, path, positions))

    for w_anchor in rank_weights:
        name = f"{args.prefix}_rank_perm_wa{_safe_float_token(w_anchor)}"
        positions = blend_rank_perm(anchor, other, w_anchor)
        path = args.output_dir / f"{name}_all_test.csv"
        _write_submission(path, sessions, positions)
        outputs.append((name, path, positions))

    for max_w_other in guard_maxw:
        for power in guard_power:
            name = (
                f"{args.prefix}_disagreement_guard_mw{_safe_float_token(max_w_other)}" f"_p{_safe_float_token(power)}"
            )
            positions = blend_disagreement_guard(anchor, other, max_w_other=max_w_other, power=power)
            path = args.output_dir / f"{name}_all_test.csv"
            _write_submission(path, sessions, positions)
            outputs.append((name, path, positions))

    rows = [_compute_diagnostics(name, positions, anchor, other) for name, _, positions in outputs]
    diagnostics = pd.DataFrame(rows).sort_values(["corr_to_other", "mae_to_anchor"], ascending=[False, True])
    diagnostics_path = args.output_dir / f"{args.prefix}_auto_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    print(f"generated_files: {len(outputs)}")
    for name, path, _ in outputs:
        print(f"- {name}: {path}")
    print(f"wrote: {diagnostics_path}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    loaded = [_load_submission(path) for path in args.inputs]
    anchor_submission, other_submission = _resolve_anchor(loaded, args.anchor_index)
    sessions, anchor_positions, other_positions = _align_submissions(anchor_submission, other_submission)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"anchor: {anchor_submission.source_path}")
    print(f"other: {other_submission.source_path}")
    print(f"sessions: {len(sessions)}")

    if args.mode == "single":
        _single_mode(args, sessions=sessions, anchor=anchor_positions, other=other_positions)
    else:
        _auto_mode(args, sessions=sessions, anchor=anchor_positions, other=other_positions)


if __name__ == "__main__":
    main()
