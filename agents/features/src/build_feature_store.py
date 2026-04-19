"""CLI for building the reusable price-only feature store."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features_price import build_test_set, build_train_set


def _default_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "hrt-eth-zurich-datathon-2026" / "data"


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "feature_store"


def _parse_args() -> argparse.Namespace:
    data_dir = _default_data_dir()

    parser = argparse.ArgumentParser(description="Build price-only session feature store from parquet bars.")
    parser.add_argument("--seen-train", type=Path, default=data_dir / "bars_seen_train.parquet")
    parser.add_argument("--unseen-train", type=Path, default=data_dir / "bars_unseen_train.parquet")
    parser.add_argument("--seen-public-test", type=Path, default=data_dir / "bars_seen_public_test.parquet")
    parser.add_argument("--seen-private-test", type=Path, default=data_dir / "bars_seen_private_test.parquet")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--expected-bars", type=int, default=50)
    parser.add_argument("--dct-coeffs", type=int, default=10)
    return parser.parse_args()


def _read_bars(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_parquet(path)


def _write_frame(df: pd.DataFrame, path: Path, output_format: str) -> None:
    if output_format == "parquet":
        df.to_parquet(path.with_suffix(".parquet"), index=True)
    else:
        df.to_csv(path.with_suffix(".csv"), index=True)


def _write_series(series: pd.Series, path: Path, output_format: str) -> None:
    frame = series.to_frame(name=series.name or "target_return")
    _write_frame(frame, path, output_format)


def main() -> None:
    args = _parse_args()

    seen_train = _read_bars(args.seen_train)
    unseen_train = _read_bars(args.unseen_train)
    seen_public = _read_bars(args.seen_public_test)
    seen_private = _read_bars(args.seen_private_test)

    x_train, y_train = build_train_set(
        seen_train,
        unseen_train,
        expected_bars=args.expected_bars,
        dct_coeffs=args.dct_coeffs,
    )
    x_public_test = build_test_set(seen_public, expected_bars=args.expected_bars, dct_coeffs=args.dct_coeffs)
    x_private_test = build_test_set(seen_private, expected_bars=args.expected_bars, dct_coeffs=args.dct_coeffs)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    _write_frame(x_train, args.output_dir / "X_train", args.output_format)
    _write_series(y_train, args.output_dir / "y_train", args.output_format)
    _write_frame(x_public_test, args.output_dir / "X_public_test", args.output_format)
    _write_frame(x_private_test, args.output_dir / "X_private_test", args.output_format)

    print("Feature store build complete.")
    print(f"output_dir={args.output_dir}")
    print(f"X_train shape={x_train.shape}")
    print(f"y_train shape={y_train.shape}")
    print(f"X_public_test shape={x_public_test.shape}")
    print(f"X_private_test shape={x_private_test.shape}")


if __name__ == "__main__":
    main()
