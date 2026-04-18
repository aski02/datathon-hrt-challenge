from __future__ import annotations

import argparse
import importlib.util
import inspect
import re
from pathlib import Path
from types import ModuleType
from typing import Callable

import pandas as pd

from .data import build_context
from .strategy import Strategy, coerce_positions, sharpe_from_positions
from .types import PipelineContext, SplitInput

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "hrt-eth-zurich-datathon-2026" / "data"
DEFAULT_SUBMISSIONS_DIR = ROOT / "submission"


class FunctionStrategy:
    def __init__(self, name: str, fn: Callable[[SplitInput], object]) -> None:
        _validate_predict_signature(fn, owner=name)
        self.name = name
        self._fn = fn

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        # Stateless function strategies do not train.
        _ = train_split, train_target_return

    def predict(self, split: SplitInput) -> object:
        return self._fn(split)


def _load_module_from_path(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"pipeline_strategy_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load strategy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")


def _positional_parts(callable_obj: Callable[..., object]) -> tuple[list[inspect.Parameter], list[inspect.Parameter]]:
    signature = inspect.signature(callable_obj)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required_positional = [parameter for parameter in positional if parameter.default is inspect.Parameter.empty]
    return positional, required_positional


def _validate_predict_signature(callable_obj: Callable[..., object], owner: str) -> None:
    positional, required = _positional_parts(callable_obj)
    if len(positional) != 1 or len(required) != 1:
        raise TypeError(
            f"`{owner}` predict signature must be exactly `predict(split)` (one required positional argument)."
        )


def _validate_fit_signature(callable_obj: Callable[..., object], owner: str) -> None:
    positional, required = _positional_parts(callable_obj)
    if len(positional) != 2 or not (1 <= len(required) <= 2):
        raise TypeError(
            f"`{owner}` fit signature must be `fit(train_split, train_target_return)`."
        )


def _fit_strategy_once(strategy: Strategy, context: PipelineContext) -> int:
    fit_calls = 1
    fit = getattr(strategy, "fit", None)
    if fit is None:
        return fit_calls
    if not callable(fit):
        raise TypeError("Strategy `fit` attribute exists but is not callable.")
    _validate_fit_signature(fit, owner=str(getattr(strategy, "name", strategy.__class__.__name__)))
    fit(context.train, context.train_target_return)
    return fit_calls


def _assert_fit_called_once(fit_calls: int) -> None:
    if fit_calls != 1:
        raise RuntimeError("Internal error: strategy must be fit exactly once before test predictions.")


def _instantiate_strategy(module: ModuleType, path: Path, symbol: str) -> Strategy:
    if hasattr(module, symbol):
        obj = getattr(module, symbol)
        if inspect.isclass(obj):
            strategy = obj()
        elif callable(obj):
            _, required_positional = _positional_parts(obj)
            if len(required_positional) >= 1:
                strategy = FunctionStrategy(path.stem, obj)
            else:
                maybe = obj()
                strategy = maybe if hasattr(maybe, "predict") else FunctionStrategy(path.stem, obj)
        elif hasattr(obj, "predict"):
            strategy = obj
        else:
            raise TypeError(f"`{symbol}` exists but is not a strategy/class/function: {type(obj)}")
    elif hasattr(module, "strategy"):
        strategy = getattr(module, "strategy")
    elif hasattr(module, "predict") and callable(getattr(module, "predict")):
        strategy = FunctionStrategy(path.stem, getattr(module, "predict"))
    else:
        raise AttributeError(
            f"Could not find strategy entrypoint. Expected `{symbol}()` or `strategy` object or `predict(split)`."
        )

    if not hasattr(strategy, "predict"):
        raise TypeError("Strategy object must expose a `predict(split)` method.")
    _validate_predict_signature(
        getattr(strategy, "predict"),
        owner=str(getattr(strategy, "name", strategy.__class__.__name__)),
    )
    if not hasattr(strategy, "name"):
        strategy.name = path.stem  # type: ignore[attr-defined]
    return strategy  # type: ignore[return-value]


def _build_submission_frame(context: PipelineContext, public_positions: pd.Series, private_positions: pd.Series) -> pd.DataFrame:
    public_frame = pd.DataFrame({"session": context.public_test.sessions.to_numpy(), "target_position": public_positions.to_numpy()})
    private_frame = pd.DataFrame(
        {"session": context.private_test.sessions.to_numpy(), "target_position": private_positions.to_numpy()}
    )
    submission = pd.concat([public_frame, private_frame], ignore_index=True).sort_values("session").reset_index(drop=True)

    expected_sessions = set(context.public_test.sessions.tolist()) | set(context.private_test.sessions.tolist())
    submission_sessions = set(submission["session"].tolist())
    if submission_sessions != expected_sessions:
        missing = len(expected_sessions - submission_sessions)
        extra = len(submission_sessions - expected_sessions)
        raise ValueError(f"Submission session mismatch (missing={missing}, extra={extra}).")
    if submission.duplicated("session").any():
        duplicates = int(submission.duplicated("session").sum())
        raise ValueError(f"Submission has duplicated sessions (duplicates={duplicates}).")
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one strategy file and produce HRT submission CSVs.")
    parser.add_argument(
        "--strategy-file",
        type=Path,
        required=True,
        help="Path to a single strategy Python file.",
    )
    parser.add_argument(
        "--entrypoint",
        default="build_strategy",
        help="Strategy symbol to resolve in the module. Default: build_strategy",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing challenge parquet files. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SUBMISSIONS_DIR,
        help=f"Directory for generated CSVs. Default: {DEFAULT_SUBMISSIONS_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output filename. If omitted, uses <strategy_name>_all_test.csv",
    )
    parser.add_argument(
        "--write-split-files",
        action="store_true",
        help="Also write <name>_public.csv and <name>_private.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = build_context(args.data_dir)
    module = _load_module_from_path(args.strategy_file)
    strategy = _instantiate_strategy(module, args.strategy_file, args.entrypoint)
    strategy_name = _sanitize_name(str(getattr(strategy, "name", args.strategy_file.stem)))

    fit_calls = _fit_strategy_once(strategy, context)
    _assert_fit_called_once(fit_calls)
    public_positions = coerce_positions(strategy.predict(context.public_test), context.public_test.sessions)
    _assert_fit_called_once(fit_calls)
    private_positions = coerce_positions(strategy.predict(context.private_test), context.private_test.sessions)

    submission = _build_submission_frame(context, public_positions, private_positions)
    train_positions = coerce_positions(strategy.predict(context.train), context.train.sessions)
    train_sharpe = sharpe_from_positions(train_positions, context.train_target_return)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{strategy_name}_all_test.csv"
    output_path = args.output_dir / output_name
    submission.to_csv(output_path, index=False)

    print(f"strategy: {strategy_name}")
    print(f"train_sharpe: {train_sharpe:.4f}")
    print(f"train_position_mean/std: {train_positions.mean():.6f} / {train_positions.std(ddof=0):.6f}")
    print(f"public_position_mean/std: {public_positions.mean():.6f} / {public_positions.std(ddof=0):.6f}")
    print(f"private_position_mean/std: {private_positions.mean():.6f} / {private_positions.std(ddof=0):.6f}")
    print(f"wrote: {output_path}")
    print(f"rows: {len(submission)} | unique_sessions: {submission['session'].nunique()}")

    if args.write_split_files:
        public_path = args.output_dir / f"{strategy_name}_public.csv"
        private_path = args.output_dir / f"{strategy_name}_private.csv"
        pd.DataFrame(
            {"session": context.public_test.sessions.to_numpy(), "target_position": public_positions.to_numpy()}
        ).to_csv(public_path, index=False)
        pd.DataFrame(
            {"session": context.private_test.sessions.to_numpy(), "target_position": private_positions.to_numpy()}
        ).to_csv(private_path, index=False)
        print(f"wrote: {public_path}")
        print(f"wrote: {private_path}")


if __name__ == "__main__":
    main()
