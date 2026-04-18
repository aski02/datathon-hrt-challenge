from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .types import PipelineContext, SplitInput


def load_bars(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"{name}.parquet").sort_values(["session", "bar_ix"]).reset_index(drop=True)


def load_headlines(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"{name}.parquet").sort_values(["session", "bar_ix"]).reset_index(drop=True)


def second_half_return(seen_bars: pd.DataFrame, unseen_bars: pd.DataFrame) -> pd.Series:
    seen_last = seen_bars.groupby("session", sort=True).last()["close"]
    unseen_last = unseen_bars.groupby("session", sort=True).last()["close"]
    return (unseen_last / seen_last - 1.0).rename("target_return")


def make_bar_features(bars: pd.DataFrame) -> pd.DataFrame:
    grouped = bars.groupby("session", sort=True)
    first = grouped.first()
    last = grouped.last()

    features = pd.DataFrame({"session": grouped.size().index})
    features["close_last_seen"] = features["session"].map(last["close"])
    features["close_mean_seen"] = features["session"].map(grouped["close"].mean())
    features["close_std_seen"] = features["session"].map(grouped["close"].std().fillna(0.0))
    features["ret_1"] = features["session"].map(grouped["close"].apply(lambda x: x.iloc[-1] / x.iloc[-2] - 1.0))
    features["ret_5"] = features["session"].map(grouped["close"].apply(lambda x: x.iloc[-1] / x.iloc[-6] - 1.0))
    features["ret_10"] = features["session"].map(grouped["close"].apply(lambda x: x.iloc[-1] / x.iloc[-11] - 1.0))
    features["ret_full_seen"] = features["session"].map(last["close"] / first["open"] - 1.0)
    features["range_full_seen"] = features["session"].map(grouped["high"].max() / grouped["low"].min() - 1.0)
    features["body_last"] = features["session"].map(last["close"] / last["open"] - 1.0)
    features["wick_up_last"] = features["session"].map(last["high"] / np.maximum(last["open"], last["close"]) - 1.0)
    features["wick_down_last"] = features["session"].map(np.minimum(last["open"], last["close"]) / last["low"] - 1.0)
    features["trend_slope"] = features["session"].map(
        grouped["close"].apply(lambda x: np.polyfit(np.arange(len(x)), x.to_numpy(), 1)[0])
    )
    return features


def make_headline_docs(headlines: pd.DataFrame, sessions: pd.Series) -> pd.Series:
    docs = headlines.groupby("session", sort=True)["headline"].apply(" || ".join)
    return sessions.map(docs).fillna("")


def build_feature_frame(seen_bars: pd.DataFrame, seen_headlines: pd.DataFrame) -> pd.DataFrame:
    features = make_bar_features(seen_bars)
    features["headline_text"] = make_headline_docs(seen_headlines, features["session"])
    return features.set_index("session", drop=False).sort_index()


def _build_split(name: str, bars: pd.DataFrame, headlines: pd.DataFrame) -> SplitInput:
    sessions = pd.Index(sorted(bars["session"].unique()), name="session")
    return SplitInput(
        name=name,
        bars=bars,
        headlines=headlines,
        sessions=sessions,
        features=build_feature_frame(bars, headlines).reindex(sessions),
    )


def build_context(data_dir: Path) -> PipelineContext:
    bars_seen_train = load_bars(data_dir, "bars_seen_train")
    bars_unseen_train = load_bars(data_dir, "bars_unseen_train")
    bars_seen_public = load_bars(data_dir, "bars_seen_public_test")
    bars_seen_private = load_bars(data_dir, "bars_seen_private_test")

    headlines_seen_train = load_headlines(data_dir, "headlines_seen_train")
    headlines_seen_public = load_headlines(data_dir, "headlines_seen_public_test")
    headlines_seen_private = load_headlines(data_dir, "headlines_seen_private_test")

    train = _build_split("train_seen", bars_seen_train, headlines_seen_train)
    public_test = _build_split("public_seen", bars_seen_public, headlines_seen_public)
    private_test = _build_split("private_seen", bars_seen_private, headlines_seen_private)

    return PipelineContext(
        train=train,
        public_test=public_test,
        private_test=private_test,
        train_target_return=second_half_return(bars_seen_train, bars_unseen_train).reindex(train.sessions),
    )
