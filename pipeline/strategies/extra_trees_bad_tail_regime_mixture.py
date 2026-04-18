from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.strategies.extra_trees_bad_tail_probability_sizer import (
    ExtraTreesBadTailProbabilitySizerStrategy,
)
from pipeline.strategies.extra_trees_bad_tail_probability_sizer_catalog_lite import (
    ExtraTreesBadTailProbabilitySizerCatalogLiteStrategy,
)
from pipeline.strategies.model_risk_utils import build_risk_features
from pipeline.types import SplitInput


class ExtraTreesBadTailRegimeMixtureStrategy(ExtraTreesBadTailProbabilitySizerCatalogLiteStrategy):
    name = "extra-trees-bad-tail-regime-mixture"

    def __init__(self) -> None:
        super().__init__()
        self.lookup_shrink: float = 25.0
        self.w_stretched: float = 0.12
        self.w_disagreement: float = 0.12
        self.w_ordinary: float = 0.08

        self._ret_hi_q: float = 0.0
        self._close_hi_q: float = 0.0
        self._disagree_hi_q: float = 0.0
        self._ret_lo_mid_q: float = 0.0
        self._ret_hi_mid_q: float = 0.0
        self._trend_lo_mid_q: float = 0.0
        self._trend_hi_mid_q: float = 0.0

        self._lookups: dict[str, dict[str, dict[str, float]]] = {}
        self._defaults: dict[str, float] = {}
        self._alpha_mu: dict[str, float] = {}
        self._alpha_sd: dict[str, float] = {}

    def _build_X(self, split) -> pd.DataFrame:
        return build_risk_features(split)

    @staticmethod
    def _safe_std(x: np.ndarray) -> float:
        std = float(np.std(np.asarray(x, dtype=float), ddof=0))
        return std if std > 1e-12 else 1.0

    @staticmethod
    def _fit_lookup(keys: pd.Series, values: pd.Series, shrink: float, default: float) -> dict[str, float]:
        frame = pd.DataFrame({"k": keys.astype(str), "y": values.astype(float)})
        agg = frame.groupby("k", sort=False)["y"].agg(["mean", "count"])
        weight = agg["count"] / (agg["count"] + float(shrink))
        blended = default + weight * (agg["mean"] - default)
        return {str(k): float(v) for k, v in blended.items()}

    @staticmethod
    def _apply_lookup(keys: pd.Series, lookup: dict[str, float], default: float) -> np.ndarray:
        return np.array([lookup.get(str(k), default) for k in keys.astype(str)], dtype=float)

    def _headline_state_frame(self, split: SplitInput) -> pd.DataFrame:
        rows = self._parse_headlines(split)
        sessions = split.sessions
        out = pd.DataFrame(index=sessions)
        out["first_company"] = "__NA__"
        out["last_intent"] = "__NA__"
        out["last_super_family"] = "__NA__"
        out["prior_disagreement_all"] = 0.0
        out["prior_balance_recent3"] = 0.0

        if rows.empty:
            return out

        first_rows = rows.groupby("session", sort=False).head(1).set_index("session")
        last_rows = rows.groupby("session", sort=False).tail(1).set_index("session")
        recent3_rows = rows.groupby("session", sort=False).tail(3).copy()

        g = rows.groupby("session", sort=False)
        prior_count = g.size().reindex(sessions, fill_value=0).astype(float)
        sign_counts = rows.assign(
            pos=(rows["prior_sign"] > 0).astype(int),
            neg=(rows["prior_sign"] < 0).astype(int),
        ).groupby("session")[["pos", "neg"]].sum().reindex(sessions, fill_value=0)
        disagreement = np.minimum(sign_counts["pos"], sign_counts["neg"]) / prior_count.clip(lower=1.0)

        recent3_g = recent3_rows.groupby("session", sort=False)
        recent3_count = recent3_g.size().reindex(sessions, fill_value=0).astype(float)
        recent3_sum = recent3_g["prior_sign"].sum().reindex(sessions, fill_value=0.0).astype(float)
        recent3_balance = recent3_sum / np.sqrt(recent3_count.clip(lower=1.0))

        out["first_company"] = first_rows["company"].reindex(sessions).fillna("__NA__").astype(str)
        out["last_intent"] = last_rows["intent"].reindex(sessions).fillna("__NA__").astype(str)
        out["last_super_family"] = last_rows["super_family"].reindex(sessions).fillna("__NA__").astype(str)
        out["prior_disagreement_all"] = disagreement.astype(float)
        out["prior_balance_recent3"] = recent3_balance.astype(float)
        return out

    def _regime_masks(self, risk_x: pd.DataFrame, state_x: pd.DataFrame) -> dict[str, np.ndarray]:
        ret = risk_x["ret_full_seen"].to_numpy(dtype=float)
        close_pos = risk_x["close_position_in_seen_range"].to_numpy(dtype=float)
        trend = risk_x["trend_slope"].to_numpy(dtype=float)
        disagree = state_x["prior_disagreement_all"].to_numpy(dtype=float)

        high_disagreement = disagree >= self._disagree_hi_q
        stretched = (ret >= self._ret_hi_q) | (close_pos >= self._close_hi_q)
        ordinary = (
            (ret >= self._ret_lo_mid_q)
            & (ret <= self._ret_hi_mid_q)
            & (trend >= self._trend_lo_mid_q)
            & (trend <= self._trend_hi_mid_q)
        )

        masks: dict[str, np.ndarray] = {}
        masks["disagreement"] = high_disagreement
        masks["stretched"] = stretched & ~masks["disagreement"]
        masks["ordinary"] = ordinary & ~masks["disagreement"] & ~masks["stretched"]
        return masks

    def _fit_regime_state(self, risk_x: pd.DataFrame, state_x: pd.DataFrame, y: pd.Series) -> None:
        self._ret_hi_q = float(risk_x["ret_full_seen"].quantile(0.80))
        self._close_hi_q = float(risk_x["close_position_in_seen_range"].quantile(0.80))
        self._disagree_hi_q = float(state_x["prior_disagreement_all"].quantile(0.80))
        self._ret_lo_mid_q = float(risk_x["ret_full_seen"].quantile(0.30))
        self._ret_hi_mid_q = float(risk_x["ret_full_seen"].quantile(0.70))
        self._trend_lo_mid_q = float(risk_x["trend_slope"].quantile(0.30))
        self._trend_hi_mid_q = float(risk_x["trend_slope"].quantile(0.70))

        masks = self._regime_masks(risk_x, state_x)
        self._lookups = {}
        self._defaults = {}
        self._alpha_mu = {}
        self._alpha_sd = {}

        for regime in ("disagreement", "stretched", "ordinary"):
            mask = masks[regime]
            y_reg = y.iloc[mask]
            state_reg = state_x.iloc[mask]
            default = float(y_reg.mean()) if len(y_reg) else float(y.mean())
            self._defaults[regime] = default
            self._lookups[regime] = {}

            if len(state_reg) == 0:
                self._lookups[regime]["first_company"] = {}
                self._lookups[regime]["last_intent"] = {}
                self._lookups[regime]["last_super_family"] = {}
                self._alpha_mu[regime] = 0.0
                self._alpha_sd[regime] = 1.0
                continue

            self._lookups[regime]["first_company"] = self._fit_lookup(
                state_reg["first_company"], y_reg, shrink=self.lookup_shrink, default=default
            )
            self._lookups[regime]["last_intent"] = self._fit_lookup(
                state_reg["last_intent"], y_reg, shrink=self.lookup_shrink, default=default
            )
            self._lookups[regime]["last_super_family"] = self._fit_lookup(
                state_reg["last_super_family"], y_reg, shrink=self.lookup_shrink, default=default
            )

            alpha = self._regime_alpha(regime, state_reg)
            self._alpha_mu[regime] = float(alpha.mean()) if len(alpha) else 0.0
            self._alpha_sd[regime] = self._safe_std(alpha) if len(alpha) else 1.0

    def _regime_alpha(self, regime: str, state_x: pd.DataFrame) -> np.ndarray:
        default = self._defaults[regime]
        lookups = self._lookups[regime]
        company = self._apply_lookup(state_x["first_company"], lookups["first_company"], default)
        intent = self._apply_lookup(state_x["last_intent"], lookups["last_intent"], default)
        sf = self._apply_lookup(state_x["last_super_family"], lookups["last_super_family"], default)
        recent_balance = state_x["prior_balance_recent3"].to_numpy(dtype=float)

        if regime == "stretched":
            return 0.55 * company + 0.25 * intent + 0.20 * sf
        if regime == "disagreement":
            return 0.15 * company + 0.40 * intent + 0.30 * sf + 0.15 * recent_balance
        return 0.20 * company + 0.40 * intent + 0.40 * sf

    def _regime_adjustment(self, risk_x: pd.DataFrame, state_x: pd.DataFrame) -> np.ndarray:
        masks = self._regime_masks(risk_x, state_x)
        adj = np.zeros(len(state_x), dtype=float)
        weights = {
            "disagreement": self.w_disagreement,
            "stretched": self.w_stretched,
            "ordinary": self.w_ordinary,
        }

        for regime, mask in masks.items():
            if not np.any(mask):
                continue
            alpha = self._regime_alpha(regime, state_x)
            alpha_z = (alpha - self._alpha_mu[regime]) / self._alpha_sd[regime]
            # Mostly de-risk, only modestly upsize.
            factor = np.where(alpha_z < 0.0, 1.0 + weights[regime] * alpha_z, 1.0 + 0.5 * weights[regime] * alpha_z)
            adj[mask] = np.clip(factor[mask], 0.75, 1.15) - 1.0
        return adj

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        ExtraTreesBadTailProbabilitySizerStrategy.fit(self, train_split, train_target_return)
        risk_x = build_risk_features(train_split)
        state_x = self._headline_state_frame(train_split)
        y = train_target_return.reindex(train_split.sessions).astype(float)
        self._fit_regime_state(risk_x, state_x, y)

    def predict(self, split: SplitInput) -> pd.Series:
        base_positions = ExtraTreesBadTailProbabilitySizerStrategy.predict(self, split).to_numpy(dtype=float)
        risk_x = build_risk_features(split)
        state_x = self._headline_state_frame(split)
        adjustment = self._regime_adjustment(risk_x, state_x)
        positions = np.clip(base_positions * (1.0 + adjustment), 0.0, 1.75)
        return pd.Series(positions, index=split.sessions, name="target_position")


def build_strategy() -> ExtraTreesBadTailRegimeMixtureStrategy:
    return ExtraTreesBadTailRegimeMixtureStrategy()
