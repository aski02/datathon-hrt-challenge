from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import KFold

from pipeline.strategy import sharpe_from_positions
from pipeline.types import SplitInput


def build_risk_features(split: SplitInput) -> pd.DataFrame:
    features = split.features.copy()
    X = pd.DataFrame(index=split.sessions)

    # --- NUR SENTIMENT FEATURES (Naked Test) ---
    if "claude_bullish" in features.columns:
        X["bullish"] = features["claude_bullish"].fillna(0)
        X["bearish"] = features["claude_bearish"].fillna(0)
        X["uncertainty"] = features["claude_uncertainty"].fillna(1.0) # Standard: Unsicher
        X["surprise"] = features["claude_surprise"].fillna(0)
        X["keyword"] = features["keyword_sentiment"].fillna(0)
        
        # Ein kombiniertes Alpha-Feature: Netto-Sentiment gewichtet mit Sicherheit
        # Wenn Uncertainty hoch ist (nahe 1), geht das Signal gegen 0.
        X["net_conviction"] = (X["bullish"] - X["bearish"]) * (1.0 - X["uncertainty"])
        
        # Interaktion: Ist die News eine Überraschung?
        X["surprise_impact"] = X["surprise"] * (X["bullish"] - X["bearish"])
    else:
        # Falls die Spalten fehlen (Sicherheit)
        X["dummy"] = 0.0

    return X.replace([np.inf, -np.inf], np.nan)
class BaseModelRiskFilterStrategy(ABC):
    """Shared long-or-flat risk-filter logic for model comparisons."""

    name = "base-model-risk-filter"

    def __init__(self) -> None:
        self.flat_quantile: float = 0.15
        self.model: RegressorMixin | None = None
        self.pred_cutoff: float = 0.0

    @abstractmethod
    def candidate_models(self) -> list[RegressorMixin]:
        raise NotImplementedError

    def candidate_quantiles(self) -> list[float]:
        return [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    def _build_X(self, split: SplitInput) -> pd.DataFrame:
        return build_risk_features(split)

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        X = self._build_X(train_split)
        y = train_target_return.reindex(train_split.sessions).astype(float)

        splitter = KFold(n_splits=5, shuffle=False, random_state=42)
        best_score = -np.inf
        best_model: RegressorMixin | None = None
        best_quantile = self.flat_quantile

        for model in self.candidate_models():
            quantile_scores = {q: [] for q in self.candidate_quantiles()}
            for train_idx, valid_idx in splitter.split(X):
                X_train = X.iloc[train_idx]
                X_valid = X.iloc[valid_idx]
                y_train = y.iloc[train_idx]
                y_valid = y.iloc[valid_idx]

                fitted = clone(model)
                fitted.fit(X_train, y_train)

                pred_train = fitted.predict(X_train)
                pred_valid = fitted.predict(X_valid)
                for quantile in self.candidate_quantiles():
                    cutoff = float(np.quantile(pred_train, quantile))
                    positions = pd.Series((pred_valid > cutoff).astype(float), index=y_valid.index)
                    quantile_scores[quantile].append(sharpe_from_positions(positions, y_valid))

            for quantile, scores in quantile_scores.items():
                score = float(np.mean(scores))
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_model = clone(model)
                    best_quantile = quantile

        if best_model is None:
            raise RuntimeError("No valid model candidate found during Sharpe tuning.")

        self.flat_quantile = best_quantile
        self.model = best_model.fit(X, y)
        pred_train = self.model.predict(X)
        self.pred_cutoff = float(np.quantile(pred_train, self.flat_quantile))

    def predict(self, split: SplitInput) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted. `fit(...)` must run before `predict(...)`.")
        X = self._build_X(split)
        pred = self.model.predict(X)
        positions = pd.Series((pred > self.pred_cutoff).astype(float), index=split.sessions, name="target_position")
        return positions


class BaseClassifierRiskFilterStrategy(ABC):
    """Shared long-or-flat risk-filter logic for bad-tail classifiers."""

    name = "base-classifier-risk-filter"

    def __init__(self) -> None:
        self.bad_tail_quantile: float = 0.20
        self.flat_quantile: float = 0.15
        self.model: ClassifierMixin | None = None
        self.risk_cutoff: float = 0.0

    @abstractmethod
    def candidate_models(self) -> list[ClassifierMixin]:
        raise NotImplementedError

    def candidate_bad_tail_quantiles(self) -> list[float]:
        return [0.15, 0.20, 0.25, 0.30]

    def candidate_flat_quantiles(self) -> list[float]:
        return [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    def _build_X(self, split: SplitInput) -> pd.DataFrame:
        return build_risk_features(split)

    @staticmethod
    def _positive_score(model: ClassifierMixin, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        raise TypeError("Classifier must support predict_proba or decision_function.")

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        X = self._build_X(train_split)
        y = train_target_return.reindex(train_split.sessions).astype(float)
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)

        best_score = -np.inf
        best_model: ClassifierMixin | None = None
        best_bad_tail_quantile = self.bad_tail_quantile
        best_flat_quantile = self.flat_quantile

        for bad_tail_quantile in self.candidate_bad_tail_quantiles():
            cutoff_y = float(np.quantile(y, bad_tail_quantile))
            labels = (y <= cutoff_y).astype(int)
            if labels.nunique() < 2:
                continue

            for model in self.candidate_models():
                quantile_scores = {q: [] for q in self.candidate_flat_quantiles()}
                for train_idx, valid_idx in splitter.split(X):
                    X_train = X.iloc[train_idx]
                    X_valid = X.iloc[valid_idx]
                    y_valid = y.iloc[valid_idx]
                    labels_train = labels.iloc[train_idx]

                    fitted = clone(model)
                    fitted.fit(X_train, labels_train)

                    risk_train = self._positive_score(fitted, X_train)
                    risk_valid = self._positive_score(fitted, X_valid)
                    for flat_quantile in self.candidate_flat_quantiles():
                        risk_cutoff = float(np.quantile(risk_train, 1.0 - flat_quantile))
                        positions = pd.Series((risk_valid < risk_cutoff).astype(float), index=y_valid.index)
                        quantile_scores[flat_quantile].append(sharpe_from_positions(positions, y_valid))

                for flat_quantile, scores in quantile_scores.items():
                    score = float(np.mean(scores))
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_model = clone(model)
                        best_bad_tail_quantile = bad_tail_quantile
                        best_flat_quantile = flat_quantile

        if best_model is None:
            raise RuntimeError("No valid classifier candidate found during Sharpe tuning.")

        self.bad_tail_quantile = best_bad_tail_quantile
        self.flat_quantile = best_flat_quantile

        cutoff_y = float(np.quantile(y, self.bad_tail_quantile))
        labels = (y <= cutoff_y).astype(int)
        self.model = best_model.fit(X, labels)
        risk_train = self._positive_score(self.model, X)
        self.risk_cutoff = float(np.quantile(risk_train, 1.0 - self.flat_quantile))

    def predict(self, split: SplitInput) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted. `fit(...)` must run before `predict(...)`.")
        X = self._build_X(split)
        risk = self._positive_score(self.model, X)
        positions = pd.Series((risk < self.risk_cutoff).astype(float), index=split.sessions, name="target_position")
        return positions


class BaseTwoStageLongOnlyStrategy(ABC):
    """Two-stage long-only strategy: bad-tail gate plus continuous bounded sizing."""

    name = "base-two-stage-long-only"

    def __init__(self) -> None:
        self.bad_tail_quantile: float = 0.20
        self.flat_quantile: float = 0.15
        self.min_position: float = 0.25
        self.max_position: float = 1.25
        self.gate_model: ClassifierMixin | None = None
        self.size_model: RegressorMixin | None = None
        self.risk_cutoff: float = 0.0
        self.size_lo: float = -1.0
        self.size_hi: float = 1.0

    @abstractmethod
    def candidate_gate_models(self) -> list[ClassifierMixin]:
        raise NotImplementedError

    @abstractmethod
    def candidate_size_models(self) -> list[RegressorMixin]:
        raise NotImplementedError

    def candidate_bad_tail_quantiles(self) -> list[float]:
        return [0.15, 0.20, 0.25, 0.30]

    def candidate_flat_quantiles(self) -> list[float]:
        return [0.10, 0.15, 0.20]

    def candidate_position_bounds(self) -> list[tuple[float, float]]:
        return [(0.20, 1.10), (0.25, 1.25), (0.35, 1.40)]

    def _build_X(self, split: SplitInput) -> pd.DataFrame:
        return build_risk_features(split)

    @staticmethod
    def _positive_score(model: ClassifierMixin, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        raise TypeError("Classifier must support predict_proba or decision_function.")

    @staticmethod
    def _scale_survivor_scores(
        scores: np.ndarray,
        survivor_mask: np.ndarray,
        min_position: float,
        max_position: float,
        lo: float,
        hi: float,
    ) -> np.ndarray:
        positions = np.zeros(len(scores), dtype=float)
        if not np.any(survivor_mask):
            return positions

        if hi <= lo + 1e-12:
            positions[survivor_mask] = 0.5 * (min_position + max_position)
            return positions

        scaled = np.clip((scores[survivor_mask] - lo) / (hi - lo), 0.0, 1.0)
        positions[survivor_mask] = min_position + (max_position - min_position) * scaled
        return positions

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        X = self._build_X(train_split)
        y = train_target_return.reindex(train_split.sessions).astype(float)
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)

        best_gate_score = -np.inf
        best_gate_model: ClassifierMixin | None = None
        best_bad_tail_quantile = self.bad_tail_quantile
        best_flat_quantile = self.flat_quantile
        for bad_tail_quantile in self.candidate_bad_tail_quantiles():
            cutoff_y = float(np.quantile(y, bad_tail_quantile))
            labels = (y <= cutoff_y).astype(int)
            if labels.nunique() < 2:
                continue

            for gate_model in self.candidate_gate_models():
                quantile_scores = {q: [] for q in self.candidate_flat_quantiles()}
                for train_idx, valid_idx in splitter.split(X):
                    X_train = X.iloc[train_idx]
                    X_valid = X.iloc[valid_idx]
                    y_valid = y.iloc[valid_idx]
                    labels_train = labels.iloc[train_idx]

                    fitted_gate = clone(gate_model)
                    fitted_gate.fit(X_train, labels_train)
                    risk_train = self._positive_score(fitted_gate, X_train)
                    risk_valid = self._positive_score(fitted_gate, X_valid)
                    for flat_quantile in self.candidate_flat_quantiles():
                        gate_cutoff = float(np.quantile(risk_train, 1.0 - flat_quantile))
                        positions = pd.Series((risk_valid < gate_cutoff).astype(float), index=y_valid.index)
                        quantile_scores[flat_quantile].append(sharpe_from_positions(positions, y_valid))

                for flat_quantile, scores in quantile_scores.items():
                    score = float(np.mean(scores))
                    if np.isfinite(score) and score > best_gate_score:
                        best_gate_score = score
                        best_gate_model = clone(gate_model)
                        best_bad_tail_quantile = bad_tail_quantile
                        best_flat_quantile = flat_quantile

        if best_gate_model is None:
            raise RuntimeError("No valid gate candidate found during Sharpe tuning.")

        best_size_score = -np.inf
        best_size_model: RegressorMixin | None = None
        best_bounds = (self.min_position, self.max_position)
        cutoff_y = float(np.quantile(y, best_bad_tail_quantile))
        labels = (y <= cutoff_y).astype(int)
        for size_model in self.candidate_size_models():
            for min_position, max_position in self.candidate_position_bounds():
                fold_scores: list[float] = []
                for train_idx, valid_idx in splitter.split(X):
                    X_train = X.iloc[train_idx]
                    X_valid = X.iloc[valid_idx]
                    y_train = y.iloc[train_idx]
                    y_valid = y.iloc[valid_idx]
                    labels_train = labels.iloc[train_idx]

                    fitted_gate = clone(best_gate_model)
                    fitted_size = clone(size_model)
                    fitted_gate.fit(X_train, labels_train)
                    fitted_size.fit(X_train, y_train)

                    risk_train = self._positive_score(fitted_gate, X_train)
                    risk_valid = self._positive_score(fitted_gate, X_valid)
                    gate_cutoff = float(np.quantile(risk_train, 1.0 - best_flat_quantile))
                    survivor_train = risk_train < gate_cutoff
                    survivor_valid = risk_valid < gate_cutoff

                    size_train = fitted_size.predict(X_train)
                    size_valid = fitted_size.predict(X_valid)
                    reference_scores = size_train[survivor_train] if np.any(survivor_train) else size_train
                    lo = float(np.quantile(reference_scores, 0.10))
                    hi = float(np.quantile(reference_scores, 0.90))
                    positions = self._scale_survivor_scores(
                        scores=size_valid,
                        survivor_mask=survivor_valid,
                        min_position=min_position,
                        max_position=max_position,
                        lo=lo,
                        hi=hi,
                    )
                    fold_scores.append(sharpe_from_positions(pd.Series(positions, index=y_valid.index), y_valid))

                score = float(np.mean(fold_scores))
                if np.isfinite(score) and score > best_size_score:
                    best_size_score = score
                    best_size_model = clone(size_model)
                    best_bounds = (min_position, max_position)

        if best_size_model is None:
            raise RuntimeError("No valid sizing candidate found during Sharpe tuning.")

        self.bad_tail_quantile = best_bad_tail_quantile
        self.flat_quantile = best_flat_quantile
        self.min_position, self.max_position = best_bounds

        labels = (y <= float(np.quantile(y, self.bad_tail_quantile))).astype(int)
        self.gate_model = best_gate_model.fit(X, labels)
        self.size_model = best_size_model.fit(X, y)

        risk_train = self._positive_score(self.gate_model, X)
        self.risk_cutoff = float(np.quantile(risk_train, 1.0 - self.flat_quantile))
        survivor_train = risk_train < self.risk_cutoff
        size_train = self.size_model.predict(X)
        reference_scores = size_train[survivor_train] if np.any(survivor_train) else size_train
        self.size_lo = float(np.quantile(reference_scores, 0.10))
        self.size_hi = float(np.quantile(reference_scores, 0.90))

    def predict(self, split: SplitInput) -> pd.Series:
        if self.gate_model is None or self.size_model is None:
            raise RuntimeError("Strategy not fitted. `fit(...)` must run before `predict(...)`.")

        X = self._build_X(split)
        risk = self._positive_score(self.gate_model, X)
        survivor_mask = risk < self.risk_cutoff
        size_scores = self.size_model.predict(X)
        positions = self._scale_survivor_scores(
            scores=size_scores,
            survivor_mask=survivor_mask,
            min_position=self.min_position,
            max_position=self.max_position,
            lo=self.size_lo,
            hi=self.size_hi,
        )
        return pd.Series(positions, index=split.sessions, name="target_position")


class BaseClassifierSizedLongOnlyStrategy(ABC):
    """Classifier-only long-only strategy: gate risky sessions, size survivors from classifier risk."""

    name = "base-classifier-sized-long-only"

    def __init__(self) -> None:
        self.bad_tail_quantile: float = 0.20
        self.flat_quantile: float = 0.15
        self.model: ClassifierMixin | None = None
        self.risk_cutoff: float = 0.0

    @abstractmethod
    def candidate_models(self) -> list[ClassifierMixin]:
        raise NotImplementedError

    @abstractmethod
    def _positions_from_risk(self, risk: np.ndarray, risk_cutoff: float) -> np.ndarray:
        raise NotImplementedError

    def candidate_bad_tail_quantiles(self) -> list[float]:
        return [0.15, 0.20, 0.25, 0.30]

    def candidate_flat_quantiles(self) -> list[float]:
        return [0.05, 0.10, 0.15, 0.20, 0.25]

    def _build_X(self, split: SplitInput) -> pd.DataFrame:
        return build_risk_features(split)

    @staticmethod
    def _positive_score(model: ClassifierMixin, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        raise TypeError("Classifier must support predict_proba or decision_function.")

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        X = self._build_X(train_split)
        y = train_target_return.reindex(train_split.sessions).astype(float)
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)

        best_score = -np.inf
        best_model: ClassifierMixin | None = None
        best_bad_tail_quantile = self.bad_tail_quantile
        best_flat_quantile = self.flat_quantile

        for bad_tail_quantile in self.candidate_bad_tail_quantiles():
            cutoff_y = float(np.quantile(y, bad_tail_quantile))
            labels = (y <= cutoff_y).astype(int)
            if labels.nunique() < 2:
                continue

            for model in self.candidate_models():
                quantile_scores = {q: [] for q in self.candidate_flat_quantiles()}
                for train_idx, valid_idx in splitter.split(X):
                    X_train = X.iloc[train_idx]
                    X_valid = X.iloc[valid_idx]
                    y_valid = y.iloc[valid_idx]
                    labels_train = labels.iloc[train_idx]

                    fitted = clone(model)
                    fitted.fit(X_train, labels_train)

                    risk_train = self._positive_score(fitted, X_train)
                    risk_valid = self._positive_score(fitted, X_valid)
                    for flat_quantile in self.candidate_flat_quantiles():
                        risk_cutoff = float(np.quantile(risk_train, 1.0 - flat_quantile))
                        positions = self._positions_from_risk(risk_valid, risk_cutoff)
                        quantile_scores[flat_quantile].append(
                            sharpe_from_positions(pd.Series(positions, index=y_valid.index), y_valid)
                        )

                for flat_quantile, scores in quantile_scores.items():
                    score = float(np.mean(scores))
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_model = clone(model)
                        best_bad_tail_quantile = bad_tail_quantile
                        best_flat_quantile = flat_quantile

        if best_model is None:
            raise RuntimeError("No valid classifier-sized candidate found during Sharpe tuning.")

        self.bad_tail_quantile = best_bad_tail_quantile
        self.flat_quantile = best_flat_quantile

        cutoff_y = float(np.quantile(y, self.bad_tail_quantile))
        labels = (y <= cutoff_y).astype(int)
        self.model = best_model.fit(X, labels)
        risk_train = self._positive_score(self.model, X)
        self.risk_cutoff = float(np.quantile(risk_train, 1.0 - self.flat_quantile))

    def predict(self, split: SplitInput) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted. `fit(...)` must run before `predict(...)`.")
        X = self._build_X(split)
        risk = self._positive_score(self.model, X)
        positions = self._positions_from_risk(risk, self.risk_cutoff)
        return pd.Series(positions, index=split.sessions, name="target_position")
