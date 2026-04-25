from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from pipeline.strategies.btp_rank_tpl import BtpRankTplStrategy
from pipeline.strategies.extra_trees_bad_tail_probability_rank_sizer import (
    ExtraTreesBadTailProbabilityRankSizerStrategy,
)
from pipeline.strategies.model_risk_utils import build_risk_features


class BtpRankHdocStrategy(BtpRankTplStrategy):
    """Canonical template/intention sequence docs with supervised OOF headline predictions."""

    name = "btp-rank-hdoc"

    def __init__(self) -> None:
        super().__init__()
        self._headline_doc_models: dict[str, Pipeline] = {}
        self._headline_doc_cache: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _repeat_count(bar_ix: float, recent3_flag: float) -> int:
        clipped = max(0.0, min(49.0, float(bar_ix)))
        repeat = 1 + int(np.floor(4.0 * clipped / 49.0))
        if clipped >= 45.0:
            repeat += 1
        if recent3_flag > 0:
            repeat += 1
        return max(1, min(7, repeat))

    def _build_headline_docs(self, split) -> pd.DataFrame:
        sessions = split.sessions
        rows = self._parse_headlines(split)
        docs = pd.DataFrame(index=sessions)
        docs["headline_count"] = 0.0
        docs["late_count"] = 0.0
        docs["unique_template_count"] = 0.0
        docs["unique_intent_count"] = 0.0
        docs["headline_last_bar_norm"] = 0.0
        docs["template_doc"] = ""
        docs["intent_doc"] = ""
        docs["transition_doc"] = ""
        docs["late_doc"] = ""

        if rows.empty:
            return docs

        def _render(frame: pd.DataFrame) -> pd.Series:
            ordered = frame.sort_values("bar_ix", kind="stable").copy()
            ordered["tpl_token"] = ordered["template"].astype(str).str.replace(" ", "_", regex=False)
            ordered["intent_token"] = ordered["intent"].astype(str)
            ordered["repeat_count"] = [
                self._repeat_count(bar_ix, recent3_flag)
                for bar_ix, recent3_flag in zip(ordered["bar_ix"], ordered["recent3"], strict=True)
            ]

            template_parts: list[str] = []
            intent_parts: list[str] = []
            for row in ordered.itertuples(index=False):
                template_parts.extend([row.tpl_token] * int(row.repeat_count))
                intent_parts.extend([row.intent_token] * int(row.repeat_count))

            transitions: list[str] = []
            for prev_row, next_row in zip(
                ordered.itertuples(index=False), ordered.iloc[1:].itertuples(index=False), strict=False
            ):
                transition = f"{prev_row.intent_token}__to__{next_row.intent_token}"
                transitions.extend([transition] * max(int(next_row.repeat_count), 1))

            late_rows = ordered[ordered["late"] > 0]
            if late_rows.empty:
                late_rows = ordered.tail(min(2, len(ordered)))
            late_doc = " ".join(late_rows["tpl_token"])

            return pd.Series(
                {
                    "headline_count": float(len(ordered)),
                    "late_count": float((ordered["late"] > 0).sum()),
                    "unique_template_count": float(ordered["template"].nunique()),
                    "unique_intent_count": float(ordered["intent"].nunique()),
                    "headline_last_bar_norm": float(ordered["bar_ix"].max() / 49.0),
                    "template_doc": " ".join(template_parts),
                    "intent_doc": " ".join(intent_parts),
                    "transition_doc": " ".join(transitions),
                    "late_doc": late_doc,
                }
            )

        rendered = rows.groupby("session", sort=False).apply(_render, include_groups=False)
        rendered.index.name = "session"
        docs.update(rendered.reindex(sessions))
        for col in ["template_doc", "intent_doc", "transition_doc", "late_doc"]:
            docs[col] = docs[col].fillna("")
        for col in [
            "headline_count",
            "late_count",
            "unique_template_count",
            "unique_intent_count",
            "headline_last_bar_norm",
        ]:
            docs[col] = docs[col].fillna(0.0).astype(float)
        return docs

    @staticmethod
    def _make_ridge_pipeline() -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.995,
                        max_features=12000,
                        sublinear_tf=True,
                    ),
                ),
                ("ridge", Ridge(alpha=2.0)),
            ]
        )

    @staticmethod
    def _make_logit_pipeline() -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.995,
                        max_features=12000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "logit",
                    LogisticRegression(
                        C=2.0,
                        solver="liblinear",
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    @staticmethod
    def _predict_model(model: Pipeline, kind: str, docs: pd.Series) -> np.ndarray:
        if kind == "regression":
            return model.predict(docs)
        return model.predict_proba(docs)[:, 1]

    def _fit_headline_doc_features(self, train_split, train_target_return: pd.Series) -> None:
        docs = self._build_headline_docs(train_split)
        y = train_target_return.reindex(train_split.sessions).astype(float)
        bad_tail_cutoff = float(np.quantile(y, self.template_bad_tail_quantile))
        up_labels = (y > 0.0).astype(int)
        bad_tail_labels = (y <= bad_tail_cutoff).astype(int)

        specs = [
            ("hdoc_ret_tpl", "template_doc", "regression", y, self._make_ridge_pipeline),
            ("hdoc_ret_trans", "transition_doc", "regression", y, self._make_ridge_pipeline),
            ("hdoc_up_prob", "intent_doc", "classification", up_labels, self._make_logit_pipeline),
            ("hdoc_bad_tail_prob", "late_doc", "classification", bad_tail_labels, self._make_logit_pipeline),
        ]

        feature_frame = docs[
            [
                "headline_count",
                "late_count",
                "unique_template_count",
                "unique_intent_count",
                "headline_last_bar_norm",
            ]
        ].copy()

        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        for feature_name, doc_col, kind, target, factory in specs:
            oof_pred = np.zeros(len(docs), dtype=float)
            doc_values = docs[doc_col]

            for train_idx, valid_idx in splitter.split(docs):
                y_train = target.iloc[train_idx]
                if kind == "classification" and y_train.nunique() < 2:
                    oof_pred[valid_idx] = float(y_train.mean())
                    continue
                model = factory()
                model.fit(doc_values.iloc[train_idx], y_train)
                oof_pred[valid_idx] = self._predict_model(model, kind, doc_values.iloc[valid_idx])

            feature_frame[feature_name] = oof_pred.astype(float)
            final_model = factory()
            final_model.fit(doc_values, target)
            self._headline_doc_models[feature_name] = final_model

        feature_frame["hdoc_signal_spread"] = (
            feature_frame["hdoc_up_prob"] - feature_frame["hdoc_bad_tail_prob"]
        ).astype(float)
        feature_frame["hdoc_combo_return"] = (
            0.5 * feature_frame["hdoc_ret_tpl"] + 0.5 * feature_frame["hdoc_ret_trans"]
        ).astype(float)
        feature_frame["hdoc_abs_gate"] = np.maximum(np.abs(feature_frame["hdoc_signal_spread"]) - 0.15, 0.0).astype(
            float
        )
        feature_frame["hdoc_late_gate"] = np.maximum(feature_frame["hdoc_bad_tail_prob"] - 0.55, 0.0).astype(float)
        feature_frame["hdoc_pos_gate"] = np.maximum(feature_frame["hdoc_up_prob"] - 0.55, 0.0).astype(float)
        feature_frame["hdoc_recent_strength"] = (
            feature_frame["hdoc_combo_return"] * feature_frame["headline_last_bar_norm"]
        ).astype(float)

        self._headline_doc_cache = {"train_seen": feature_frame.reindex(train_split.sessions).fillna(0.0)}

    def _build_headline_features(self, split) -> pd.DataFrame:
        cached = self._headline_doc_cache.get(split.name)
        if cached is not None:
            return cached.reindex(split.sessions).fillna(0.0)

        docs = self._build_headline_docs(split)
        feature_frame = docs[
            [
                "headline_count",
                "late_count",
                "unique_template_count",
                "unique_intent_count",
                "headline_last_bar_norm",
            ]
        ].copy()

        if not self._headline_doc_models:
            for col in ["hdoc_ret_tpl", "hdoc_ret_trans", "hdoc_up_prob", "hdoc_bad_tail_prob"]:
                feature_frame[col] = 0.0
        else:
            feature_frame["hdoc_ret_tpl"] = self._predict_model(
                self._headline_doc_models["hdoc_ret_tpl"], "regression", docs["template_doc"]
            )
            feature_frame["hdoc_ret_trans"] = self._predict_model(
                self._headline_doc_models["hdoc_ret_trans"], "regression", docs["transition_doc"]
            )
            feature_frame["hdoc_up_prob"] = self._predict_model(
                self._headline_doc_models["hdoc_up_prob"], "classification", docs["intent_doc"]
            )
            feature_frame["hdoc_bad_tail_prob"] = self._predict_model(
                self._headline_doc_models["hdoc_bad_tail_prob"], "classification", docs["late_doc"]
            )

        feature_frame["hdoc_signal_spread"] = (
            feature_frame["hdoc_up_prob"] - feature_frame["hdoc_bad_tail_prob"]
        ).astype(float)
        feature_frame["hdoc_combo_return"] = (
            0.5 * feature_frame["hdoc_ret_tpl"] + 0.5 * feature_frame["hdoc_ret_trans"]
        ).astype(float)
        feature_frame["hdoc_abs_gate"] = np.maximum(np.abs(feature_frame["hdoc_signal_spread"]) - 0.15, 0.0).astype(
            float
        )
        feature_frame["hdoc_late_gate"] = np.maximum(feature_frame["hdoc_bad_tail_prob"] - 0.55, 0.0).astype(float)
        feature_frame["hdoc_pos_gate"] = np.maximum(feature_frame["hdoc_up_prob"] - 0.55, 0.0).astype(float)
        feature_frame["hdoc_recent_strength"] = (
            feature_frame["hdoc_combo_return"] * feature_frame["headline_last_bar_norm"]
        ).astype(float)

        feature_frame = feature_frame.reindex(split.sessions).fillna(0.0)
        self._headline_doc_cache[split.name] = feature_frame
        return feature_frame

    def fit(self, train_split, train_target_return: pd.Series) -> None:
        self._fit_headline_doc_features(train_split, train_target_return)
        ExtraTreesBadTailProbabilityRankSizerStrategy.fit(self, train_split, train_target_return)

    def _build_X(self, split):
        risk_x = build_risk_features(split)
        headline_x = self._build_headline_features(split)
        return risk_x.join(headline_x).replace([np.inf, -np.inf], np.nan)


def build_strategy() -> BtpRankHdocStrategy:
    return BtpRankHdocStrategy()
