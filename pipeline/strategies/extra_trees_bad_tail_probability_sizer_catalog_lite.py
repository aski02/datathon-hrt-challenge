from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pipeline.strategies.model_risk_utils import BaseClassifierSizedLongOnlyStrategy, build_risk_features
from pipeline.types import SplitInput


class ExtraTreesBadTailProbabilitySizerCatalogLiteStrategy(BaseClassifierSizedLongOnlyStrategy):
    name = "extra-trees-bad-tail-probability-sizer-catalog-lite"

    def __init__(self) -> None:
        super().__init__()
        self.min_survivor_position: float = 0.35
        self.max_survivor_position: float = 1.25
        self._headline_patterns: list[tuple[re.Pattern[str], dict[str, object]]] = []
        self._headline_parse_cache: dict[str, tuple[str, str, str, float]] = {}
        self._intent_labels: list[str] = []
        self._super_family_labels: list[str] = []

    def candidate_models(self) -> list[Pipeline]:
        configs = [
            (300, 4, 15),
            (400, 5, 10),
            (500, 6, 8),
        ]
        return [
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "etc",
                        ExtraTreesClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            max_features="sqrt",
                            random_state=42,
                            n_jobs=1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            )
            for n_estimators, max_depth, min_samples_leaf in configs
        ]

    def candidate_position_ranges(self) -> list[tuple[float, float]]:
        return [
            (0.35, 1.25),
            (0.35, 1.50),
            (0.35, 2.00),
            (0.50, 2.00),
        ]

    @staticmethod
    def _company_key(headline: str) -> str:
        parts = str(headline).split()
        return " ".join(parts[:2]).lower() if len(parts) >= 2 else str(headline).lower()

    @staticmethod
    def _clean_token(value: str) -> str:
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def _load_catalog(self) -> None:
        if self._headline_patterns:
            return

        root = Path(__file__).resolve().parents[2]
        catalog_path = root / "hrt-eth-zurich-datathon-2026" / "headlines" / "headline_template_catalog.csv"
        catalog = pd.read_csv(catalog_path)

        self._intent_labels = sorted(self._clean_token(v) for v in catalog["intent"].astype(str).tolist())
        self._super_family_labels = sorted(self._clean_token(v) for v in catalog["super_family"].astype(str).tolist())

        for row in catalog.itertuples(index=False):
            template = str(getattr(row, "template"))
            intent = self._clean_token(str(getattr(row, "intent")))
            super_family = self._clean_token(str(getattr(row, "super_family")))
            direction_prior = str(getattr(row, "direction_prior"))

            prior_sign = 0.0
            if direction_prior == "positive":
                prior_sign = 1.0
            elif direction_prior == "negative":
                prior_sign = -1.0

            parts = re.split(r"(<[^>]+>)", template)
            pattern_parts: list[str] = [r"^.+?\s+"]
            placeholder_ix = 0

            for part in parts:
                if not part:
                    continue
                if part.startswith("<") and part.endswith(">"):
                    group_name = f"ph_{placeholder_ix}"
                    placeholder_ix += 1
                    token_kind = part[1:-1].lower()
                    if token_kind == "num":
                        pattern_parts.append(
                            rf"(?P<{group_name}>[-+]?\$?\d+(?:\.\d+)?(?:\s*(?:[KMBT]|bn|mn|billion|million|thousand|year-over-year|%))?)"
                        )
                    elif token_kind == "year":
                        pattern_parts.append(rf"(?P<{group_name}>(?:19|20)\d{{2}})")
                    else:
                        pattern_parts.append(rf"(?P<{group_name}>.+?)")
                else:
                    pattern_parts.append(re.escape(part))

            pattern = re.compile("".join(pattern_parts) + r"$", re.IGNORECASE)
            metadata: dict[str, object] = {
                "intent": intent,
                "super_family": super_family,
                "prior_sign": prior_sign,
            }
            self._headline_patterns.append((pattern, metadata))

    def _parse_headline_text(self, text: str) -> tuple[str, str, str, float]:
        cached = self._headline_parse_cache.get(text)
        if cached is not None:
            return cached

        parsed = ("<unknown>", "<unknown>", "<unknown>", 0.0)
        for pattern, metadata in self._headline_patterns:
            if pattern.match(text) is None:
                continue
            parsed = (
                "<matched>",
                str(metadata["intent"]),
                str(metadata["super_family"]),
                float(metadata["prior_sign"]),
            )
            break

        self._headline_parse_cache[text] = parsed
        return parsed

    def _parse_headlines(self, split: SplitInput) -> pd.DataFrame:
        self._load_catalog()
        if split.headlines.empty:
            return pd.DataFrame(
                columns=["session", "bar_ix", "headline", "company", "intent", "super_family", "prior_sign", "late"]
            )

        rows = split.headlines[["session", "bar_ix", "headline"]].copy()
        parsed = rows["headline"].astype(str).map(self._parse_headline_text)
        parsed_df = pd.DataFrame(
            parsed.tolist(),
            columns=["template_tag", "intent", "super_family", "prior_sign"],
            index=rows.index,
        )
        rows = pd.concat([rows, parsed_df], axis=1)
        rows["company"] = rows["headline"].astype(str).map(self._company_key)
        rows["late"] = rows["bar_ix"] >= 40
        return rows.sort_values(["session", "bar_ix", "headline"]).reset_index(drop=True)

    @staticmethod
    def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        return a.astype(float) / b.astype(float).clip(lower=1e-9)

    def _build_headline_features(self, split: SplitInput) -> pd.DataFrame:
        sessions = split.sessions
        rows = self._parse_headlines(split)
        out = pd.DataFrame(index=sessions)

        for col in [
            "headline_count",
            "late_headline_count",
            "late_headline_share",
            "prior_balance_all",
            "prior_balance_late",
            "prior_balance_recent3",
            "prior_disagreement_all",
            "prior_disagreement_recent3",
            "last_headline_sign",
            "count_regime_low",
            "count_regime_mid",
            "count_regime_high",
        ]:
            out[col] = 0.0

        if rows.empty:
            for label in self._intent_labels:
                out[f"last_intent__{label}"] = 0.0
            for label in self._super_family_labels:
                out[f"last_super_family__{label}"] = 0.0
            return out

        headline_count = rows.groupby("session").size().reindex(sessions, fill_value=0).astype(float)
        late_count = rows[rows["late"]].groupby("session").size().reindex(sessions, fill_value=0).astype(float)
        recent3_rows = rows.groupby("session", sort=False).tail(3).copy()

        def prior_stats(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
            if frame.empty:
                zeros = pd.Series(0.0, index=sessions)
                return zeros, zeros
            g = frame.groupby("session", sort=False)
            prior_sum = g["prior_sign"].sum().reindex(sessions, fill_value=0.0).astype(float)
            prior_count = g.size().reindex(sessions, fill_value=0).astype(float)
            sign_counts = (
                frame.assign(
                    pos=(frame["prior_sign"] > 0).astype(int),
                    neg=(frame["prior_sign"] < 0).astype(int),
                )
                .groupby("session")[["pos", "neg"]]
                .sum()
                .reindex(sessions, fill_value=0)
            )
            balance = prior_sum / np.sqrt(prior_count.clip(lower=1.0))
            disagreement = np.minimum(sign_counts["pos"], sign_counts["neg"]) / prior_count.clip(lower=1.0)
            return balance.astype(float), disagreement.astype(float)

        balance_all, disagreement_all = prior_stats(rows)
        balance_late, _ = prior_stats(rows[rows["late"]])
        balance_recent3, disagreement_recent3 = prior_stats(recent3_rows)

        last_rows = rows.groupby("session", sort=False).tail(1).set_index("session")
        last_intent = pd.get_dummies(last_rows["intent"]).reindex(columns=self._intent_labels, fill_value=0.0)
        last_intent.index = last_rows.index
        last_intent = last_intent.reindex(sessions, fill_value=0.0)

        last_super_family = pd.get_dummies(last_rows["super_family"]).reindex(
            columns=self._super_family_labels, fill_value=0.0
        )
        last_super_family.index = last_rows.index
        last_super_family = last_super_family.reindex(sessions, fill_value=0.0)

        out["headline_count"] = headline_count
        out["late_headline_count"] = late_count
        out["late_headline_share"] = self._safe_div(late_count, headline_count.clip(lower=1.0))
        out["prior_balance_all"] = balance_all
        out["prior_balance_late"] = balance_late
        out["prior_balance_recent3"] = balance_recent3
        out["prior_disagreement_all"] = disagreement_all
        out["prior_disagreement_recent3"] = disagreement_recent3
        out["last_headline_sign"] = last_rows["prior_sign"].reindex(sessions).fillna(0.0).astype(float)
        out["count_regime_low"] = (headline_count <= 8).astype(float)
        out["count_regime_mid"] = ((headline_count >= 9) & (headline_count <= 11)).astype(float)
        out["count_regime_high"] = (headline_count >= 12).astype(float)

        last_intent.columns = [f"last_intent__{label}" for label in self._intent_labels]
        last_super_family.columns = [f"last_super_family__{label}" for label in self._super_family_labels]

        return out.join(last_intent).join(last_super_family).fillna(0.0)

    def _build_X(self, split: SplitInput) -> pd.DataFrame:
        risk_x = build_risk_features(split)
        headline_x = self._build_headline_features(split)
        return risk_x.join(headline_x).replace([np.inf, -np.inf], np.nan)

    def _positions_from_risk(self, risk: np.ndarray, risk_cutoff: float) -> np.ndarray:
        return self._positions_from_risk_with_bounds(
            risk,
            risk_cutoff,
            min_position=self.min_survivor_position,
            max_position=self.max_survivor_position,
        )

    @staticmethod
    def _positions_from_risk_with_bounds(
        risk: np.ndarray,
        risk_cutoff: float,
        min_position: float,
        max_position: float,
    ) -> np.ndarray:
        positions = np.zeros(len(risk), dtype=float)
        survivors = risk < risk_cutoff
        if not np.any(survivors):
            return positions

        score = 1.0 - risk[survivors] / max(risk_cutoff, 1e-9)
        score = np.clip(score, 0.0, 1.0)
        positions[survivors] = min_position + (max_position - min_position) * score
        return positions


def build_strategy() -> ExtraTreesBadTailProbabilitySizerCatalogLiteStrategy:
    return ExtraTreesBadTailProbabilitySizerCatalogLiteStrategy()
