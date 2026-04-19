from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pipeline.strategies.extra_trees_bad_tail_probability_rank_sizer import (
    ExtraTreesBadTailProbabilityRankSizerStrategy,
)
from pipeline.strategies.model_risk_utils import build_risk_features
from pipeline.types import SplitInput


NUM_PAT = re.compile(
    r"(?:\$\s*)?\d+(?:[\.,]\d+)?\s*(?:[kmbt]|bn|mn|billion|million|thousand)?%?",
    re.I,
)
YEAR_PAT = re.compile(r"\b(19|20)\d{2}\b")
WS_PAT = re.compile(r"\s+")
ROLE_PAT = re.compile(r"\b(?:ceo|cfo|cto|chief\s+[a-z]+\s+officer)\b", re.I)
REGION_PAT = re.compile(
    r"\b(?:europe|scandinavia|southeast\s+asia|middle\s+east|latin\s+america|north\s+america|asia\s+pacific|africa|central\s+asia)\b",
    re.I,
)
LEADING_CORP_NOISE = re.compile(r"^(?:co|group|holdings|corp|corporation|inc|ltd|plc|ag)\s+", re.I)

DOMAIN_TERMS = [
    "cloud infrastructure",
    "supply chain optimization",
    "wireless connectivity",
    "renewable storage",
    "process automation",
    "enterprise software",
    "precision manufacturing",
    "digital payments",
    "automated logistics",
]
DOMAIN_PAT = re.compile(r"\b(?:" + "|".join(re.escape(x) for x in DOMAIN_TERMS) + r")\b", re.I)

CANON_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(secures\s+<NUM>\s+contract\s+with)\s+.+$", re.I), r"\1 <PARTNER>"),
    (re.compile(r"\b(signs\s+multi-year\s+partnership\s+with)\s+.+$", re.I), r"\1 <PARTNER>"),
    (re.compile(r"\b(forms\s+strategic\s+alliance\s+with)\s+.+$", re.I), r"\1 <PARTNER>"),
    (re.compile(r"\b(expands\s+distribution\s+deal\s+with)\s+.+$", re.I), r"\1 <PARTNER>"),
    (re.compile(r"\b(enters\s+joint\s+venture\s+with)\s+.+$", re.I), r"\1 <PARTNER>"),
    (re.compile(r"\bin\s+.+\s+segment\b", re.I), "in <DOMAIN> segment"),
    (re.compile(r"\bof\s+.+\s+systems\b", re.I), "of <DOMAIN> systems"),
    (re.compile(r"\bfor\s+regulatory\s+approval\s+of\s+new\s+.+\s+offering\b", re.I), "for regulatory approval of new <DOMAIN> offering"),
    (re.compile(r"\bin\s+.+\s+line\s+due\s+to\s+quality\s+concerns\b", re.I), "in <DOMAIN> line due to quality concerns"),
    (re.compile(r"\bof\s+.+\s+practices\b", re.I), "of <DOMAIN> practices"),
    (re.compile(r"\bfor\s+.+\s+unit\b", re.I), "for <DOMAIN> unit"),
    (re.compile(r"\bin\s+.+\s+pilot\s+program\b", re.I), "in <DOMAIN> pilot program"),
    (re.compile(r"\bfocus\s+on\s+.+$", re.I), "focus on <DOMAIN>"),
    (re.compile(r"\bfor\s+excellence\s+in\s+.+$", re.I), "for excellence in <DOMAIN>"),
    (re.compile(r"\bfiles\s+routine\s+patent\s+applications\s+in\s+.+$", re.I), "files routine patent applications in <DOMAIN>"),
    (re.compile(r"\breports\s+rising\s+costs\s+pressuring\s+margins\s+in\s+.+$", re.I), "reports rising costs pressuring margins in <DOMAIN>"),
    (re.compile(r"\blaunches\s+next-generation\s+.+\s+platform\b", re.I), "launches next-generation <DOMAIN> platform"),
    (re.compile(r"\bannounces\s+breakthrough\s+in\s+.+$", re.I), "announces breakthrough in <DOMAIN>"),
    (re.compile(r"\bfaces\s+class\s+action\s+over\s+.+\s+service\s+disruption\b", re.I), "faces class action over <DOMAIN> service disruption"),
    (re.compile(r"\binto\s+.+\s+markets\b", re.I), "into <REGION> markets"),
    (re.compile(r"\bopens\s+new\s+office\s+in\s+.+$", re.I), "opens new office in <REGION>"),
    (re.compile(r"\bannounces\s+significant\s+capital\s+expenditure\s+plan\s+for\s+.+$", re.I), "announces significant capital expenditure plan for <REGION>"),
    (re.compile(r"\breports\s+strong\s+demand\s+in\s+.+,\s+raises\s+outlook\b", re.I), "reports strong demand in <REGION>, raises outlook"),
    (re.compile(r"\bwarns\s+of\s+supply\s+chain\s+disruptions\s+affecting\s+.+\s+operations\b", re.I), "warns of supply chain disruptions affecting <REGION> operations"),
    (re.compile(r"\bcompletes\s+planned\s+facility\s+upgrade\s+in\s+.+$", re.I), "completes planned facility upgrade in <REGION>"),
    (re.compile(r"\bloses\s+key\s+contract\s+in\s+.+\s+to\s+competitor\b", re.I), "loses key contract in <REGION> to competitor"),
    (re.compile(r"\breports\s+unexpected\s+decline\s+in\s+.+\s+revenue\b", re.I), "reports unexpected decline in <REGION> revenue"),
    (re.compile(r"\bwithdraws\s+from\s+.+\s+market\s+citing\s+unfavorable\s+conditions\b", re.I), "withdraws from <REGION> market citing unfavorable conditions"),
    (re.compile(r"\bappoints\s+new\s+.+\s+to\s+board\b", re.I), "appoints new <ROLE> to board"),
    (re.compile(r"\bnames\s+new\s+head\s+of\s+.+\s+division\b", re.I), "names new head of <DOMAIN> division"),
    (re.compile(r"\b(?:ceo|cfo|cto|chief\s+[a-z]+\s+officer)\s+steps\s+down\s+unexpectedly\s+citing\s+personal\s+reasons\b", re.I), "<ROLE> steps down unexpectedly citing personal reasons"),
    (re.compile(r"\b(?:ceo|cfo|cto|chief\s+[a-z]+\s+officer)\s+addresses\s+investor\s+concerns\s+in\s+open\s+letter\b", re.I), "<ROLE> addresses investor concerns in open letter"),
]


class BtpRankTplStrategy(ExtraTreesBadTailProbabilityRankSizerStrategy):
    """ExtraTrees base features plus recency-weighted template-supervised headline features."""

    name = "btp-rank-tpl"

    def __init__(self) -> None:
        super().__init__()
        self.template_bad_tail_quantile: float = 0.20
        self.template_smoothing: float = 20.0
        self.recency_alpha: float = 0.10
        self._template_meta: dict[str, tuple[str, str, float]] | None = None
        self._headline_parse_cache: dict[str, tuple[str, str, str, float]] = {}
        self._template_feature_cache: dict[str, pd.DataFrame] = {}
        self._train_template_features: pd.DataFrame | None = None
        self._full_template_tables: dict[str, pd.DataFrame | float] = {}

    @staticmethod
    def _split_company_and_rest(headline: str) -> tuple[str, str]:
        parts = str(headline).split()
        company = " ".join(parts[:2]) if len(parts) >= 2 else str(headline)
        rest = " ".join(parts[2:]) if len(parts) >= 2 else ""
        return company, rest

    @classmethod
    def _canonicalize_rest(cls, rest: str) -> str:
        x = str(rest).lower().strip()
        x = LEADING_CORP_NOISE.sub("", x)
        x = YEAR_PAT.sub("<YEAR>", x)
        x = NUM_PAT.sub("<NUM>", x)
        x = ROLE_PAT.sub("<ROLE>", x)
        x = REGION_PAT.sub("<REGION>", x)
        x = DOMAIN_PAT.sub("<DOMAIN>", x)
        x = WS_PAT.sub(" ", x).strip()
        for pat, rep in CANON_RULES:
            x = pat.sub(rep, x)
            x = WS_PAT.sub(" ", x).strip()
        x = (
            x.replace("<num>", "<NUM>")
            .replace("<role>", "<ROLE>")
            .replace("<region>", "<REGION>")
            .replace("<domain>", "<DOMAIN>")
            .replace("<partner>", "<PARTNER>")
        )
        return x

    def _load_template_meta(self) -> None:
        if self._template_meta is not None:
            return

        root = Path(__file__).resolve().parents[2]
        catalog_path = root / "hrt-eth-zurich-datathon-2026" / "headlines" / "headline_template_catalog.csv"
        catalog = pd.read_csv(catalog_path)

        prior_sign_map = {"positive": 1.0, "negative": -1.0}
        self._template_meta = {
            str(row["template"]): (
                str(row["intent"]).strip().lower(),
                str(row["super_family"]).strip().lower(),
                float(prior_sign_map.get(str(row["direction_prior"]).strip().lower(), 0.0)),
            )
            for _, row in catalog.iterrows()
        }

    def _parse_headline_text(self, headline: str) -> tuple[str, str, str, float]:
        cached = self._headline_parse_cache.get(headline)
        if cached is not None:
            return cached

        self._load_template_meta()
        _, rest = self._split_company_and_rest(headline)
        template = self._canonicalize_rest(rest)
        intent, super_family, prior_sign = self._template_meta.get(template, ("unmapped", "unmapped", 0.0))
        parsed = (template, intent, super_family, prior_sign)
        self._headline_parse_cache[headline] = parsed
        return parsed

    def _parse_headlines(self, split: SplitInput) -> pd.DataFrame:
        sessions = split.sessions
        cols = ["session", "bar_ix", "headline", "template", "intent", "super_family", "prior_sign", "recency_exp_weight", "late", "recent3"]
        if split.headlines.empty:
            return pd.DataFrame(columns=cols)

        rows = split.headlines[["session", "bar_ix", "headline"]].copy()
        rows["headline"] = rows["headline"].astype(str)
        parsed = rows["headline"].map(self._parse_headline_text)
        rows["template"] = parsed.map(lambda x: x[0])
        rows["intent"] = parsed.map(lambda x: x[1])
        rows["super_family"] = parsed.map(lambda x: x[2])
        rows["prior_sign"] = parsed.map(lambda x: x[3]).astype(float)
        rows["late"] = (rows["bar_ix"] >= 40).astype(float)
        rows["recency_exp_weight"] = np.exp(self.recency_alpha * (rows["bar_ix"].astype(float) - 49.0))
        rows["headline_ix"] = rows.groupby("session", sort=False).cumcount()
        rows["headline_count_session"] = rows.groupby("session", sort=False)["headline"].transform("size")
        rows["recent3"] = (rows["headline_ix"] >= (rows["headline_count_session"] - 3).clip(lower=0)).astype(float)
        return rows[cols]

    @staticmethod
    def _smooth_group_stats(
        keys: pd.Series,
        target_return: pd.Series,
        up_label: pd.Series,
        bad_tail_label: pd.Series,
        global_return: float,
        global_up: float,
        global_bad_tail: float,
        smoothing: float,
    ) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "key": keys.astype(str),
                "target_return": target_return.astype(float),
                "up_label": up_label.astype(float),
                "bad_tail_label": bad_tail_label.astype(float),
            }
        )
        grouped = frame.groupby("key", sort=False).agg(
            count=("target_return", "size"),
            sum_return=("target_return", "sum"),
            sum_up=("up_label", "sum"),
            sum_bad_tail=("bad_tail_label", "sum"),
        )
        denom = grouped["count"] + smoothing
        grouped["enc_return"] = (grouped["sum_return"] + smoothing * global_return) / denom
        grouped["enc_up"] = (grouped["sum_up"] + smoothing * global_up) / denom
        grouped["enc_bad_tail"] = (grouped["sum_bad_tail"] + smoothing * global_bad_tail) / denom
        return grouped[["enc_return", "enc_up", "enc_bad_tail"]]

    @staticmethod
    def _lookup_with_fallback(
        rows: pd.DataFrame,
        template_table: pd.DataFrame,
        intent_table: pd.DataFrame,
        family_table: pd.DataFrame,
        global_value: float,
        col: str,
    ) -> pd.Series:
        template_map = rows["template"].map(template_table[col]) if not template_table.empty else pd.Series(np.nan, index=rows.index)
        intent_map = rows["intent"].map(intent_table[col]) if not intent_table.empty else pd.Series(np.nan, index=rows.index)
        family_map = rows["super_family"].map(family_table[col]) if not family_table.empty else pd.Series(np.nan, index=rows.index)
        return template_map.fillna(intent_map).fillna(family_map).fillna(float(global_value)).astype(float)

    @staticmethod
    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        return num.astype(float) / den.astype(float).clip(lower=1e-9)

    def _session_aggregate_features(self, rows: pd.DataFrame, sessions: pd.Index) -> pd.DataFrame:
        out = pd.DataFrame(index=sessions)
        base_cols = [
            "tpl_headline_count",
            "tpl_late_count",
            "tpl_recent3_count",
            "tpl_unmapped_share",
            "tpl_positive_prior_share",
            "tpl_negative_prior_share",
            "tpl_weight_sum",
            "tpl_enc_return_mean",
            "tpl_enc_return_last",
            "tpl_enc_return_recent3",
            "tpl_enc_return_late",
            "tpl_up_prob_mean",
            "tpl_bad_tail_prob_mean",
            "tpl_signal_balance",
            "tpl_positive_pressure",
            "tpl_negative_pressure",
            "tpl_recent3_positive_pressure",
            "tpl_recent3_negative_pressure",
            "tpl_strong_signal_count",
            "tpl_strong_signal_late_count",
        ]
        for col in base_cols:
            out[col] = 0.0

        if rows.empty:
            return out

        row_weight = rows["recency_exp_weight"].astype(float)
        headline_count = rows.groupby("session", sort=False).size().reindex(sessions, fill_value=0).astype(float)
        late_rows = rows[rows["late"] > 0]
        recent3_rows = rows[rows["recent3"] > 0]
        last_rows = rows.groupby("session", sort=False).tail(1).set_index("session")

        out["tpl_headline_count"] = headline_count
        out["tpl_late_count"] = late_rows.groupby("session", sort=False).size().reindex(sessions, fill_value=0).astype(float)
        out["tpl_recent3_count"] = recent3_rows.groupby("session", sort=False).size().reindex(sessions, fill_value=0).astype(float)
        out["tpl_unmapped_share"] = self._safe_div(
            rows.groupby("session", sort=False)["is_unmapped"].sum().reindex(sessions, fill_value=0.0),
            headline_count,
        ).astype(float)
        out["tpl_positive_prior_share"] = self._safe_div(
            rows.groupby("session", sort=False)["is_positive_prior"].sum().reindex(sessions, fill_value=0.0),
            headline_count,
        ).astype(float)
        out["tpl_negative_prior_share"] = self._safe_div(
            rows.groupby("session", sort=False)["is_negative_prior"].sum().reindex(sessions, fill_value=0.0),
            headline_count,
        ).astype(float)

        weight_sum = rows.groupby("session", sort=False)["recency_exp_weight"].sum().reindex(sessions, fill_value=0.0)
        out["tpl_weight_sum"] = weight_sum.astype(float)

        weighted_return = (
            rows["enc_return"] * row_weight
        ).groupby(rows["session"], sort=False).sum().reindex(sessions, fill_value=0.0)
        weighted_up = (
            rows["enc_up"] * row_weight
        ).groupby(rows["session"], sort=False).sum().reindex(sessions, fill_value=0.0)
        weighted_bad = (
            rows["enc_bad_tail"] * row_weight
        ).groupby(rows["session"], sort=False).sum().reindex(sessions, fill_value=0.0)

        out["tpl_enc_return_mean"] = self._safe_div(weighted_return, weight_sum).astype(float)
        out["tpl_up_prob_mean"] = self._safe_div(weighted_up, weight_sum).astype(float)
        out["tpl_bad_tail_prob_mean"] = self._safe_div(weighted_bad, weight_sum).astype(float)
        out["tpl_signal_balance"] = (out["tpl_up_prob_mean"] - out["tpl_bad_tail_prob_mean"]).astype(float)

        out["tpl_enc_return_last"] = last_rows["enc_return"].reindex(sessions).fillna(0.0).astype(float)
        out["tpl_enc_return_recent3"] = recent3_rows.groupby("session", sort=False)["enc_return"].mean().reindex(sessions, fill_value=0.0).astype(float)
        out["tpl_enc_return_late"] = late_rows.groupby("session", sort=False)["enc_return"].mean().reindex(sessions, fill_value=0.0).astype(float)

        out["tpl_positive_pressure"] = (
            (rows["enc_return"].clip(lower=0.0) * row_weight).groupby(rows["session"], sort=False).sum().reindex(sessions, fill_value=0.0).astype(float)
        )
        out["tpl_negative_pressure"] = (
            ((-rows["enc_return"].clip(upper=0.0)) * row_weight).groupby(rows["session"], sort=False).sum().reindex(sessions, fill_value=0.0).astype(float)
        )
        out["tpl_recent3_positive_pressure"] = (
            (recent3_rows["enc_return"].clip(lower=0.0) * recent3_rows["recency_exp_weight"])
            .groupby(recent3_rows["session"], sort=False)
            .sum()
            .reindex(sessions, fill_value=0.0)
            .astype(float)
        )
        out["tpl_recent3_negative_pressure"] = (
            ((-recent3_rows["enc_return"].clip(upper=0.0)) * recent3_rows["recency_exp_weight"])
            .groupby(recent3_rows["session"], sort=False)
            .sum()
            .reindex(sessions, fill_value=0.0)
            .astype(float)
        )

        out["tpl_strong_signal_count"] = rows.groupby("session", sort=False)["strong_signal"].sum().reindex(sessions, fill_value=0.0).astype(float)
        out["tpl_strong_signal_late_count"] = late_rows.groupby("session", sort=False)["strong_signal"].sum().reindex(sessions, fill_value=0.0).astype(float)

        return out.fillna(0.0)

    def _apply_tables(self, rows: pd.DataFrame, tables: dict[str, pd.DataFrame | float]) -> pd.DataFrame:
        if rows.empty:
            return rows.assign(
                enc_return=pd.Series(dtype=float),
                enc_up=pd.Series(dtype=float),
                enc_bad_tail=pd.Series(dtype=float),
                is_unmapped=pd.Series(dtype=float),
                is_positive_prior=pd.Series(dtype=float),
                is_negative_prior=pd.Series(dtype=float),
                strong_signal=pd.Series(dtype=float),
            )

        enriched = rows.copy()
        template_table = tables["template"]  # type: ignore[assignment]
        intent_table = tables["intent"]  # type: ignore[assignment]
        family_table = tables["super_family"]  # type: ignore[assignment]

        enriched["enc_return"] = self._lookup_with_fallback(
            enriched,
            template_table,
            intent_table,
            family_table,
            float(tables["global_return"]),
            "enc_return",
        )
        enriched["enc_up"] = self._lookup_with_fallback(
            enriched,
            template_table,
            intent_table,
            family_table,
            float(tables["global_up"]),
            "enc_up",
        )
        enriched["enc_bad_tail"] = self._lookup_with_fallback(
            enriched,
            template_table,
            intent_table,
            family_table,
            float(tables["global_bad_tail"]),
            "enc_bad_tail",
        )

        enriched["is_unmapped"] = (enriched["intent"] == "unmapped").astype(float)
        enriched["is_positive_prior"] = (enriched["prior_sign"] > 0).astype(float)
        enriched["is_negative_prior"] = (enriched["prior_sign"] < 0).astype(float)
        enriched["strong_signal"] = (
            (enriched["enc_up"] >= 0.60)
            | (enriched["enc_bad_tail"] >= 0.60)
            | (enriched["enc_return"].abs() >= 0.004)
        ).astype(float)
        return enriched

    def _build_tables_for_rows(self, rows: pd.DataFrame, session_target: pd.Series, bad_tail_cutoff: float) -> dict[str, pd.DataFrame | float]:
        train_rows = rows.copy()
        target_map = session_target.astype(float)
        train_rows["target_return"] = train_rows["session"].map(target_map)
        train_rows["up_label"] = (train_rows["target_return"] > 0.0).astype(float)
        train_rows["bad_tail_label"] = (train_rows["target_return"] <= bad_tail_cutoff).astype(float)

        global_return = float(session_target.mean())
        global_up = float((session_target > 0.0).mean())
        global_bad_tail = float((session_target <= bad_tail_cutoff).mean())

        return {
            "template": self._smooth_group_stats(
                train_rows["template"],
                train_rows["target_return"],
                train_rows["up_label"],
                train_rows["bad_tail_label"],
                global_return,
                global_up,
                global_bad_tail,
                self.template_smoothing,
            ),
            "intent": self._smooth_group_stats(
                train_rows["intent"],
                train_rows["target_return"],
                train_rows["up_label"],
                train_rows["bad_tail_label"],
                global_return,
                global_up,
                global_bad_tail,
                self.template_smoothing,
            ),
            "super_family": self._smooth_group_stats(
                train_rows["super_family"],
                train_rows["target_return"],
                train_rows["up_label"],
                train_rows["bad_tail_label"],
                global_return,
                global_up,
                global_bad_tail,
                self.template_smoothing,
            ),
            "global_return": global_return,
            "global_up": global_up,
            "global_bad_tail": global_bad_tail,
        }

    def _fit_template_features(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        rows = self._parse_headlines(train_split)
        sessions = train_split.sessions
        y = train_target_return.reindex(sessions).astype(float)
        bad_tail_cutoff = float(np.quantile(y, self.template_bad_tail_quantile))

        if rows.empty:
            self._train_template_features = self._session_aggregate_features(rows, sessions)
            self._template_feature_cache = {"train_seen": self._train_template_features}
            self._full_template_tables = self._build_tables_for_rows(rows, y, bad_tail_cutoff)
            return

        oof_rows: list[pd.DataFrame] = []
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, valid_idx in splitter.split(sessions):
            train_sessions = sessions[train_idx]
            valid_sessions = sessions[valid_idx]
            fold_y = y.reindex(train_sessions)
            fold_rows = rows[rows["session"].isin(train_sessions)]
            valid_rows = rows[rows["session"].isin(valid_sessions)]
            fold_tables = self._build_tables_for_rows(fold_rows, fold_y, bad_tail_cutoff)
            oof_rows.append(self._apply_tables(valid_rows, fold_tables))

        oof_scored = pd.concat(oof_rows, ignore_index=True) if oof_rows else rows.copy()
        self._train_template_features = self._session_aggregate_features(oof_scored, sessions)
        self._template_feature_cache = {"train_seen": self._train_template_features}
        self._full_template_tables = self._build_tables_for_rows(rows, y, bad_tail_cutoff)

    def _build_template_features(self, split: SplitInput) -> pd.DataFrame:
        cached = self._template_feature_cache.get(split.name)
        if cached is not None:
            return cached.reindex(split.sessions).fillna(0.0)

        rows = self._parse_headlines(split)
        scored_rows = self._apply_tables(rows, self._full_template_tables)
        features = self._session_aggregate_features(scored_rows, split.sessions)
        self._template_feature_cache[split.name] = features
        return features

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        self._fit_template_features(train_split, train_target_return)
        super().fit(train_split, train_target_return)

    def _build_X(self, split: SplitInput) -> pd.DataFrame:
        risk_x = build_risk_features(split)
        template_x = self._build_template_features(split)
        return risk_x.join(template_x).replace([np.inf, -np.inf], np.nan)


def build_strategy() -> BtpRankTplStrategy:
    return BtpRankTplStrategy()
