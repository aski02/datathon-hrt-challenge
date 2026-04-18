from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.types import SplitInput

"""
Based on always_long_239
+ small headline risk penalty: counts of "risk" patterns in headline text, with a negative weight.
The headline risk patterns are:
- "class action"
- "loses key contract"
- "scheduled maintenance"
- "decline" (as a whole word, to avoid false positives like "declined" or "declining")

+ small positive headline feature: counts of "expands" in headline text, with a small positive weight.

+ template-catalog prior integration:
uses `headline_template_catalog.csv` to assign each headline a prior (positive/negative/neutral) based on its template, 
and computes a session-level balance of positive vs negative priors to adjust position sizing.
Prior is semantic, decided by codex-based review of templates, not data-driven.
"""


class AlwaysLong239HeadlineV3TemplatePriorStrategy:
    """always_long_239 with template-catalog prior integration.

    Uses `headline_template_catalog.csv` to map each headline to a template prior:
    positive / negative / neutral.
    Then increases long exposure when a session has more positive than negative
    priors, and reduces exposure in the opposite case.
    """

    name = "always-long-239-h3-templateprior"

    def __init__(self) -> None:
        # Base from always_long_239.
        self.base_position = 1.0
        self.w_vol = 0.15
        self.w_ret = 0.08
        self.w_trend = 0.03

        # Existing conservative headline features from h2.
        self.w_headline_risk = -0.03
        self.w_headline_positive = 0.015

        # New: template-catalog prior balance signal.
        self.w_template_prior = 0.05

        self.min_pos = 0.25
        self.max_pos = 1.75

        self._mu_vol: float | None = None
        self._sd_vol: float | None = None
        self._mu_ret: float | None = None
        self._sd_ret: float | None = None
        self._mu_trend: float | None = None
        self._sd_trend: float | None = None
        self._mu_headline_risk: float | None = None
        self._sd_headline_risk: float | None = None
        self._mu_headline_positive: float | None = None
        self._sd_headline_positive: float | None = None
        self._mu_template_prior: float | None = None
        self._sd_template_prior: float | None = None

        self._template_patterns: list[tuple[re.Pattern[str], str]] = []
        self._headline_prior_cache: dict[str, str] = {}

    @staticmethod
    def _safe_std(x: pd.Series) -> float:
        std = float(x.std(ddof=0))
        return std if std > 1e-12 else 1.0

    @staticmethod
    def _headline_risk_score(headline_text: pd.Series) -> pd.Series:
        low = headline_text.fillna("").str.lower()
        return (
            low.str.count(r"class action")
            + low.str.count(r"loses key contract")
            + low.str.count(r"scheduled maintenance")
            + low.str.count(r"\bdecline\b")
        ).astype(float)

    @staticmethod
    def _headline_positive_score(headline_text: pd.Series) -> pd.Series:
        low = headline_text.fillna("").str.lower()
        return low.str.count(r"\bexpands\b").astype(float)

    @staticmethod
    def _template_body_regex(template: str) -> str:
        placeholder_patterns = {
            "<NUM>": r"[-+]?\$?\d+(?:\.\d+)?(?:[MB%])?(?:\s*(?:billion|million|year-over-year))?",
            "<REGION>": r".+?",
            "<DOMAIN>": r".+?",
            "<PARTNER>": r".+?",
            "<ROLE>": r".+?",
        }
        parts = re.split(r"(<[^>]+>)", template)
        out: list[str] = []
        for part in parts:
            if not part:
                continue
            if part.startswith("<") and part.endswith(">"):
                out.append(placeholder_patterns.get(part, r".+?"))
            else:
                out.append(re.escape(part))
        return "".join(out)

    def _load_template_patterns(self) -> None:
        if self._template_patterns:
            return

        root = Path(__file__).resolve().parents[2]
        catalog_path = root / "hrt-eth-zurich-datathon-2026" / "headlines" / "headline_template_catalog.csv"
        catalog = pd.read_csv(catalog_path)
        prior_col = "prediction_prior" if "prediction_prior" in catalog.columns else "direction_prior"

        patterns: list[tuple[re.Pattern[str], str]] = []
        for row in catalog.itertuples(index=False):
            body = self._template_body_regex(getattr(row, "template"))
            pattern = re.compile(r"^.+?\s+" + body + r"$", re.IGNORECASE)
            prior = str(getattr(row, prior_col))
            if prior == "neutral_or_event":
                prior = "neutral"
            patterns.append((pattern, prior))

        self._template_patterns = patterns

    def _headline_prior_from_text(self, text: str) -> str:
        cached = self._headline_prior_cache.get(text)
        if cached is not None:
            return cached
        prior = "neutral"
        for pattern, p in self._template_patterns:
            if pattern.match(text):
                prior = p
                break
        self._headline_prior_cache[text] = prior
        return prior

    def _session_template_prior_balance(self, split: SplitInput) -> pd.Series:
        self._load_template_patterns()
        headlines = split.headlines[["session", "headline"]].copy()
        headlines["prior"] = headlines["headline"].astype(str).map(self._headline_prior_from_text)

        counts = headlines.groupby(["session", "prior"]).size().unstack(fill_value=0)
        for col in ("positive", "negative", "neutral"):
            if col not in counts.columns:
                counts[col] = 0
        counts = counts[["positive", "negative", "neutral"]]
        total = counts.sum(axis=1).clip(lower=1.0)
        balance = (counts["positive"] - counts["negative"]) / np.sqrt(total)
        return balance.rename("template_prior_balance")

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        _ = train_target_return
        f = train_split.features

        vol = f["close_std_seen"].fillna(f["close_std_seen"].median())
        ret = f["ret_full_seen"].fillna(0.0)
        trend = f["trend_slope"].fillna(0.0)
        headline_risk = self._headline_risk_score(f["headline_text"])
        headline_positive = self._headline_positive_score(f["headline_text"])
        template_prior_balance = self._session_template_prior_balance(train_split).reindex(train_split.sessions).fillna(0.0)

        self._mu_vol = float(vol.mean())
        self._sd_vol = self._safe_std(vol)
        self._mu_ret = float(ret.mean())
        self._sd_ret = self._safe_std(ret)
        self._mu_trend = float(trend.mean())
        self._sd_trend = self._safe_std(trend)
        self._mu_headline_risk = float(headline_risk.mean())
        self._sd_headline_risk = self._safe_std(headline_risk)
        self._mu_headline_positive = float(headline_positive.mean())
        self._sd_headline_positive = self._safe_std(headline_positive)
        self._mu_template_prior = float(template_prior_balance.mean())
        self._sd_template_prior = self._safe_std(template_prior_balance)

    def predict(self, split: SplitInput) -> pd.Series:
        if any(
            value is None
            for value in (
                self._mu_vol,
                self._sd_vol,
                self._mu_ret,
                self._sd_ret,
                self._mu_trend,
                self._sd_trend,
                self._mu_headline_risk,
                self._sd_headline_risk,
                self._mu_headline_positive,
                self._sd_headline_positive,
                self._mu_template_prior,
                self._sd_template_prior,
            )
        ):
            raise RuntimeError("Strategy must be fit before predict.")

        f = split.features
        vol = f["close_std_seen"].fillna(self._mu_vol).to_numpy(dtype=float)
        ret = f["ret_full_seen"].fillna(0.0).to_numpy(dtype=float)
        trend = f["trend_slope"].fillna(0.0).to_numpy(dtype=float)
        headline_risk = self._headline_risk_score(f["headline_text"]).to_numpy(dtype=float)
        headline_positive = self._headline_positive_score(f["headline_text"]).to_numpy(dtype=float)
        template_prior_balance = self._session_template_prior_balance(split).reindex(split.sessions).fillna(0.0).to_numpy(dtype=float)

        z_vol = (vol - self._mu_vol) / self._sd_vol
        z_ret = (ret - self._mu_ret) / self._sd_ret
        z_trend = (trend - self._mu_trend) / self._sd_trend
        z_headline_risk = (headline_risk - self._mu_headline_risk) / self._sd_headline_risk
        z_headline_positive = (headline_positive - self._mu_headline_positive) / self._sd_headline_positive
        z_template_prior = (template_prior_balance - self._mu_template_prior) / self._sd_template_prior

        raw = (
            self.base_position
            - self.w_vol * z_vol
            - self.w_ret * z_ret
            - self.w_trend * z_trend
            + self.w_headline_risk * z_headline_risk
            + self.w_headline_positive * z_headline_positive
            + self.w_template_prior * z_template_prior
        )
        positions = np.clip(raw, self.min_pos, self.max_pos)
        return pd.Series(positions, index=split.sessions, name="target_position")


def build_strategy() -> AlwaysLong239HeadlineV3TemplatePriorStrategy:
    return AlwaysLong239HeadlineV3TemplatePriorStrategy()
