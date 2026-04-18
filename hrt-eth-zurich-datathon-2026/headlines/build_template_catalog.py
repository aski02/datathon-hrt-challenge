#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


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
    (
        re.compile(r"\bfor\s+regulatory\s+approval\s+of\s+new\s+.+\s+offering\b", re.I),
        "for regulatory approval of new <DOMAIN> offering",
    ),
    (
        re.compile(r"\bin\s+.+\s+line\s+due\s+to\s+quality\s+concerns\b", re.I),
        "in <DOMAIN> line due to quality concerns",
    ),
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
    (
        re.compile(r"\bannounces\s+significant\s+capital\s+expenditure\s+plan\s+for\s+.+$", re.I),
        "announces significant capital expenditure plan for <REGION>",
    ),
    (
        re.compile(r"\breports\s+strong\s+demand\s+in\s+.+,\s+raises\s+outlook\b", re.I),
        "reports strong demand in <REGION>, raises outlook",
    ),
    (
        re.compile(r"\bwarns\s+of\s+supply\s+chain\s+disruptions\s+affecting\s+.+\s+operations\b", re.I),
        "warns of supply chain disruptions affecting <REGION> operations",
    ),
    (re.compile(r"\bcompletes\s+planned\s+facility\s+upgrade\s+in\s+.+$", re.I), "completes planned facility upgrade in <REGION>"),
    (re.compile(r"\bloses\s+key\s+contract\s+in\s+.+\s+to\s+competitor\b", re.I), "loses key contract in <REGION> to competitor"),
    (re.compile(r"\breports\s+unexpected\s+decline\s+in\s+.+\s+revenue\b", re.I), "reports unexpected decline in <REGION> revenue"),
    (
        re.compile(r"\bwithdraws\s+from\s+.+\s+market\s+citing\s+unfavorable\s+conditions\b", re.I),
        "withdraws from <REGION> market citing unfavorable conditions",
    ),
    (re.compile(r"\bappoints\s+new\s+.+\s+to\s+board\b", re.I), "appoints new <ROLE> to board"),
    (re.compile(r"\bnames\s+new\s+head\s+of\s+.+\s+division\b", re.I), "names new head of <DOMAIN> division"),
    (
        re.compile(r"\b(?:ceo|cfo|cto|chief\s+[a-z]+\s+officer)\s+steps\s+down\s+unexpectedly\s+citing\s+personal\s+reasons\b", re.I),
        "<ROLE> steps down unexpectedly citing personal reasons",
    ),
    (
        re.compile(r"\b(?:ceo|cfo|cto|chief\s+[a-z]+\s+officer)\s+addresses\s+investor\s+concerns\s+in\s+open\s+letter\b", re.I),
        "<ROLE> addresses investor concerns in open letter",
    ),
]

INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("contract_win", re.compile(r"^secures <NUM> contract with <PARTNER>$")),
    ("contract_loss_region", re.compile(r"^loses key contract in <REGION> to competitor$")),
    ("partnership", re.compile(r"^signs multi-year partnership with <PARTNER>$")),
    ("joint_venture", re.compile(r"^enters joint venture with <PARTNER>$")),
    ("launch_delay", re.compile(r"^delays product launch in <DOMAIN> segment$")),
    ("product_breakthrough", re.compile(r"^announces breakthrough in <DOMAIN>$")),
    ("next_gen_launch", re.compile(r"^launches next-generation <DOMAIN> platform$")),
    ("regulatory_review", re.compile(r"^faces regulatory review of <DOMAIN> practices$")),
    ("regulatory_approval_filing", re.compile(r"^files for regulatory approval of new <DOMAIN> offering$")),
    ("regulatory_milestone", re.compile(r"^achieves key regulatory milestone ahead of schedule$")),
    ("legal_class_action", re.compile(r"^faces class action over <DOMAIN> service disruption$")),
    ("patent_filing", re.compile(r"^files routine patent applications in <DOMAIN>$")),
    ("quality_recall", re.compile(r"^recalls products in <DOMAIN> line due to quality concerns$")),
    ("scheduled_maintenance", re.compile(r"^begins scheduled maintenance of <DOMAIN> systems$")),
    ("award_recognition", re.compile(r"^wins industry award for excellence in <DOMAIN>$")),
    ("revenue_record_up", re.compile(r"^reports record quarterly revenue, up <NUM> year-over-year$")),
    ("revenue_miss", re.compile(r"^misses quarterly revenue estimates by <NUM>$")),
    ("customer_acq_up", re.compile(r"^reports <NUM> increase in customer acquisition$")),
    ("new_orders_down", re.compile(r"^sees <NUM> drop in new customer orders this quarter$")),
    ("operating_income_down", re.compile(r"^reports <NUM> decline in operating income$")),
    ("margin_improvement", re.compile(r"^sees <NUM> margin improvement in latest quarter$")),
    ("margin_pressure", re.compile(r"^reports rising costs pressuring margins in <DOMAIN>$")),
    ("demand_strong_raise_outlook", re.compile(r"^reports strong demand in <REGION>, raises outlook$")),
    ("guidance_raise", re.compile(r"^raises full-year guidance citing robust demand$")),
    ("guidance_lower", re.compile(r"^lowers full-year guidance amid softening demand$")),
    ("region_revenue_decline", re.compile(r"^reports unexpected decline in <REGION> revenue$")),
    ("share_buyback", re.compile(r"^announces <NUM> share buyback program$")),
    ("capex_region", re.compile(r"^announces significant capital expenditure plan for <REGION>$")),
    ("expansion_region", re.compile(r"^expands operations into <REGION> markets$")),
    ("office_opening_region", re.compile(r"^opens new office in <REGION>$")),
    ("facility_upgrade_region", re.compile(r"^completes planned facility upgrade in <REGION>$")),
    ("withdraw_region", re.compile(r"^withdraws from <REGION> market citing unfavorable conditions$")),
    ("supply_chain_disruption_region", re.compile(r"^warns of supply chain disruptions affecting <REGION> operations$")),
    ("strategic_acquisition", re.compile(r"^completes strategic acquisition to strengthen (?:<DOMAIN>|data analytics)$")),
    ("merger_talks", re.compile(r"^in talks for potential merger, details undisclosed$")),
    ("strategy_focus_domain", re.compile(r"^revises long-term strategy with focus on <DOMAIN>$")),
    ("strategic_alternatives", re.compile(r"^explores strategic alternatives for <DOMAIN> unit$")),
    ("investor_day_focus_domain", re.compile(r"^to host investor day focused on (?:<DOMAIN>|data analytics) strategy$")),
    ("major_restructuring", re.compile(r"^announces major organizational restructuring$")),
    ("restructuring_plan", re.compile(r"^announces restructuring plan, cites challenging market conditions$")),
    ("division_leadership_change", re.compile(r"^names new head of <DOMAIN> division$")),
    ("exec_steps_down", re.compile(r"^<ROLE> steps down unexpectedly citing personal reasons$")),
    ("exec_investor_letter", re.compile(r"^<ROLE> addresses investor concerns in open letter$")),
    ("earnings_beat", re.compile(r"^beats analyst expectations with strong earnings growth$")),
    ("event_present", re.compile(r"^to present at .+$")),
    ("event_confirm", re.compile(r"^confirms participation in .+$")),
    ("shareholder_meeting", re.compile(r"^schedules annual shareholder meeting for next month$")),
    ("sustainability_report", re.compile(r"^publishes annual sustainability report$")),
    ("pilot_mixed", re.compile(r"^sees mixed results in <DOMAIN> pilot program$")),
    ("board_meeting_strategy", re.compile(r"^board meeting to discuss major strategic initiative$")),
]

SUPER_FAMILY = {
    "contract_win": "commercial_deals",
    "contract_loss_region": "commercial_deals",
    "partnership": "commercial_deals",
    "joint_venture": "commercial_deals",
    "launch_delay": "product_technology",
    "product_breakthrough": "product_technology",
    "next_gen_launch": "product_technology",
    "quality_recall": "product_technology",
    "scheduled_maintenance": "product_technology",
    "pilot_mixed": "product_technology",
    "award_recognition": "product_technology",
    "patent_filing": "product_technology",
    "regulatory_review": "regulatory_legal",
    "regulatory_approval_filing": "regulatory_legal",
    "regulatory_milestone": "regulatory_legal",
    "legal_class_action": "regulatory_legal",
    "revenue_record_up": "financial_performance",
    "revenue_miss": "financial_performance",
    "customer_acq_up": "financial_performance",
    "new_orders_down": "financial_performance",
    "operating_income_down": "financial_performance",
    "margin_improvement": "financial_performance",
    "margin_pressure": "financial_performance",
    "earnings_beat": "financial_performance",
    "region_revenue_decline": "financial_performance",
    "guidance_raise": "guidance_capital",
    "guidance_lower": "guidance_capital",
    "share_buyback": "guidance_capital",
    "capex_region": "guidance_capital",
    "demand_strong_raise_outlook": "guidance_capital",
    "expansion_region": "geo_operations",
    "office_opening_region": "geo_operations",
    "facility_upgrade_region": "geo_operations",
    "withdraw_region": "geo_operations",
    "supply_chain_disruption_region": "geo_operations",
    "strategic_acquisition": "strategy_ma_reorg",
    "merger_talks": "strategy_ma_reorg",
    "strategy_focus_domain": "strategy_ma_reorg",
    "strategic_alternatives": "strategy_ma_reorg",
    "major_restructuring": "strategy_ma_reorg",
    "restructuring_plan": "strategy_ma_reorg",
    "board_meeting_strategy": "strategy_ma_reorg",
    "division_leadership_change": "leadership_governance_ir",
    "exec_steps_down": "leadership_governance_ir",
    "exec_investor_letter": "leadership_governance_ir",
    "investor_day_focus_domain": "leadership_governance_ir",
    "event_present": "leadership_governance_ir",
    "event_confirm": "leadership_governance_ir",
    "shareholder_meeting": "leadership_governance_ir",
    "sustainability_report": "leadership_governance_ir",
}

DIRECTION_PRIOR = {
    "contract_win": "positive",
    "partnership": "positive",
    "joint_venture": "positive",
    "product_breakthrough": "positive",
    "next_gen_launch": "positive",
    "regulatory_milestone": "positive",
    "award_recognition": "positive",
    "revenue_record_up": "positive",
    "customer_acq_up": "positive",
    "margin_improvement": "positive",
    "demand_strong_raise_outlook": "positive",
    "guidance_raise": "positive",
    "share_buyback": "positive",
    "expansion_region": "positive",
    "office_opening_region": "positive",
    "facility_upgrade_region": "positive",
    "earnings_beat": "positive",
    "launch_delay": "negative",
    "regulatory_review": "negative",
    "legal_class_action": "negative",
    "quality_recall": "negative",
    "revenue_miss": "negative",
    "new_orders_down": "negative",
    "operating_income_down": "negative",
    "margin_pressure": "negative",
    "guidance_lower": "negative",
    "region_revenue_decline": "negative",
    "withdraw_region": "negative",
    "supply_chain_disruption_region": "negative",
    "contract_loss_region": "negative",
    "exec_steps_down": "negative",
}


def split_company_and_rest(headline: str) -> tuple[str, str]:
    parts = str(headline).split()
    company = " ".join(parts[:2]) if len(parts) >= 2 else str(headline)
    rest = " ".join(parts[2:]) if len(parts) >= 2 else ""
    return company, rest


def normalize_numbers(text: str) -> str:
    x = str(text).lower().strip()
    x = YEAR_PAT.sub("<YEAR>", x)
    x = NUM_PAT.sub("<NUM>", x)
    return WS_PAT.sub(" ", x).strip()


def canonicalize_rest(rest: str) -> str:
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


def map_intent(template: str) -> str:
    for name, pat in INTENT_RULES:
        if pat.match(template):
            return name
    return "unmapped"


def build_dataset(data_dir: Path) -> pd.DataFrame:
    specs = [
        ("headlines_seen_train.parquet", "train", "seen", "train_seen"),
        ("headlines_unseen_train.parquet", "train", "unseen", "train_unseen"),
        ("headlines_seen_public_test.parquet", "test", "seen", "test_public_seen"),
        ("headlines_seen_private_test.parquet", "test", "seen", "test_private_seen"),
    ]
    frames: list[pd.DataFrame] = []
    for filename, split, visibility, dataset in specs:
        path = data_dir / filename
        df = pd.read_parquet(path)[["session", "bar_ix", "headline"]].copy()
        df["file"] = filename
        df["split"] = split
        df["visibility"] = visibility
        df["dataset"] = dataset
        company_rest = df["headline"].map(split_company_and_rest)
        df["company"] = company_rest.map(lambda x: x[0])
        df["rest"] = company_rest.map(lambda x: x[1])
        df["raw_template"] = df["rest"].map(normalize_numbers)
        df["template"] = df["rest"].map(canonicalize_rest)
        df["intent"] = df["template"].map(map_intent)
        df["super_family"] = df["intent"].map(lambda x: SUPER_FAMILY.get(x, "unmapped"))
        df["direction_prior"] = df["intent"].map(lambda x: DIRECTION_PRIOR.get(x, "neutral_or_event"))
        df["session_uid"] = df["file"] + ":" + df["session"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def join_top_values(df: pd.DataFrame, group_col: str, value_col: str, n: int = 3) -> pd.Series:
    out = {}
    for key, g in df.groupby(group_col):
        top_vals = g[value_col].value_counts().head(n).index.tolist()
        out[key] = " || ".join(str(v) for v in top_vals)
    return pd.Series(out, name=f"top_{value_col}_{n}")


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den_safe = den.replace(0, np.nan)
    return (num / den_safe).fillna(0.0)


def build_catalog(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)

    base = df.groupby("template", as_index=False).agg(
        total_rows=("headline", "size"),
        unique_companies=("company", "nunique"),
        unique_raw_templates=("raw_template", "nunique"),
        unique_headlines=("headline", "nunique"),
        unique_sessions=("session_uid", "nunique"),
        mean_bar_ix=("bar_ix", "mean"),
        median_bar_ix=("bar_ix", "median"),
        p10_bar_ix=("bar_ix", lambda s: float(np.percentile(s.to_numpy(), 10))),
        p90_bar_ix=("bar_ix", lambda s: float(np.percentile(s.to_numpy(), 90))),
    )
    base["share_of_all_rows"] = base["total_rows"] / float(total_rows)

    for col in ["intent", "super_family", "direction_prior"]:
        base = base.merge(
            df.groupby("template")[col].agg(lambda s: s.mode().iloc[0]).rename(col),
            on="template",
            how="left",
        )

    rows_by_dataset = (
        df.groupby(["template", "dataset"]).size().unstack(fill_value=0).reset_index()
    )
    for c in ["train_seen", "train_unseen", "test_public_seen", "test_private_seen"]:
        if c not in rows_by_dataset.columns:
            rows_by_dataset[c] = 0
    base = base.merge(rows_by_dataset, on="template", how="left")

    rows_by_visibility = (
        df.groupby(["template", "visibility"]).size().unstack(fill_value=0).reset_index()
    )
    for c in ["seen", "unseen"]:
        if c not in rows_by_visibility.columns:
            rows_by_visibility[c] = 0
    rows_by_visibility = rows_by_visibility.rename(columns={"seen": "rows_seen", "unseen": "rows_unseen"})
    base = base.merge(rows_by_visibility, on="template", how="left")

    rows_by_split = df.groupby(["template", "split"]).size().unstack(fill_value=0).reset_index()
    for c in ["train", "test"]:
        if c not in rows_by_split.columns:
            rows_by_split[c] = 0
    rows_by_split = rows_by_split.rename(columns={"train": "rows_train", "test": "rows_test"})
    base = base.merge(rows_by_split, on="template", how="left")

    late_seen = (
        df[(df["visibility"] == "seen") & (df["bar_ix"] >= 40)]
        .groupby("template")
        .size()
        .rename("rows_seen_late_40_49")
    )
    early_seen = (
        df[(df["visibility"] == "seen") & (df["bar_ix"] <= 9)]
        .groupby("template")
        .size()
        .rename("rows_seen_early_0_9")
    )
    base = base.merge(late_seen, on="template", how="left")
    base = base.merge(early_seen, on="template", how="left")
    base["rows_seen_late_40_49"] = base["rows_seen_late_40_49"].fillna(0).astype(int)
    base["rows_seen_early_0_9"] = base["rows_seen_early_0_9"].fillna(0).astype(int)
    base["late_seen_share"] = safe_ratio(base["rows_seen_late_40_49"], base["rows_seen"])
    base["early_seen_share"] = safe_ratio(base["rows_seen_early_0_9"], base["rows_seen"])
    base["seen_share"] = safe_ratio(base["rows_seen"], base["total_rows"])
    base["unseen_share"] = safe_ratio(base["rows_unseen"], base["total_rows"])
    base["train_share"] = safe_ratio(base["rows_train"], base["total_rows"])
    base["test_share"] = safe_ratio(base["rows_test"], base["total_rows"])

    per_session = (
        df.groupby(["template", "session_uid"]).size().rename("rows_in_session").reset_index()
    )
    per_session_stats = per_session.groupby("template", as_index=False).agg(
        mean_rows_per_session=("rows_in_session", "mean"),
        median_rows_per_session=("rows_in_session", "median"),
        max_rows_in_single_session=("rows_in_session", "max"),
    )
    base = base.merge(per_session_stats, on="template", how="left")

    top_headlines = join_top_values(df, "template", "headline", n=3)
    top_companies = join_top_values(df, "template", "company", n=3)
    base = base.merge(top_headlines, left_on="template", right_index=True, how="left")
    base = base.merge(top_companies, left_on="template", right_index=True, how="left")
    base = base.rename(
        columns={
            "top_headline_3": "example_headlines_top3",
            "top_company_3": "example_companies_top3",
        }
    )

    base = base.sort_values(["total_rows", "template"], ascending=[False, True]).reset_index(drop=True)
    base.insert(0, "template_id", [f"T{i:03d}" for i in range(1, len(base) + 1)])
    return base


def write_summary(df: pd.DataFrame, catalog: pd.DataFrame, out_path: Path) -> None:
    total_rows = len(df)
    n_templates = catalog["template"].nunique()
    n_intents = catalog["intent"].nunique()
    n_super = catalog["super_family"].nunique()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    top_templates = catalog.head(12)[["template", "total_rows", "share_of_all_rows"]]
    super_counts = (
        catalog.groupby("super_family", as_index=False)["total_rows"].sum().sort_values("total_rows", ascending=False)
    )
    super_counts["share"] = super_counts["total_rows"] / float(total_rows)

    direction_counts = (
        catalog.groupby("direction_prior", as_index=False)["total_rows"].sum().sort_values("total_rows", ascending=False)
    )
    direction_counts["share"] = direction_counts["total_rows"] / float(total_rows)

    lines: list[str] = []
    lines.append("# Headline Template Catalog Summary")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Sources: `headlines_seen_train`, `headlines_unseen_train`, `headlines_seen_public_test`, `headlines_seen_private_test`.")
    lines.append("- Company tokens (first 2 words) are treated as non-structural features.")
    lines.append("- Templates are canonicalized by replacing numbers, regions, partner names, roles, and domain phrases with placeholders.")
    lines.append("")
    lines.append("## Key Numbers")
    lines.append("")
    lines.append(f"- Total headline rows: **{total_rows:,}**")
    lines.append(f"- Canonical templates: **{n_templates}**")
    lines.append(f"- Intents: **{n_intents}**")
    lines.append(f"- Super-families: **{n_super}**")
    lines.append("")
    lines.append("## Top Templates")
    lines.append("")
    lines.append("| Template | Rows | Share |")
    lines.append("|---|---:|---:|")
    for _, r in top_templates.iterrows():
        lines.append(f"| `{r['template']}` | {int(r['total_rows']):,} | {float(r['share_of_all_rows']) * 100:.2f}% |")
    lines.append("")
    lines.append("## Super-Family Distribution")
    lines.append("")
    lines.append("| Super-family | Rows | Share |")
    lines.append("|---|---:|---:|")
    for _, r in super_counts.iterrows():
        lines.append(f"| `{r['super_family']}` | {int(r['total_rows']):,} | {float(r['share']) * 100:.2f}% |")
    lines.append("")
    lines.append("## Direction Prior Distribution")
    lines.append("")
    lines.append("| Direction prior | Rows | Share |")
    lines.append("|---|---:|---:|")
    for _, r in direction_counts.iterrows():
        lines.append(f"| `{r['direction_prior']}` | {int(r['total_rows']):,} | {float(r['share']) * 100:.2f}% |")
    lines.append("")
    lines.append("## How To Use This Catalog")
    lines.append("")
    lines.append("- Start with `template`, `intent`, `super_family`, `direction_prior` as low-cardinality NLP features.")
    lines.append("- Add timing features from this catalog (`mean_bar_ix`, `late_seen_share`, `seen_share`).")
    lines.append("- Use `example_headlines_top3` to manually inspect edge templates before modeling.")
    lines.append("- Prefer template-level statistics over company names for generalization across independent sessions.")
    lines.append("")
    lines.append("## Main CSV")
    lines.append("")
    lines.append("- `analysis/headline_template_catalog.csv`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consolidated headline template catalog.")
    parser.add_argument("--data-dir", default="data", help="Directory containing headline parquet files.")
    parser.add_argument(
        "--out-csv",
        default="analysis/headline_template_catalog.csv",
        help="Output CSV catalog path.",
    )
    parser.add_argument(
        "--out-summary",
        default="analysis/headline_template_catalog_summary.md",
        help="Output markdown summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataset(data_dir)
    catalog = build_catalog(df)
    catalog.to_csv(out_csv, index=False)
    write_summary(df, catalog, out_summary)

    print(f"Rows analyzed: {len(df)}")
    print(f"Templates in catalog: {catalog['template'].nunique()}")
    print(f"Catalog CSV: {out_csv.resolve()}")
    print(f"Summary MD: {out_summary.resolve()}")


if __name__ == "__main__":
    main()
