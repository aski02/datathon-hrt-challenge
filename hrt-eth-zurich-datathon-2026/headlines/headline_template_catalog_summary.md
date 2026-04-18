# Headline Template Catalog Summary

Generated: 2026-04-18 13:57:45 UTC

## Scope

- Sources: `headlines_seen_train`, `headlines_unseen_train`, `headlines_seen_public_test`, `headlines_seen_private_test`.
- Company tokens (first 2 words) are treated as non-structural features.
- Templates are canonicalized by replacing numbers, regions, partner names, roles, and domain phrases with placeholders.

## Key Numbers

- Total headline rows: **215,827**
- Canonical templates: **62**
- Intents: **50**
- Super-families: **8**

## Top Templates

| Template | Rows | Share |
|---|---:|---:|
| `secures <NUM> contract with <PARTNER>` | 35,279 | 16.35% |
| `delays product launch in <DOMAIN> segment` | 5,180 | 2.40% |
| `faces regulatory review of <DOMAIN> practices` | 5,169 | 2.39% |
| `files for regulatory approval of new <DOMAIN> offering` | 5,144 | 2.38% |
| `reports record quarterly revenue, up <NUM> year-over-year` | 5,130 | 2.38% |
| `wins industry award for excellence in <DOMAIN>` | 5,114 | 2.37% |
| `launches next-generation <DOMAIN> platform` | 5,091 | 2.36% |
| `recalls products in <DOMAIN> line due to quality concerns` | 5,087 | 2.36% |
| `names new head of <DOMAIN> division` | 5,079 | 2.35% |
| `misses quarterly revenue estimates by <NUM>` | 5,071 | 2.35% |
| `sees <NUM> margin improvement in latest quarter` | 5,061 | 2.34% |
| `announces breakthrough in <DOMAIN>` | 5,058 | 2.34% |

## Super-Family Distribution

| Super-family | Rows | Share |
|---|---:|---:|
| `commercial_deals` | 46,400 | 21.50% |
| `product_technology` | 40,561 | 18.79% |
| `financial_performance` | 39,632 | 18.36% |
| `leadership_governance_ir` | 22,167 | 10.27% |
| `geo_operations` | 20,184 | 9.35% |
| `strategy_ma_reorg` | 17,137 | 7.94% |
| `regulatory_legal` | 15,742 | 7.29% |
| `guidance_capital` | 14,004 | 6.49% |

## Direction Prior Distribution

| Direction prior | Rows | Share |
|---|---:|---:|
| `positive` | 95,221 | 44.12% |
| `neutral_or_event` | 61,002 | 28.26% |
| `negative` | 59,604 | 27.62% |

## How To Use This Catalog

- Start with `template`, `intent`, `super_family`, `direction_prior` as low-cardinality NLP features.
- Add timing features from this catalog (`mean_bar_ix`, `late_seen_share`, `seen_share`).
- Use `example_headlines_top3` to manually inspect edge templates before modeling.
- Prefer template-level statistics over company names for generalization across independent sessions.

## Main CSV

- `analysis/headline_template_catalog.csv`
