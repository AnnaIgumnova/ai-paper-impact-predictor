## Feature Overview

### Original Features (Notebook 00 — Data Pull)

| Feature | Type | Status | Notes |
|---|---|---|---|
| `id` | String | Dropped in FE | Join key only |
| `title` | String | Dropped in FE | Used for junk filtering in cleaning |
| `publication_year` | Numerical | ✅ Model | Citation maturity signal, ρ=-0.04 with target |
| `language` | Categorical | ✅ Model (OHE) | Consolidated to en/other |
| `cited_by_count` | Numerical | Dropped in FE | Leaky — not available at publication time |
| `referenced_works_count` | Numerical | ✅ Model | **Strongest predictor — GBC importance 0.40** |
| `fwci` | Numerical | Dropped in FE | Leaky — field-weighted citation impact |
| `citation_top_1_percent` | Binary | Dropped in FE | Leaky — derived from citations |
| `citation_top_10_percent` | Binary | 🎯 Target | 82/18 class balance |
| `cited_by_percentile_year_min` | Numerical | Dropped in cleaning | ~1/3 null, leaky |
| `cited_by_percentile_year_max` | Numerical | Dropped in cleaning | ~1/3 null, leaky |
| `first_year_citations` | Numerical | Dropped in FE | Leaky — citations after publication |
| `authorship_count` | Numerical | ⚠️ Check after model | Renamed from institutions_distinct_count. GBC importance 0.12. Correlation 0.99 with unique_authors_count |
| `countries_distinct_count` | Numerical | ⚠️ Check after model | 25% zeros, OpenAlex coverage gap. ρ=0.24. Correlation 0.87 with unique_institutions_count |
| `publication_type` | Categorical | ✅ Model (OHE) | Journal articles 24.5% vs proceedings 15.2% high impact rate |
| `is_oa` | Binary | ✅ Model | Near-zero GBC importance despite ρ=0.07 |
| `oa_status` | Categorical | ✅ Model (OHE) | Gold OA strongest — GBC importance 0.05 |
| `keyword_count` | Numerical | ✅ Model | Near-zero GBC importance despite ρ=0.13 |
| `primary_topic_score` | Numerical | ✅ Model | Near-zero GBC importance. Counterintuitively lower for high impact papers — interdisciplinary signal |
| `topic_name` | Categorical | ✅ Model (OHE) | Strong signal — GBC importance varies by topic |
| `topic_id` | String | Dropped in FE | Redundant with topic_name |

---

### Enriched Features (Notebook 05 — Second Data Pull)

| Feature | Type | Status | Notes |
|---|---|---|---|
| `unique_authors_count` | Numerical | ⚠️ Check after model | Capped at 100 for 30 large papers. Correlation 0.99 with authorship_count. More accurate for small papers, less reliable for large collaborations |
| `unique_institutions_count` | Numerical | ⚠️ Check after model | True distinct institution count. 25% zeros — OpenAlex coverage gap. ρ=0.24 with target. Correlation 0.87 with countries_distinct_count |
| `institution_edu_count` | Numerical | ⚠️ Check after model | Education institution count. ρ=0.22. Correlation 0.88 with unique_institutions_count — likely redundant |
| `institution_nonprofit_count` | Numerical | Dropped in pull | 98.5% zeros — too sparse |
| `institution_gov_count` | Numerical | Dropped in pull | 97.8% zeros — too sparse |
| `institution_company_count` | Numerical | Dropped in pull | 92.2% zeros — too sparse |
| `funder_count` | Numerical | ✅ Model | **Strong new signal** — ρ=0.23. Funded papers 2.4x more likely high impact. Correlation 0.83 with award_count |
| `award_count` | Numerical | ⚠️ Check after model | 81% zeros. ρ=0.22. Correlation 0.83 with funder_count — likely redundant |
| `funder_names` | String | Dropped before saving | Only needed for h-index lookup which was skipped |
| `sdg_count` | Numerical | ✅ Model | 49% zeros. ρ=0.04 weak individually. Correlation 0.92 with sdg_max_score |
| `sdg_max_score` | Numerical | ⚠️ Check after model | ρ=0.04. Correlation 0.92 with sdg_count — likely redundant |
| `sdg_avg_score` | Numerical | Dropped in notebook 06 | Redundant — 99% of papers have 0 or 1 SDG |
| `sdg_display_names` | String | 📊 EDA only | Pipe-separated SDG names. Dropped in FE |
| `sdg_numbers` | List | Dropped after OHE | One-hot encoded into sdg_1 through sdg_17 |
| `sdg_1` through `sdg_17` | Binary | ✅ Model | Individual SDG flags. High impact rate varies 13%–33% across SDGs |
| `referenced_works_list` | List | Dropped before saving | Scale estimation only — 1.7M unique IDs, lookup not feasible |
| `countries_recalculated_count` | Numerical | Dropped in notebook 06 | Only fixed 43 papers — no meaningful improvement over original |
| `countries_recalculated_list` | List | Dropped in notebook 06 | Same coverage gap as original |

---

### Features Considered and Rejected

| Feature | Reason Rejected |
|---|---|
| Funder h-index | 74% of papers have no funder data — too sparse |
| Venue h-index | source field None for many proceedings papers — patchy coverage |
| referenced_works_avg_citations | 1.7M unique referenced work IDs — lookup not feasible |
| referenced_works_max_citations | Same as above |
| Funder type (government/private) | Not directly in funders field — requires separate lookup |

---

### Features by Role Summary

**🎯 Target**
`citation_top_10_percent`

**✅ First model (30 features)**
7 numerical + 23 OHE:
`publication_year`, `authorship_count`, `countries_distinct_count`,
`referenced_works_count`, `keyword_count`, `primary_topic_score`, `is_oa`
+ OHE of `publication_type`, `oa_status`, `topic_name`, `language`

**✅ Second model (54 features)**
31 numerical + 23 OHE — all first model features plus:
`unique_authors_count`, `unique_institutions_count`, `institution_edu_count`,
`funder_count`, `award_count`, `sdg_count`, `sdg_max_score`,
`sdg_1` through `sdg_17`

**⚠️ Keep or drop — decide after second model feature importance**
- `authorship_count` vs `unique_authors_count` — correlation 0.99
- `countries_distinct_count` vs `unique_institutions_count` — correlation 0.87
- `institution_edu_count` vs `unique_institutions_count` — correlation 0.88
- `funder_count` vs `award_count` — correlation 0.83
- `sdg_count` vs `sdg_max_score` — correlation 0.92
- `sdg_1` through `sdg_17` — check if they add value over `sdg_count`

**📊 EDA only (never in model)**
`sdg_display_names`, `title`, `id`

**🔴 Dropped — leaky**
`cited_by_count`, `fwci`, `citation_top_1_percent`, `first_year_citations`,
`cited_by_percentile_year_min`, `cited_by_percentile_year_max`

**🔴 Dropped — too sparse**
`institution_nonprofit_count` (98.5% zeros),
`institution_gov_count` (97.8% zeros),
`institution_company_count` (92.2% zeros)

**🔴 Dropped — redundant or no value**
`topic_id`, `sdg_avg_score`, `countries_recalculated_count`,
`countries_recalculated_list`, `referenced_works_list`, `funder_names`

---

### First Model Feature Importance (GBC)

| Feature | Importance | Signal Strength |
|---|---|---|
| `referenced_works_count` | 0.40 | ⭐⭐⭐⭐⭐ Dominant |
| `authorship_count` | 0.12 | ⭐⭐⭐⭐ Strong |
| `countries_distinct_count` | 0.09 | ⭐⭐⭐ Moderate |
| `publication_year` | 0.09 | ⭐⭐⭐ Moderate |
| `oa_status_gold` | 0.05 | ⭐⭐ Moderate |
| `topic_name_Topic Modeling` | 0.05 | ⭐⭐ Moderate |
| `publication_type_journal-article` | 0.04 | ⭐⭐ Moderate |
| All other features | <0.02 each | ⭐ Weak |
| `keyword_count`, `primary_topic_score`, `is_oa` | ~0.00 | Negligible |