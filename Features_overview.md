## Feature Overview

### Original Features (Notebook 00 — Data Pull)

| Feature | Type | Status | Notes |
|---|---|---|---|
| `id` | String | Dropped in FE | Join key only |
| `title` | String | Dropped in FE | Used for junk filtering in cleaning |
| `publication_year` | Numerical | ✅ Final model | Citation maturity signal, ρ=-0.04, Mann-Whitney r=0.062 |
| `language` | Categorical | ✅ Final model (OHE) | Consolidated to en/other |
| `cited_by_count` | Numerical | 🔴 Dropped in FE | Leaky — not available at publication time |
| `referenced_works_count` | Numerical | ✅ Final model | **Strongest predictor** — GBC importance 0.40, Mann-Whitney r=0.573 |
| `fwci` | Numerical | 🔴 Dropped in FE | Leaky — field-weighted citation impact |
| `citation_top_1_percent` | Binary | 🔴 Dropped in FE | Leaky — derived from citations |
| `citation_top_10_percent` | Binary | 🎯 Target | 82/18 class balance |
| `cited_by_percentile_year_min` | Numerical | 🔴 Dropped in cleaning | ~1/3 null, leaky |
| `cited_by_percentile_year_max` | Numerical | 🔴 Dropped in cleaning | ~1/3 null, leaky |
| `first_year_citations` | Numerical | 🔴 Dropped in FE | Leaky — citations after publication |
| `authorship_count` | Numerical | 🔴 Dropped in NB09 | Replaced by `unique_authors_count` — misnamed, counts author entries not distinct authors. Correlation 0.99 |
| `countries_distinct_count` | Numerical | ✅ Final model | 25% zeros flagged with `countries_missing`. Mann-Whitney r=0.322 |
| `publication_type` | Categorical | ✅ Final model (OHE) | Journal articles 24.5% vs proceedings 15.2% high impact rate. Cramér's V=0.156 |
| `is_oa` | Binary | 🔴 Dropped in NB09 | Near-zero importance in both GBC and XGBoost. Redundant with `oa_status` |
| `oa_status` | Categorical | ✅ Final model (OHE) | Gold OA strongest — GBC importance 0.05, Cramér's V=0.132 |
| `keyword_count` | Numerical | 🔴 Dropped in NB09 | Near-zero importance in both models despite ρ=0.13 |
| `primary_topic_score` | Numerical | 🔴 Dropped in NB09 | Near-zero importance in both models |
| `topic_name` | Categorical | ✅ Final model (OHE) | 10 topics retained complete for project scope consistency |
| `topic_id` | String | 🔴 Dropped in FE | Redundant with topic_name |

---

### Enriched Features (Notebook 05 — Second Data Pull)

| Feature | Type | Status | Notes |
|---|---|---|---|
| `unique_authors_count` | Numerical | ✅ Final model | Replaces authorship_count. GBC importance 0.10, Mann-Whitney r=0.296 |
| `unique_institutions_count` | Numerical | ✅ Final model | 25% zeros flagged with `institutions_missing`. Mann-Whitney r=0.351 |
| `institution_edu_count` | Numerical | 🔴 Dropped in NB09 | Subset of unique_institutions_count. Correlation 0.88, near-zero importance both models |
| `institution_nonprofit_count` | Numerical | 🔴 Dropped in pull | 98.5% zeros — too sparse |
| `institution_gov_count` | Numerical | 🔴 Dropped in pull | 97.8% zeros — too sparse |
| `institution_company_count` | Numerical | 🔴 Dropped in pull | 92.2% zeros — too sparse |
| `funder_count` | Numerical | ✅ Final model | Strong signal — funded papers 2.4x more likely high impact. Mann-Whitney r=0.263 |
| `award_count` | Numerical | 🔴 Dropped in NB09 | 81% zeros, near-zero importance. Replaced by funder_count. Correlation 0.83 |
| `funder_names` | String | 🔴 Dropped before saving | Only needed for h-index lookup which was skipped |
| `sdg_count` | Numerical | ✅ Final model | Retained despite weak signal (ρ=0.04, r=0.050) — SDG alignment is a project feature |
| `sdg_max_score` | Numerical | 🔴 Dropped in NB09 | Redundant with sdg_count. Correlation 0.92 |
| `sdg_avg_score` | Numerical | 🔴 Dropped in NB06 | Redundant — 99% of papers have 0 or 1 SDG |
| `sdg_display_names` | String | 📊 EDA only | Pipe-separated SDG names. Dropped in FE |
| `sdg_numbers` | List | 🔴 Dropped after OHE | One-hot encoded into sdg_1 through sdg_17 |
| `sdg_4` | Binary | ✅ Final model | Quality Education — largest SDG tag (n=72k). Cramér's V=0.012 |
| `sdg_1`–`sdg_17` (excl. `sdg_4`) | Binary | 🔴 Dropped in NB09 | Sparse binary flags — near-zero importance both models |
| `referenced_works_list` | List | 🔴 Dropped before saving | 1.7M unique IDs — lookup not feasible |
| `countries_recalculated_count` | Numerical | 🔴 Dropped in NB06 | Only fixed 43 papers — no meaningful improvement |
| `countries_recalculated_list` | List | 🔴 Dropped in NB06 | Same coverage gap as original |

---

### Missingness Flags (Notebook 09 — Feature Engineering)

| Feature | Type | Status | Notes |
|---|---|---|---|
| `references_missing` | Binary | ✅ Final model | 24.9% of papers. Cramér's V=0.222 — strongest binary signal |
| `countries_missing` | Binary | ✅ Final model | 25.4% of papers. Cramér's V=0.190 |
| `institutions_missing` | Binary | ✅ Final model | 25.1% of papers. Cramér's V=0.190 |

Zero values in `referenced_works_count`, `countries_distinct_count` and
`unique_institutions_count` confirmed as OpenAlex coverage gaps via manual
inspection — not genuine zeros. Flags distinguish missing data from true zeros.

---

### Features Considered and Rejected

| Feature | Reason Rejected |
|---|---|
| Funder h-index | 74% of papers have no funder data — too sparse |
| Venue h-index | source field None for many proceedings — patchy coverage |
| referenced_works_avg_citations | 1.7M unique IDs — lookup not feasible |
| referenced_works_max_citations | Same as above |
| Funder type (government/private) | Not directly in funders field — requires separate lookup |
| Venue quality (CORE ranking, Scimago) | Identified as future work — timeline constraint |
| Abstract text features | No abstract data in pull — future work |
| Author reputation (h-index) | Requires additional API pull — future work |

---

### Final Model Feature Set (34 features)

**7 Numerical:**
`publication_year`, `referenced_works_count`, `unique_authors_count`,
`countries_distinct_count`, `unique_institutions_count`,
`funder_count`, `sdg_count`

**3 Binary flags:**
`sdg_4`, `references_missing`, `countries_missing`, `institutions_missing`

**24 OHE:**
`publication_type` × 5, `oa_status` × 6, `topic_name` × 10, `language` × 2

---

### Final Model Feature Importance (GBC Trimmed + Flags, Pre-Tuning)

| Feature | GBC Importance | Mann-Whitney r | Cramér's V | Signal |
|---|---|---|---|---|
| `referenced_works_count` | 0.40 | 0.573 | — | ⭐⭐⭐⭐⭐ Dominant |
| `unique_authors_count` | 0.10 | 0.296 | — | ⭐⭐⭐⭐ Strong |
| `publication_year` | 0.09 | 0.062 | — | ⭐⭐⭐ Moderate |
| `funder_count` | 0.05 | 0.263 | — | ⭐⭐⭐ Moderate |
| `oa_status_gold` | 0.05 | — | 0.132 | ⭐⭐ Moderate |
| `sdg_count` | 0.04 | 0.050 | — | ⭐⭐ Weak-moderate |
| `countries_distinct_count` | 0.04 | 0.322 | — | ⭐⭐ Moderate |
| `references_missing` | ~0.01 | — | 0.222 | ⭐⭐ Moderate |
| `countries_missing` | ~0.01 | — | 0.190 | ⭐⭐ Moderate |
| `institutions_missing` | ~0.01 | — | 0.190 | ⭐⭐ Moderate |
| All OHE features | <0.02 each | — | — | ⭐ Weak |