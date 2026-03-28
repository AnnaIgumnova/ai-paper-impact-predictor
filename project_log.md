# AI Paper Impact Predictor — Project Log

## Notebook 1 — Data Cleaning
**Input**: `data/OpenAlex/openalex_ai_papers_full.csv`
**Output**: `data/OpenAlex/openalex_ai_papers_cleaned.csv` — ~293,000 rows, 19 columns

### Steps performed

**1. Null values**
- Checked null distribution across topics and years before dropping
- Dropped rows where `citation_top_10_percent` is null — distribution was equal
  across topics and years so dropping was safe
- Dropped `cited_by_percentile_year_min` and `cited_by_percentile_year_max` entirely —
  ~1/3 of data null, and both columns are leaky so not needed for modelling
- Filled null `publication_type` with `'unknown'` — valuable feature, equal null
  distribution across topics and years, mostly missing for older years
- Filled null `language` and `title` with `'unknown'` — very small fractions

**2. Duplicates**
- Checked for duplicate `id` values — none found
- Dropped duplicates on `title + publication_year + cited_by_count`, keeping first
- This logic retains version duplicates where the same paper appears under different
  OpenAlex IDs with different citation counts — these are treated as distinct records
- Many retained pairs differ in `countries_distinct_count`, `referenced_works_count`,
  `primary_topic_score`, `oa_status` and `publication_type` — likely the same paper
  indexed by OpenAlex from different sources (e.g. publisher page vs institutional
  repository) with inconsistent metadata. Resolving this at scale would require
  manual verification and is out of scope. Flagged as a known limitation.

**3. Junk titles**
- Dropped records with known junk titles: `'Front Matter'`, `'Table of Contents'`,
  `'Preface'`, `'Foreword'`, `'Keynote'`, `'Tutorials'` and similar metadata records
- Dropped IEEE index documents — unrealistic `authorship_count` values (up to 4787)
  and 0 citations, not individual research papers
- Dropped NeurIPS proceedings volumes — entire conference volumes misidentified as
  individual papers, accumulating citations from all papers within the volume
- Dropped CNS annual meeting abstract collections — published as single journal
  articles but contain abstracts from hundreds of presenters
- Dropped other non-research records: workshop proceedings collections and dataset
  release announcements misclassified as journal articles
- Note: some IEEE journal metadata pages (author guidelines, topic codes, publication
  information) survived junk title filtering — all have 0 citations and
  `citation_top_10_percent = 0` so impact on modelling is negligible

**4. Publication type consolidation**
- Consolidated `publication_type` from 579 raw values to 5 clean categories:
  `journal-article`, `proceedings-article`, `thesis`, `other`, `unknown`
- Used keyword mapping function covering multilingual variants

**5. Data types**
- Converted `citation_top_1_percent` and `citation_top_10_percent` from object
  to integer (0/1)
- Converted `is_oa` from bool to integer (0/1)
- Reset index after all cleaning steps

**6. Authors count investigation**
- Confirmed `authors_count` is capped at 100 by OpenAlex — only 87 rows differ
  from `institutions_distinct_count`
- Verified through OpenAlex URLs that `institutions_distinct_count` reflects
  actual author count, not distinct institutions
- Dropped `authors_count` — redundant and capped
- Renamed `institutions_distinct_count` → `authorship_count`

**7. Zero value investigation**
- `countries_distinct_count = 0` — 25% of papers, equal distribution across
  topics and years, likely OpenAlex coverage gap, kept as zeros
- `referenced_works_count = 0` — 25% of papers, same pattern, kept as zeros
- Both confirmed as OpenAlex coverage gaps in Notebook 09 — missingness flags
  added to distinguish missing data from genuine zeros

### Known issues retained
- 25% of papers have `countries_distinct_count = 0` and `referenced_works_count = 0`
  — OpenAlex coverage gap. Addressed in Notebook 09 with missingness flags.
- `citation_top_10_percent` labels for 2022–2024 may be noisy — papers have not
  had enough time to accumulate citations; label noise expected in this period
- Version duplicates and same-paper records with different metadata are retained
  where `cited_by_count` differs — OpenAlex data quality issue, out of scope to
  resolve at scale
- Some IEEE journal metadata pages survived junk title filtering — negligible
  modelling impact as all have 0 citations and are labelled non-high-impact

### OpenAlex Data Quality Finding — institutions_distinct_count

During cleaning, `institutions_distinct_count` was investigated and found to be
misnamed. Rather than counting distinct institutions involved in a paper, it counts
author-institution pairs — i.e. if 3 authors all work at the same institution,
it returns 3, not 1.

This was confirmed by comparing `institutions_distinct_count` against manually
computed distinct institution IDs from the `authorships` field in the enriched
data pull. The correlation between the two was only 0.226, confirming they measure
fundamentally different things.

**Decision:** `institutions_distinct_count` was renamed to `authorship_count` to
accurately reflect what it measures. A true distinct institution count
(`unique_institutions_count`) was computed in the enriched data pull by collecting
unique institution IDs across all authors using the `authorships` field directly.

---

## Notebook 2 — EDA

**Input**: `data/OpenAlex/openalex_ai_papers_cleaned.csv`
**Target variable**: `citation_top_10_percent` — binary 0/1, class balance 82/18

### Analysis performed

**Univariate analysis — numerical features**
- Box plots for all numerical columns: `publication_year`, `cited_by_count`,
  `referenced_works_count`, `fwci`, `authorship_count`, `countries_distinct_count`,
  `keyword_count`, `primary_topic_score`, `first_year_citations`,
  `citation_top_1_percent`, `citation_top_10_percent`
- Investigated top outliers in `authorship_count`, `fwci`, `cited_by_count`,
  `countries_distinct_count`, `first_year_citations` — verified against OpenAlex
  URLs, confirmed as legitimate records

**Univariate analysis — categorical features**
- Bar charts for `language`, `publication_type`, `is_oa`, `oa_status`, `topic_name`
- English dominates (~280k), long tail of other languages
- Journal articles and proceedings articles roughly equal and dominate
- Majority closed (~161k), gold is largest OA route (~51k)
- NLP and Topic Modeling dominate publication volume; Evolutionary Algorithms smallest

**Target variable analysis**
- Class balance: 82% not high impact / 18% high impact
- Rate is above the expected 10% because selected topics are above-average impact
  areas within the broader AI subfield
- High impact rate per year — relatively equal 16–17% across 2017–2024, with higher
  rates of 23–26% in 2015–2016
- High impact rate per topic per year — large variation between 6% and 42% across
  topics, confirming `topic_name` and `publication_year` as strong features

**Publication volume per topic per year**
- NLP dominates throughout and grows strongly post-2022
- Topic Modeling exploded post-2022 (LLM effect: ~2k papers in 2022 → ~8.5k in 2023)
- Evolutionary Algorithms and Metaheuristic Optimization flat and small throughout
- Most topics show strong acceleration post-2022 driven by the broader AI boom

**Bivariate analysis — continuous features vs target**
- Grouped box plots for `authorship_count`, `countries_distinct_count`,
  `referenced_works_count`, `keyword_count`, `primary_topic_score`

**Bivariate analysis — publication year vs target**
- Line chart of high impact rate per year with citation maturity annotation at 2022

**Bivariate analysis — is_oa vs target**
- Grouped bar chart: open access 21.0% vs not open access 15.6%

**Bivariate analysis — categorical features vs target**
- High impact rate bar charts for `publication_type`, `oa_status`,
  `topic_name`, `language`

**Correlation matrix**
- Spearman correlation on all numerical features including target

### Key findings
- `referenced_works_count` is the strongest numerical predictor (ρ=0.39)
- `countries_distinct_count` (ρ=0.24) and `primary_topic_score` (ρ=0.25) also
  show meaningful signal
- `publication_year` is slightly negative (ρ=-0.04) — citation maturity effect
- Journal articles have nearly double the high impact rate of proceedings articles
  (24.5% vs 15.2%)
- Open access papers outperform closed (21.0% vs 15.6%)
- Gold OA has the highest impact rate (29.0%), diamond the lowest (10.5%)
- Metaheuristic Optimization and Privacy-Preserving Technologies lead on impact rate
- NLP and Neural Networks have the lowest impact rates despite highest publication
  volumes — high volume dilutes the rate
- High impact rate peaked in 2015–2016, reflecting survivorship bias and the deep
  learning boom, then stabilised at ~16–18% with cyclical fluctuations
- `language` dominated by English (n=279k) — non-English languages consolidated
  to `other` in feature engineering
- `primary_topic_score` is counterintuitively lower for high impact papers —
  interdisciplinary papers score lower but attract broader citation interest
- `topic_name` and `publication_year` confirmed as strong features given large
  variation in high impact rates across topics (6%–42%) and years

---

## Notebook 3 — Feature Engineering

**Input**: `data/OpenAlex/openalex_ai_papers_cleaned.csv`
**Outputs**:
- `data/features/X_train.csv`
- `data/features/X_test.csv`
- `data/features/y_train.csv`
- `data/features/y_test.csv`
- `models/ohe.pkl`

### Steps performed

**1. Drop leaky and non-modelling columns**
Dropped: `id`, `title`, `cited_by_count`, `fwci`, `citation_top_1_percent`,
`first_year_citations`

**2. Language consolidation**
All non-English languages collapsed to `'other'` — English dominates at 279k papers,
all other languages have negligible individual sample sizes

**3. Separate features and target**
- Target: `citation_top_10_percent`
- Features: all remaining columns

**4. Null check**
No nulls found in feature matrix — ready to split

**5. Define feature sets**
- Numerical (7): `publication_year`, `authorship_count`, `countries_distinct_count`,
  `referenced_works_count`, `keyword_count`, `primary_topic_score`, `is_oa`
- Categorical (4): `publication_type`, `oa_status`, `topic_name`, `language`

**6. Train/test split**
80/20 split, `random_state=42`, `stratify=y` to preserve 82/18 class balance

**7. One-hot encode categoricals**
- Fitted `OneHotEncoder` on train only to avoid data leakage
- Transformed both train and test using the fitted encoder
- `handle_unknown='ignore'` ensures unseen categories in future data are encoded
  as all zeros rather than causing errors
- Combined numerical columns with encoded categorical columns

**8. Save artefacts**
All feature matrices, target vectors and fitted encoder saved to disk

### Notes
- Encoding done after the train/test split to avoid data leakage —
  encoder is fitted on train only, then applied to both sets
- `ohe.pkl` saved for use in dashboard and future data pulls
- Final feature matrix: 7 numerical + 23 one-hot encoded columns = 30 features total
- `is_oa` kept as numerical (already binary 0/1) — one-hot encoding would be redundant

---

## Notebook 4 — Modelling (First Model)
**Input**: `data/features/X_train.csv`, `data/features/X_test.csv`,
`data/features/y_train.csv`, `data/features/y_test.csv`
**Output**: `models/gbc_enriched.pkl` — saved after enriched modelling run

### Setup
- PyCaret 3.3.2 installed in dedicated `pycaret-env` environment (Python 3.10)
  to resolve NumPy and Python version conflicts with base environment (Python 3.13)
- Train and test data concatenated into single DataFrame for PyCaret `setup()`
- Target column: `citation_top_10_percent`

### PyCaret setup() configuration
- `fix_imbalance=True` — SMOTE applied to training folds only to handle 82/18
  class imbalance. Test set never touched by SMOTE.
- `session_id=42` — reproducible results
- `index=False` — resets index to RangeIndex to avoid duplicate index error
- `fold=10` — 10-fold StratifiedKFold cross-validation
- Transformed train set shape: 384,388 rows — expanded from 234k by SMOTE
- All 30 features recognised as numerical — correct since OHE applied during FE

### Model comparison — compare_models() ranked by F1
15 classification models compared using 10-fold stratified CV with SMOTE.

| Model | F1 | AUC | Recall | Precision |
|---|---|---|---|---|
| Gradient Boosting (gbc) | 0.54 | 0.84 | 0.61 | 0.49 |
| AdaBoost (ada) | 0.52 | 0.82 | 0.65 | 0.43 |
| Ridge / LDA | 0.50 | 0.81 | 0.75 | 0.38 |
| XGBoost | 0.49 | 0.85 | 0.41 | 0.61 |
| CatBoost | 0.48 | 0.85 | 0.39 | 0.63 |
| Dummy baseline | 0.00 | 0.50 | 0.00 | 0.00 |

### Model evaluation — GBC (test set)

| Metric | Value |
|---|---|
| F1 | 0.54 |
| AUC | 0.84 |
| Recall | 0.61 |
| Precision | 0.49 |
| True Positives | 6,421 |
| False Negatives | 4,139 |
| False Positives | 6,776 |
| True Negatives | 41,273 |

### Model evaluation — XGBoost (test set)

| Metric | Value |
|---|---|
| F1 | 0.49 |
| AUC | 0.85 |
| Recall | 0.41 |
| Precision | 0.61 |
| True Positives | 4,282 |
| False Negatives | 6,278 |
| False Positives | 2,738 |
| True Negatives | 45,311 |

GBC preferred over XGBoost — higher recall (0.61 vs 0.41) more important
than precision for publisher use case. GBC wins on F1 (0.54 vs 0.49).

### Feature importance findings — GBC
1. `referenced_works_count` — dominant predictor (0.40)
2. `authorship_count` — second strongest (0.12)
3. `countries_distinct_count` — third (0.09)
4. `publication_year` — fourth (0.09)
5. `oa_status_gold` — fifth (0.05)

Notably weak: `keyword_count`, `primary_topic_score`, `is_oa` — near-zero
importance despite meaningful EDA correlation with target.

### Decision — no tuning, proceed to enriched data pull
Hyperparameter tuning deferred. Feature importance revealed that the two most
important features (`authorship_count`, `countries_distinct_count`) have known
data quality issues, and new features from the enriched pull may change which
model wins. Correct sequence: enrich → retrain → tune → SHAP.

---

## Notebook 5 — Enriched Data Pull
**Input**: `data/OpenAlex/openalex_ai_papers_cleaned.csv`
**Output**: `data/OpenAlex/openalex_ai_papers_enriched.csv` — 293,002 rows, 14 columns

## ⚠️ Do Not Rerun
Data pulled March 2026. OpenAlex updates continuously — rerunning will return
different results. Use saved CSV directly.

### Motivation
Feature importance from Notebook 4 identified gaps:
- `authorship_count` counts author-institution pairs, not true distinct authors —
  the 2nd most important feature is based on imprecise data
- `countries_distinct_count` has 25% zeros — suspected OpenAlex coverage gap
- No funding data available despite funded research tending to be higher impact
- No SDG alignment data available
- No institution type breakdown available

### Fields pulled
```python
.select([
    "id",
    "authorships",
    "funders",
    "awards",
    "sustainable_development_goals",
    "referenced_works",
])
```

### Features extracted
- `unique_authors_count` — distinct author count using author IDs
- `unique_institutions_count` — distinct institution IDs across all authors
- `institution_edu_count` — count of education type institutions
- `funder_count` — number of distinct funding organisations
- `award_count` — number of individual grants
- `sdg_count`, `sdg_max_score`, `sdg_avg_score` — SDG alignment metrics
- `sdg_display_names`, `sdg_numbers` — for EDA and OHE

### Pull results
293,002 papers pulled. Match rate 100.0% — 43 papers missing, negligible.

### Zero value investigation
| Column | Zero % | Decision |
|---|---|---|
| `institution_nonprofit_count` | 98.5% | Dropped — too sparse |
| `institution_gov_count` | 97.8% | Dropped — too sparse |
| `institution_company_count` | 92.2% | Dropped — too sparse |
| `award_count` | 81.0% | Kept — signal confirmed in EDA |
| `funder_count` | 73.9% | Kept — strong signal confirmed |
| `sdg_count` | 48.9% | Kept — sufficient coverage |
| `unique_institutions_count` | 25.2% | Kept — genuine coverage gap, flagged in NB09 |
| `unique_authors_count` | 0.6% | Kept — very complete |

### Countries recalculation
Attempted to fix 25% zeros in `countries_distinct_count` by recalculating
from institution country codes — only 43 papers changed. Confirmed genuine
OpenAlex coverage gap. `countries_recalculated_count` dropped.
Coverage gap addressed with `countries_missing` flag in Notebook 09.

### Features considered and rejected
- Funder h-index — 74% null, too sparse
- Venue h-index — source field None for many proceedings
- Referenced works citation quality — 1.7M unique IDs, not feasible
- Funder type — requires separate lookup per funder ID

### Known issues
- Bug fixed in second run: `unique_authors_count` underestimated for older
  papers where `author.id = None` — fixed to fall back to entry count
- 1,694 remaining zero authors — genuine OpenAlex data gap
- 73,703 zero institutions — confirmed genuine coverage gap

---

## Notebook 6 — Data Merge, Cleaning and EDA of Enriched Features
**Input**:
- `data/OpenAlex/openalex_ai_papers_cleaned.csv`
- `data/OpenAlex/openalex_ai_papers_enriched.csv`
**Output**: `data/OpenAlex/openalex_ai_papers_enriched_cleaned.csv` — 293,045 rows, 54 columns

### Steps performed

**1. Merge**
Left join on `id` — 43 unmatched papers filled with 0. Kept — 2 are high-impact,
43 rows is 0.015% of dataset.

**2. SDG column handling**
- `sdg_display_names` joined with `|` separator
- `sdg_numbers` converted back to Python lists
- `sdg_avg_score` dropped — redundant, 99% papers have 0 or 1 SDG
- One-hot encoded into `sdg_1` through `sdg_17` binary columns

**3. Author and institution count investigation**
- `unique_authors_count` and `authorship_count` correlation = 0.99
- Decision at this stage: keep both, let model feature importance decide
- Final decision made in Notebook 09: `authorship_count` dropped,
  `unique_authors_count` retained — confirmed more accurate and higher
  model importance when both present

### EDA findings

**Funded vs unfunded papers**
Funded papers 31% high impact vs 13% unfunded — 2.4x more likely high impact.
Strong signal for `funder_count`.

**SDG analysis**
- Quality Education (SDG 4) most common (~73k papers)
- High impact rate varies 13%–33% across SDGs
- Life below water (SDG 14) highest impact rate (~33%, n=1,295)

**Full correlation matrix**
`referenced_works_count` remains strongest (ρ=0.39). New signals:
- `unique_institutions_count` (0.24), `funder_count` (0.23) — meaningful
- SDG features weak individually (ρ=0.04)

Key multicollinearity pairs:
- `unique_authors_count` / `authorship_count` — 0.99
- `sdg_count` / `sdg_max_score` — 0.92
- `unique_institutions_count` / `institution_edu_count` — 0.88
- `unique_institutions_count` / `countries_distinct_count` — 0.87
- `funder_count` / `award_count` — 0.83

Decision: all kept — tree models handle multicollinearity, feature importance
in Notebook 08 will guide final drops.

---

## Notebook 7 — Feature Engineering (Enriched)
**Input**: `data/OpenAlex/openalex_ai_papers_enriched_cleaned.csv`
**Outputs**:
- `data/features/X_train_enriched.csv`
- `data/features/X_test_enriched.csv`
- `data/features/y_train_enriched.csv`
- `data/features/y_test_enriched.csv`
- `models/ohe_enriched.pkl`

### Steps performed
Same structure as Notebook 3. Additional drops: `sdg_display_names`, `topic_id`.
Same 80/20 split, `random_state=42`, `stratify=y`.
OHE refit on enriched train set — saved as `ohe_enriched.pkl`.

### Final feature matrix
31 numerical + 23 OHE = 54 features total.

---

## Notebook 8 — Modelling (Enriched)
**Input**: `data/features/X_train_enriched.csv`, `data/features/X_test_enriched.csv`,
`data/features/y_train_enriched.csv`, `data/features/y_test_enriched.csv`
**Output**: `models/gbc_enriched.pkl`, `models/xgb_enriched.pkl`

### PyCaret setup() configuration
Same as Notebook 4 — SMOTE, 10-fold StratifiedKFold, session_id=42.
54 features, 293,045 rows.

### Model comparison — compare_models() ranked by F1

| Model | F1 | AUC | Recall | Precision |
|---|---|---|---|---|
| GBC | 0.53 | 0.84 | 0.52 | 0.53 |
| AdaBoost | 0.51 | 0.82 | 0.59 | 0.45 |
| XGBoost | 0.48 | 0.85 | 0.39 | 0.63 |
| CatBoost | 0.48 | 0.86 | 0.38 | 0.84 |

### Model evaluation — GBC (test set)

| Metric | First Model | Enriched |
|---|---|---|
| F1 | 0.54 | 0.53 |
| AUC | 0.84 | 0.84 |
| Recall | 0.61 | 0.53 |
| Precision | 0.49 | 0.53 |
| True Positives | 6,421 | 5,563 |
| False Negatives | 4,139 | 4,997 |

Recall dropped from 0.61 → 0.53 — enriched features added noise.
AUC unchanged — discriminative power intact.

### Model evaluation — XGBoost (test set)

| Metric | Value |
|---|---|
| Recall | 0.39 |
| Precision | 0.63 |
| F1 | 0.48 |
| AUC | 0.85 |
| PR AUC | 0.58 |

XGBoost feature importance distorted — `sdg_count` dominant despite ρ=0.04.
`referenced_works_count` only 9th despite being strongest predictor.

### Feature importance — GBC (enriched)
1. `referenced_works_count` — 0.38 (dominant, consistent with first model)
2. `unique_authors_count` — 0.06 (replaces `authorship_count` as author signal)
3. `publication_year` — 0.07
4. `funder_count` — 0.03 (new signal confirmed)
5. `sdg_count` — 0.03 (new signal confirmed, weak)

Near-zero: `authorship_count`, `institution_edu_count`, `award_count`,
`sdg_max_score`, all individual `sdg_1`–`sdg_17` flags

### Decision
Recall drop caused by noise in 54-feature set. Feature trimming required before
tuning. XGBoost ruled out — precision-heavy and feature importance unreliable.

---

## Notebook 9 — Feature Trimming
**Input**: `data/features/X_train_enriched.csv`, `data/features/X_test_enriched.csv`
**Outputs**:
- `data/features/X_train_trimmed.csv`
- `data/features/X_test_trimmed.csv`
- `data/features/y_train_trimmed.csv`
- `data/features/y_test_trimmed.csv`

### Features dropped

**Correlated pairs — weaker feature dropped:**
| Dropped | Kept | Correlation | Reason |
|---|---|---|---|
| `authorship_count` | `unique_authors_count` | 0.99 | Misnamed — counts entries not distinct authors. Lower model importance |
| `institution_edu_count` | `unique_institutions_count` | 0.88 | Subset of broader feature, near-zero importance |
| `award_count` | `funder_count` | 0.83 | 81% zeros, near-zero importance |
| `sdg_max_score` | `sdg_count` | 0.92 | Redundant — 99% papers have 0 or 1 SDG |

**Near-zero importance across GBC and XGBoost:**
`sdg_1`–`sdg_17` (excl. `sdg_4`), `primary_topic_score`, `keyword_count`, `is_oa`

**`sdg_4` retained** — largest SDG tag (n=72k), meaningful GBC importance.

### Missingness flags added
25% of papers have zeros in three key features confirmed as OpenAlex coverage
gaps via manual inspection — not genuine zeros.

| Flag | Coverage | Cramér's V |
|---|---|---|
| `references_missing` | 24.9% | 0.222 |
| `countries_missing` | 25.4% | 0.190 |
| `institutions_missing` | 25.1% | 0.190 |

Flags have stronger statistical association with target than most OHE features.

### Statistical tests

**Chi-squared and Cramér's V — binary features vs target:**
- Missingness flags confirmed as strongest binary predictors (V=0.19–0.22)
- `publication_type_journal-article` (V=0.156) and `oa_status_gold` (V=0.132)
  next strongest
- `oa_status_green` (V=0.000, p=0.87) and `topic_name_Anomaly Detection`
  (V=0.001, p=0.73) show no significant association — retained for project
  scope consistency

**Mann-Whitney U — continuous features vs target:**
| Feature | Effect Size (r) |
|---|---|
| `referenced_works_count` | 0.573 (large) |
| `unique_institutions_count` | 0.351 (medium) |
| `countries_distinct_count` | 0.322 (medium) |
| `unique_authors_count` | 0.296 (medium) |
| `funder_count` | 0.263 (medium) |
| `publication_year` | 0.062 (small) |
| `sdg_count` | 0.050 (small) |

`referenced_works_count` confirmed dominant by three independent methods:
EDA (ρ=0.39), GBC importance (0.40), Mann-Whitney effect size (0.573).
`sdg_count` weak by all three methods — XGBoost/AdaBoost distortion confirmed.

### Final feature set: 54 → 34 features
7 numerical + 3 binary flags + 1 SDG flag + 23 OHE = 34 features

---

## Notebook 10 — Modelling (Trimmed) — IN PROGRESS
**Input**: `data/features/X_train_trimmed.csv`, `data/features/X_test_trimmed.csv`,
`data/features/y_train_trimmed.csv`, `data/features/y_test_trimmed.csv`
**Output**: `models/gbc_trimmed.pkl` (pre-tuning), `models/gbc_tuned.pkl` (in progress)

### PyCaret setup() configuration
Same as Notebooks 4 and 8 — SMOTE, 10-fold StratifiedKFold, session_id=42.
34 features, 293,045 rows.

### Model comparison — compare_models() ranked by F1

| Model | F1 | AUC | Recall | Precision |
|---|---|---|---|---|
| GBC | 0.54 | 0.84 | 0.60 | 0.50 |
| AdaBoost | 0.52 | 0.82 | 0.65 | 0.43 |
| XGBoost | 0.49 | 0.85 | 0.40 | 0.61 |
| CatBoost | 0.48 | 0.85 | 0.39 | 0.63 |

### Results across all runs — GBC

| Metric | First Model | Enriched | Trimmed + Flags |
|---|---|---|---|
| F1 | 0.54 | 0.53 | 0.54 |
| AUC | 0.84 | 0.84 | 0.84 |
| Recall | 0.61 | 0.53 | 0.60 |
| Precision | 0.49 | 0.53 | 0.50 |
| True Positives | 6,421 | 5,563 | 6,305 |
| False Negatives | 4,139 | 4,997 | 4,255 |

Trimming recovered recall from 0.53 → 0.60. AUC stable at 0.84 throughout.

### Model selection — GBC confirmed
Four models evaluated in depth. GBC selected based on:

**1. Performance** — best F1 and precision/recall balance across all runs

**2. Feature importance consistency** — only model consistent with statistical
evidence. `referenced_works_count` dominant (0.40) matching EDA and
Mann-Whitney. XGBoost, AdaBoost, CatBoost all distort `sdg_count` importance
despite near-zero statistical association (V=0.03, r=0.050).

**3. Business case** — AdaBoost higher recall (0.65) but 9,319 false positives
vs GBC's 6,396 — too noisy. CatBoost precision-heavy (recall 0.38) — misses
too many high-impact papers.

### Tuning — COMPLETE
```python
tune_model(
    gbc_trimmed,
    optimize         = 'F1',
    n_iter           = 50,
    search_library   = 'scikit-optimize',
    search_algorithm = 'bayesian',
    early_stopping   = True,
    choose_better    = True
)
```

Bayesian search completed 50 iterations — approximately 5 hours runtime.
Best hyperparameters saved to `models/gbc_tuned.pkl`.

### Tuning Results — CV

| Metric | Pre-Tuning | Tuned |
|---|---|---|
| F1 | 0.5405 | 0.5523 |
| AUC | 0.8406 | 0.8462 |
| Recall | 0.5859 | 0.6084 |
| Precision | 0.5016 | 0.4867 |

### Test Set Evaluation — Final Model Comparison

| Metric | First Model | Enriched | Trimmed+Flags | Tuned GBC |
|---|---|---|---|---|
| F1 | 0.5428 | 0.5257 | 0.5405 | 0.5519 |
| AUC | 0.8413 | 0.8412 | 0.8406 | 0.8446 |
| Recall | 0.5961 | 0.5195 | 0.5859 | 0.6442 |
| Precision | 0.4982 | 0.5323 | 0.5016 | 0.4827 |
| True Positives | 6,421 | 5,563 | 6,305 | 6,803 |
| False Negatives | 4,139 | 4,997 | 4,255 | 3,757 |

Tuned GBC is the best model across all runs — highest F1, AUC, recall and
true positives. Catches 382 more high-impact papers than the first model.

### Feature Importance — Notable Changes After Tuning
- `referenced_works_count` remains dominant (0.30)
- `references_missing` jumped to 4th — missingness flag more important
  than pre-tuning results suggested
- `funder_count` strengthened to 3rd
- `publication_year` rose to 2nd — citation maturity signal amplified

---

## Notebook 11 — Threshold Analysis and SHAP
**Input**: `data/features/X_test_trimmed.csv`, `data/features/y_test_trimmed.csv`,
`data/features/X_train_trimmed.csv`, `data/features/y_train_trimmed.csv`,
`models/gbc_tuned.pkl`
**Environment**: pycaret-env (Python 3.10)

### Threshold Analysis
Tested thresholds from 0.10 to 0.90 in steps of 0.01.
Default threshold of 0.50 selected — optimal F1 threshold (0.49) showed
negligible improvement (F1 0.5519 vs 0.5518) at cost of 729 additional
false positives for only 278 more true positives.
Streamlit dashboard displays raw probability scores rather than binary
flag — users apply their own judgement.

### Train vs Test Generalisation

| Metric | Train | Test | Gap |
|---|---|---|---|
| F1 | 0.5703 | 0.5519 | 0.018 |
| Recall | 0.6665 | 0.6442 | 0.022 |
| Precision | 0.4984 | 0.4827 | 0.016 |
| AUC | 0.8587 | 0.8446 | 0.014 |

Small train/test gap confirms good generalisation — model learned genuine
patterns rather than memorising training data.

### Performance by Publication Year

| Year | F1 | Recall |
|---|---|---|
| 2015–2016 | 0.62–0.64 | 0.75–0.76 |
| 2017–2021 | 0.54–0.56 | 0.58–0.65 |
| 2022–2023 | 0.53 | 0.60–0.63 |
| 2024 | 0.49 | 0.59 |

Performance degrades gradually for recent papers due to citation maturity
bias in the target variable. Model remains useful for 2024 papers —
catching 59% of high-impact papers at publication time.

### SHAP Analysis
TreeExplainer on 5,000 test set sample (random_state=42).
Four plots: bar importance, beeswarm, waterfall, dependence plot.

Key findings:
- `referenced_works_count` dominant (mean |SHAP| 0.27) — confirmed by
  EDA, statistical tests, GBC importance and SHAP independently
- `publication_year` second (0.11) — citation maturity bias confirmed
- `references_missing` third (0.07) — missingness flag outperforms most
  engineered features
- Survey/review papers with 500+ references correctly flagged as high impact
- NLP papers penalised — consistent with lowest impact rate in EDA (10.2%)
- Funded papers and gold OA papers consistently pushed toward high impact

### Citation Maturity Bias
`publication_year` high SHAP importance confirmed as bias from target variable
definition — cumulative citations disadvantage recent papers. Per-year analysis
shows gradual performance degradation, not a cliff edge. Model remains useful
for recent papers but predictions less reliable post-2021.

---

## Future Work
- Redefine target using first-year citations — removes citation maturity bias,
  `first_year_citations` already available in dataset
- Temporal train/test split (2015-2021 train, 2022-2024 test) — more realistic
  evaluation, best done alongside target redefinition
- Venue quality features (CORE ranking, Scimago journal quartile)
- Abstract text features
- Author reputation features (h-index at time of publication)
- Referenced works citation quality (blocked by 1.7M unique ID scale)