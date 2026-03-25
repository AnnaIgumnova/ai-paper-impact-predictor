# Notebook 1 — Data cleaning
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
  topics and years, likely OpenAlex coverage gap, kept as zeros for now
- `referenced_works_count = 0` — 25% of papers, same pattern, kept as zeros

### Known issues retained
- 25% of papers have `countries_distinct_count = 0` and `referenced_works_count = 0`
  — OpenAlex coverage gap, to be investigated in second data pull
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


# Notebook 2 — EDA

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


# Notebook 3 — Feature engineering

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
- `ohe.pkl` saved for use in dashboard and future data pulls — when second data
  pull adds new columns, the encoder will be refit on the enriched dataset
- Final feature matrix: 7 numerical + 23 one-hot encoded columns = 30 features total
- `is_oa` kept as numerical (already binary 0/1) — one-hot encoding would be redundant


## Notebook 4 — Modelling (first iteration)
**Input**: `data/features/X_train.csv`, `data/features/X_test.csv`,
`data/features/y_train.csv`, `data/features/y_test.csv`
**Output**: no model saved — enriched data pull planned before final tuning

### Setup
- PyCaret 3.3.2 installed in dedicated `pycaret-env` environment (Python 3.10)
  to resolve NumPy and Python version conflicts with base environment (Python 3.13)
- Train and test data combined into single DataFrame for PyCaret `setup()`
- Target column renamed to `target`

### PyCaret setup() configuration
- `fix_imbalance=True` — SMOTE applied to training folds only to handle 82/18
  class imbalance. Test set never touched by SMOTE.
- `session_id=42` — reproducible results
- `index=False` — resets index to RangeIndex to avoid duplicate index error
  when combining train and test data
- `fold=10` — 10-fold StratifiedKFold cross-validation
- Transformed train set shape: 384,388 rows — expanded from 234k by SMOTE
- All 30 features recognised as numerical — correct since OHE was applied
  during feature engineering

### Model comparison — compare_models(sort='F1', n_select=5)
15 classification models compared using 10-fold stratified cross-validation
with SMOTE applied to training folds. Ranked by F1 score.

| Model | F1 | AUC | Precision | Recall |
|---|---|---|---|---|
| Gradient Boosting (gbc) | 0.54 | 0.84 | 0.50 | 0.60 |
| AdaBoost (ada) | 0.52 | 0.82 | 0.43 | 0.65 |
| Ridge / LDA | 0.50 | 0.81 | 0.38 | 0.75 |
| XGBoost | 0.49 | 0.85 | 0.61 | 0.41 |
| CatBoost | 0.48 | 0.85 | 0.63 | 0.39 |
| Dummy baseline | 0.00 | 0.50 | 0.00 | 0.00 |

XGBoost and CatBoost were not available in the default PyCaret installation —
installed separately via pip into `pycaret-env`.

### Model evaluation — Gradient Boosting Classifier
Evaluated on holdout test set of 58,609 papers.

**Confusion matrix:**
- True positives: 6,421 — correctly identified high-impact papers
- False negatives: 4,139 — missed high-impact papers
- False positives: 6,776 — incorrectly flagged as high impact
- True negatives: 41,273 — correctly identified non-high-impact papers

The model catches ~60% of high-impact papers, missing 4 in 10.

**AUC-ROC:** 0.84 — strong class separation. Given a random high-impact and
a random non-high-impact paper, the model correctly ranks the high-impact one
higher 84% of the time.

**Precision-Recall:** Average precision = 0.53. Curve starts near 1.0 at low
recall then drops — model is conservative, only flagging papers as high impact
when quite confident.

**Classification report:**
- High impact class: precision 0.487, recall 0.608, F1 0.541
- Non-high-impact class: precision 0.909, recall 0.859, F1 0.883

### Model evaluation — XGBoost
Evaluated on same holdout test set for comparison.

**Confusion matrix:**
- True positives: 4,282 — significantly fewer than GBC
- False negatives: 6,278 — misses far more high-impact papers than GBC
- False positives: 2,738 — fewer false alarms than GBC
- True negatives: 45,311

**Classification report:**
- High impact class: precision 0.610, recall 0.405, F1 0.487
- Non-high-impact class: precision 0.878, recall 0.943, F1 0.910

**GBC vs XGBoost comparison:**
- GBC has higher recall (0.608 vs 0.405) — catches more high-impact papers
- XGBoost has higher precision (0.610 vs 0.487) — fewer false positives
- GBC wins on F1 (0.54 vs 0.49) — better overall balance
- For publisher use case, GBC is preferred — missing fewer high-impact papers
  matters more than minimising false alarms

### Feature importance findings — GBC
Feature importance measured by impurity reduction across all trees.

Top features:
1. `referenced_works_count` — dominant predictor (0.40 importance)
2. `authorship_count` — second strongest (0.12)
3. `countries_distinct_count` — third (0.09)
4. `publication_year` — fourth (0.09)
5. `oa_status_gold` — fifth (0.05)

Notably weak: `keyword_count`, `primary_topic_score`, `is_oa` — near-zero
importance despite showing meaningful correlation with target in EDA.

Feature importance for XGBoost is distributed more evenly across all features,
with topic name columns dominating. This reflects a different importance
calculation method (gain-based vs impurity-based) rather than a better model.

### Decision — no tuning, proceed to enriched data pull
Hyperparameter tuning was deferred. The feature importance analysis revealed
that the two most important features (`authorship_count` and
`countries_distinct_count`) have known data quality issues, and new features
(funder data, SDGs, true institution counts) may change which model wins.

**Rationale:** tuning GBC on incomplete features risks wasted effort if a
different model wins after enrichment. The correct sequence is:
1. Pull enriched data
2. Rerun compare_models() with all features
3. Tune the actual winner
4. Do SHAP analysis once on the final model

### Known limitations
- No hyperparameter tuning performed — deferred to post-enrichment modelling
- SHAP analysis not available as built-in PyCaret plot in version 3.3.2 —
  will be run manually using the shap library after final model is selected
- Results may differ slightly on rerun due to SMOTE randomness within
  cross-validation folds despite session_id=42


## Notebook 5 — Enriched Data Pull
**Input**: `data/OpenAlex/openalex_ai_papers_cleaned.csv`
**Output**: `data/OpenAlex/openalex_ai_papers_enriched.csv` — 293,002 rows, 14 columns

## ⚠️ Do Not Rerun
Data pulled March 2026. OpenAlex updates continuously — rerunning will return
different results. Use saved CSV directly.

### Motivation
Feature importance analysis from Notebook 4 identified gaps in the existing
feature set that warranted a second data pull:
- `authorship_count` counts author-institution pairs, not true distinct authors
  or institutions — the 2nd most important feature is based on imprecise data
- `countries_distinct_count` has 25% zeros — suspected OpenAlex coverage gap
- No funding data available despite funded research tending to be higher impact
- No SDG alignment data available
- No institution type breakdown available

### Pull structure
Same topic/year structure as original pull (Notebook 00) for efficiency.
Filtered to only papers present in `openalex_ai_papers_cleaned.csv` after
each year/topic batch. Pull ran twice — second run after fixing author count
bug (see Known Issues below).

### Fields pulled
```python
.select([
    "id",
    "authorships",                   # unique authors, institution count and type
    "funders",                        # funder count
    "awards",                         # individual grant count
    "sustainable_development_goals",  # SDG alignment
    "referenced_works",               # raw list for scale estimation
])
```

### Features extracted

**From `authorships`:**
- `unique_authors_count` — distinct author count using author IDs where available,
  falling back to authorship entry count for older papers where `author.id = None`
- `unique_institutions_count` — distinct institution IDs across all authors
- `institution_edu_count` — count of education type institutions
- `institution_nonprofit_count` — dropped, 98.5% zeros
- `institution_gov_count` — dropped, 97.8% zeros
- `institution_company_count` — dropped, 92.2% zeros

**From `funders`:**
- `funder_count` — number of distinct funding organisations

**From `awards`:**
- `award_count` — number of individual grants (different from funder_count —
  same funder can give multiple grants)

**From `sustainable_development_goals`:**
- `sdg_count` — number of SDGs tagged
- `sdg_max_score` — highest confidence score among tagged SDGs
- `sdg_avg_score` — average confidence score across tagged SDGs
- `sdg_display_names` — list of SDG names for EDA
- `sdg_numbers` — list of SDG numbers (1-17) for one-hot encoding later

### Pull results
| Topic | Papers |
|---|---|
| Natural Language Processing Techniques | 53,776 |
| Neural Networks and Applications | 34,497 |
| Topic Modeling | 48,744 |
| Speech Recognition and Synthesis | 21,173 |
| Sentiment Analysis and Opinion Mining | 26,015 |
| Anomaly Detection Techniques and Applications | 28,186 |
| Evolutionary Algorithms and Applications | 7,861 |
| Metaheuristic Optimization Algorithms Research | 21,356 |
| Privacy-Preserving Technologies in Data | 27,136 |
| Quantum Computing Algorithms and Architecture | 24,258 |
| **Total** | **293,002** |

Match rate: 100.0% — 43 papers missing, negligible, likely removed or updated
by OpenAlex since original pull.

### Zero value investigation
| Column | Zero % | Decision |
|---|---|---|
| `institution_nonprofit_count` | 98.5% | Dropped — too sparse |
| `institution_gov_count` | 97.8% | Dropped — too sparse |
| `institution_company_count` | 92.2% | Dropped — too sparse |
| `award_count` | 81.0% | Kept — funded papers may show strong signal |
| `funder_count` | 73.9% | Kept — same reasoning |
| `sdg_count` | 48.9% | Kept — sufficient coverage |
| `institution_edu_count` | 30.7% | Kept — 69% of papers have education institutions |
| `unique_institutions_count` | 25.2% | Kept — genuine OpenAlex coverage gap |
| `unique_authors_count` | 0.6% | Kept — very complete |

### Countries recalculation investigation
Attempted to fix the 25% zeros in `countries_distinct_count` by recalculating
country count directly from institution country codes in `authorships`.
Result: only 43 papers moved from 0 to non-zero — no meaningful improvement.
The zeros reflect a genuine OpenAlex coverage gap where institution country data
is unavailable, not a data extraction issue. `countries_recalculated_count`
was dropped — adds no value over existing `countries_distinct_count`.

### Features considered and rejected
- **Funder h-index** — 74% of papers have no funder data, too sparse to add
  meaningful signal. `funder_count` and `award_count` retained as simpler proxies.
- **Venue h-index** — `primary_location.source` is None for many proceedings
  papers, coverage too patchy to be useful.
- **Referenced works citation quality** — 1.7M unique referenced work IDs across
  dataset, lookup not feasible within reasonable time. `referenced_works_count`
  already captures quantity signal as the strongest feature in the first model.
- **Funder type** — not directly available in the `funders` field, would require
  separate lookup per funder ID. Skipped given 74% null rate on funder data.

### Known issues
**Bug in first run — author count underestimated:**
The first version of `parse_authorships` counted unique author IDs only.
For older papers where `author.id = None`, this returned 0 even though authors
existed. After investigation, the function was fixed to fall back to authorship
entry count when no IDs are available. Pull was rerun after the fix.
`unique_authors_count` zeros reduced from 5,078 to 1,694 after the fix.

**Remaining 1,694 zero authors:**
Papers where `authorships` field is genuinely empty in OpenAlex — legitimate
data gap, not a code issue.

**73,703 zero institutions:**
Papers where authors have no institution data in OpenAlex (`institutions: []`).
Confirmed by manual inspection — genuine coverage gap, not extractable even
with different parsing logic.

### OpenAlex data quality finding
`institutions_distinct_count` (pulled in original notebook, renamed to
`authorship_count` during cleaning) is misnamed by OpenAlex. It counts
author-institution pairs, not distinct institutions. Confirmed by comparing
against `unique_institutions_count` computed from raw `authorships` field —
correlation of only 0.226 between the two columns. See Notebook 1 log for
full details.