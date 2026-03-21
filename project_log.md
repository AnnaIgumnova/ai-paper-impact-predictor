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