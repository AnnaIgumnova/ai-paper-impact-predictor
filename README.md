# AI Paper Impact Predictor

> Predicting whether an AI research paper will become highly cited using machine learning and metadata available at publication time.

## Overview

This project builds a machine learning pipeline to predict whether an AI research paper will become highly cited — defined as reaching the top 10% of citations within its subfield and year.

The dataset contains ~293,000 AI research papers published between 2015–2024, pulled from the [OpenAlex API](https://openalex.org/) across 10 topics in the Artificial Intelligence subfield. The model uses only metadata available at publication time — no full text, no citation history, no author reputation scores required.

## Final Model

**Gradient Boosting Classifier — tuned, 34 features**

| Metric | Score |
|---|---|
| F1 | 0.552 |
| AUC | 0.845 |
| Recall | 0.644 |
| Precision | 0.483 |
| True Positives | 6,803 / 10,560 |

The model correctly identifies **6,803 out of 10,560 high-impact papers** at publication time — catching 6.4 in 10 papers that would otherwise be indistinguishable from the majority.

Train/test gap is ~0.02 across all metrics — confirms good generalisation, no overfitting.

## Pipeline

| Stage | Notebook | Description |
|---|---|---|
| Data Pull | 00 | OpenAlex API via `pyalex` — 293,045 papers |
| Data Cleaning | 01 | Deduplication, junk title removal, type consolidation |
| EDA | 02 | Univariate, bivariate analysis, correlation matrix |
| Feature Engineering | 03 | OHE, train/test split, artefact saving |
| Modelling (First) | 04 | PyCaret compare_models(), GBC selected |
| Enriched Data Pull | 05 | Second OpenAlex pull — funder, SDG, institution data |
| Enriched EDA | 06 | EDA on enriched features |
| Feature Engineering (Enriched) | 07 | 54-feature set |
| Modelling (Enriched) | 08 | compare_models() on enriched features |
| Feature Trimming | 09 | 54 → 34 features, missingness flags added |
| Modelling (Trimmed + Tuned) | 10 | compare_models() + bayesian tuning |
| Threshold Analysis + SHAP | 11 | Threshold analysis, SHAP explainability, generalisation |

## Key Findings

**What predicts citation impact in AI research:**
- `referenced_works_count` — dominant predictor confirmed by four independent methods: EDA (ρ=0.39), statistical tests (Mann-Whitney r=0.573), GBC importance (0.40) and SHAP (mean |SHAP| 0.27). Papers that engage broadly with prior work — especially survey and review articles — are consistently flagged as high impact
- International collaboration — `countries_distinct_count` and `unique_institutions_count` both show medium effect sizes (r=0.32–0.35)
- Funding — funded papers have 31% high impact rate vs 13% unfunded — 2.4x difference
- Gold open access — freely available papers get cited significantly more
- Journal articles outperform proceedings (24.5% vs 15.2% high impact rate)
- Topic matters — Metaheuristic Optimization and Privacy-Preserving Technologies lead on impact rate; NLP lags despite highest volume

**Model selection — why GBC over XGBoost, AdaBoost and CatBoost:**
GBC is the only model whose feature importance rankings are consistent with statistical evidence. XGBoost, AdaBoost and CatBoost all over-weight `sdg_count` (Cramér's V=0.03, Mann-Whitney r=0.050) despite near-zero statistical association with the target.

**Known limitation — citation maturity bias:**
`publication_year` is the second most important SHAP feature but reflects a bias in the target variable — cumulative citations disadvantage recent papers. Performance degrades gradually from F1 0.64 (2015–2016) to F1 0.49 (2024). The model remains useful for recent papers but predictions are less reliable post-2021.

## Use Cases

- **Academic publishers** — identify high-impact papers early in the editorial process, prioritise peer review resources
- **Research funders** — allocate grants to emerging fields before they peak; funded papers are 2.4x more likely to be high impact
- **University strategy offices** — identify growing AI subfields for faculty hiring and lab investment

## Data

~293,000 AI research papers (2015–2024) across 10 topics:

- Natural Language Processing Techniques
- Neural Networks and Applications
- Topic Modeling
- Speech Recognition and Synthesis
- Sentiment Analysis and Opinion Mining
- Anomaly Detection Techniques and Applications
- Evolutionary Algorithms and Applications
- Metaheuristic Optimization Algorithms Research
- Privacy-Preserving Technologies in Data
- Quantum Computing Algorithms and Architecture

## Features

34 features used in the final model — all available at publication time:

**Numerical (7):** `publication_year`, `referenced_works_count`, `unique_authors_count`, `countries_distinct_count`, `unique_institutions_count`, `funder_count`, `sdg_count`

**Binary flags (4):** `sdg_4`, `references_missing`, `countries_missing`, `institutions_missing`

**OHE (23):** `publication_type` × 5, `oa_status` × 6, `topic_name` × 10, `language` × 2

Missingness flags added to handle OpenAlex coverage gaps — ~25% of papers have zeros in `referenced_works_count`, `countries_distinct_count` and `unique_institutions_count` confirmed as indexing gaps, not genuine zeros.

## Tech Stack

- **Python 3.10** — modelling notebooks (pycaret-env)
- **Python 3.13.5** — data pull and EDA notebooks (base)
- **PyCaret 3.3.2** — model comparison and tuning
- **scikit-optimize** — bayesian hyperparameter search
- **SHAP** — model explainability
- **Plotly** — visualisation
- **pyalex** — OpenAlex API
- **pandas, numpy, scikit-learn** — data processing

## Project Structure
```
├── data/
│   ├── OpenAlex/                              # Raw and cleaned datasets
│   └── features/                             # Train/test feature matrices
├── models/
│   ├── gbc_trimmed.pkl                       # Pre-tuning baseline
│   ├── gbc_tuned.pkl                         # Final tuned model
│   ├── gbc_enriched.pkl                      # Enriched model
│   ├── xgb_enriched.pkl                      # XGBoost enriched (ruled out)
│   ├── ohe.pkl                               # First model encoder
│   └── ohe_enriched.pkl                      # Enriched model encoder
├── notebooks/
│   ├── 00_AI_Papers_data_pull.ipynb          # ⚠️ Do not rerun
│   ├── 01_AI_Papers_data_cleaning.ipynb
│   ├── 02_AI_Papers_data_EDA.ipynb
│   ├── 03_AI_Papers_feature_engineering.ipynb
│   ├── 04_AI_Papers_modelling.ipynb
│   ├── 05_AI_Papers_data_pull_enriched.ipynb # ⚠️ Do not rerun
│   ├── 06_AI_Papers_data_merge_EDA_enriched.ipynb
│   ├── 07_AI_Papers_feature_engineering_enriched.ipynb
│   ├── 08_AI_Papers_modelling_enriched.ipynb
│   ├── 09_AI_Papers_feature_trimming.ipynb
│   ├── 10_AI_Papers_modelling_trimmed_and_tuning.ipynb
│   └── 11_AI_Papers_threshold_and_SHAP.ipynb
├── project_log.md
├── modelling_overview.md
├── feature_overview.md
└── README.md
```

## Future Work

- Redefine target using first-year citations — removes citation maturity bias, `first_year_citations` already available in dataset
- Temporal train/test split (2015–2021 train, 2022–2024 test) — more realistic evaluation, best done alongside target redefinition
- Venue quality features (CORE ranking, Scimago journal quartile)
- Abstract text features via OpenAlex abstract inverted index
- Author reputation features (h-index at time of publication)
- Referenced works citation quality (blocked by 1.7M unique ID scale)
