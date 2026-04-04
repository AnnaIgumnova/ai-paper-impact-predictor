# AI Paper Impact Predictor

> Predicting whether an AI research paper will become highly cited using machine learning and metadata available at publication time.

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyCaret 3.3.2](https://img.shields.io/badge/PyCaret-3.3.2-orange)](https://pycaret.org/)
[![Data: OpenAlex](https://img.shields.io/badge/Data-OpenAlex-green)](https://openalex.org/)

---

## Overview

This project builds a machine learning pipeline to predict whether an AI research paper will become highly cited — defined as reaching the **top 10% of citations within its subfield and year** (OpenAlex benchmark definition).

The dataset contains ~293,000 AI research papers published between 2015–2024, pulled from the [OpenAlex API](https://openalex.org/) across 10 topics in the Artificial Intelligence subfield. The model uses only metadata available at publication time — no full text, no citation history, no author reputation scores required.

**This is a screening tool, not a decision maker.** It is designed to help publishers, funders and universities prioritise which papers to review first — not to determine whether a paper is good or bad.

---

## The Problem

Academic publishers, research funders and universities spend millions deciding which research fields to invest in — launching new journals, allocating grants, building research labs. These are long-term strategic decisions made largely reactively, based on citation trends that are already visible. By the time a field is obviously growing, the first-mover advantage is gone.

According to the [Stanford AI Index 2025](https://hai.stanford.edu/ai-index/2025-ai-index-report/research-and-development), there were 242,000 AI publications in computer science in 2023 alone — 2.5× more than in 2013. Finding the papers that genuinely matter is getting harder every year.

---

## The Solution

A machine learning model that predicts citation impact **at the moment of publication** — before any citations exist. Given 14 metadata inputs available at submission time, the model returns a probability score and an explanation of which features drove the prediction.

The app lets users set their own **time horizon** (years since publication) rather than entering a fixed calendar year. This removes citation maturity bias: a user can ask "given 5 years to accumulate citations, how likely is this paper to reach the top 10%?" — making predictions comparable regardless of when the paper was published.

---

## Final Model

**Gradient Boosting Classifier — tuned, 34 features**

| Metric | Score | Plain language |
|---|---|---|
| Accuracy | 81.4% | Correct prediction 8 out of 10 times |
| AUC | 84.6% | 85% better at ranking high-impact papers than random chance |
| Recall | 64.4% | Correctly identifies 6 in 10 high-impact papers |
| Precision | 48.3% | When model predicts high impact, correct ~half the time |
| True Positives | 6,803 / 10,560 | Papers caught in test set of 58,609 |

Train/test gap is ~0.02 across all metrics — confirms good generalisation, no overfitting.

**Why Gradient Boosting over XGBoost, AdaBoost and CatBoost:**
GBC is the only model whose feature importance rankings are consistent with independent statistical tests. XGBoost, AdaBoost and CatBoost all over-weight `sdg_count` (Cramér's V=0.03, Mann-Whitney r=0.050) despite near-zero statistical association with the target. We selected the model we could trust, not just the one with the best number.

---

## Key Findings from EDA

**What predicts citation impact in AI research** (all figures from our dataset of 293,000 papers):

| Feature | Finding |
|---|---|
| `referenced_works_count` | Dominant predictor — confirmed by EDA (ρ=0.39), Mann-Whitney (r=0.573), GBC importance and SHAP. Papers engaging broadly with prior work are consistently high impact. |
| Funding | Funded papers: 31% high impact rate vs 13% unfunded — **2.4× difference** |
| Gold Open Access | 29% high impact rate vs 15.6% closed access — freely available research gets read and cited more |
| Publication type | Journal articles: 24.5% vs 15.2% for proceedings — more rigorous review drives long-term advantage |
| International collaboration | `countries_distinct_count`, `unique_institutions_count` and `authorship_count` all correlate with impact (ρ=0.19–0.24) |
| Topic | Impact rate ranges from **8.8% to 27.6%** across the 10 topics — topic choice matters as much as any other feature |

**Counterintuitive findings:**
- NLP has the highest publication volume (54K papers) but one of the **lowest** high-impact rates (10.2%) — crowding dilutes impact
- Neural Networks: 35K papers, 8.8% high-impact rate — the lowest of all topics despite high volume
- `primary_topic_score` is **negatively** correlated with impact — interdisciplinary papers score lower on OpenAlex topic confidence but attract broader citations
- Metaheuristic Optimization: 21K papers, **27.6% impact rate** — highest of all topics despite modest volume

---

## Streamlit App

The model is deployed as an interactive Streamlit app. Users input paper metadata and receive:
- A probability score (likelihood of reaching top 10% citations)
- A SHAP explanation showing which features drove the prediction
- The ability to change any input and see the score respond in real time

**Key design decision:** The app uses **years since publication** instead of publication year. This lets users set their own time horizon and removes citation maturity bias from the user-facing experience.

---

## Pipeline

| Stage | Notebook | Description |
|---|---|---|
| Data Pull | `00` | OpenAlex API via `pyalex` — 293,045 papers ⚠️ Do not rerun |
| Data Cleaning | `01` | Deduplication, junk title removal, type consolidation |
| EDA | `02` | Univariate, bivariate analysis, correlation matrix |
| Feature Engineering | `03` | OHE, train/test split, artefact saving |
| Modelling (First) | `04` | PyCaret `compare_models()`, GBC selected |
| Enriched Data Pull | `05` | Second OpenAlex pull — funder, SDG, institution data ⚠️ Do not rerun |
| Enriched EDA | `06` | EDA on enriched features |
| Feature Engineering (Enriched) | `07` | 54-feature set |
| Modelling (Enriched) | `08` | `compare_models()` on enriched features |
| Feature Trimming | `09` | 54 → 34 features, missingness flags added |
| Modelling (Trimmed + Tuned) | `10` | `compare_models()` + Bayesian tuning (50 iterations, ~5 hours) |
| Threshold Analysis + SHAP | `11` | Threshold analysis, SHAP explainability, generalisation checks |
| Streamlit model Save | `12` | Saving the model for streamlit deployment |

---

## Features

34 features used in the final model — all available at publication time:

**Numerical (7):** `publication_year`, `referenced_works_count`, `unique_authors_count`, `countries_distinct_count`, `unique_institutions_count`, `funder_count`, `sdg_count`

**Binary flags (4):** `sdg_4`, `references_missing`, `countries_missing`, `institutions_missing`

**OHE (23):** `publication_type` × 5, `oa_status` × 6, `topic_name` × 10, `language` × 2

> **Missingness flags:** ~25% of papers have zeros in `referenced_works_count`, `countries_distinct_count` and `unique_institutions_count` confirmed as OpenAlex indexing gaps, not genuine zeros. Binary flags were added to distinguish true zeros from missing data.

---

## Data

~293,000 AI research papers (2015–2024) across 10 topics:

| Topic | Papers | High Impact Rate |
|---|---|---|
| Natural Language Processing Techniques | 53,790 | 10.2% |
| Topic Modeling | 48,748 | 22.9% |
| Neural Networks and Applications | 34,499 | 8.8% |
| Anomaly Detection Techniques and Applications | 28,191 | 18.1% |
| Privacy-Preserving Technologies in Data | 27,136 | 25.3% |
| Sentiment Analysis and Opinion Mining | 26,018 | 20.5% |
| Quantum Computing Algorithms and Architecture | 24,263 | 23.7% |
| Metaheuristic Optimization Algorithms Research | 21,361 | 27.6% |
| Speech Recognition and Synthesis | 21,175 | 15.8% |
| Evolutionary Algorithms and Applications | 7,864 | 11.3% |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.10** | Modelling notebooks (pycaret-env) |
| **Python 3.13** | Data pull and EDA notebooks (base env) |
| **PyCaret 3.3.2** | Model comparison and tuning |
| **scikit-optimize** | Bayesian hyperparameter search |
| **SHAP** | Model explainability |
| **Plotly** | Visualisation |
| **Streamlit** | Interactive web app |
| **pyalex** | OpenAlex API client |
| **pandas, numpy, scikit-learn** | Data processing |

---

## Project Structure

```
├── data/
│   ├── OpenAlex/                              # Raw and cleaned datasets
│   └── features/                             # Train/test feature matrices
├── models/
│   └── gbc_tuned_streamlit.pkl               # Final tuned model ✓
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
│   ├── 11_AI_Papers_threshold_and_SHAP.ipynb
│   └── 12_AI_Papers_streamlit_model_save.ipynb
├── streamlit_app_final.py                    # Interactive prediction app
├── project_log.md                            # Detailed session log
├── modelling_overview.md                     # Modelling decisions
├── feature_overview.md                       # Feature notes
├── requirements.txt 
└── README.md
```

---

## Known Limitations

| Limitation | Severity | Status |
|---|---|---|
| **Citation maturity bias** — cumulative citation target disadvantages recent papers. Partially mitigated in the app via years-since-publication input. | Medium | App-level workaround live; model-level fix in development |
| **Metadata only** — no full text, no abstract content, no author reputation | Medium | Planned (see Future Work) |
| **10 AI topics only** — does not generalise to other fields without retraining | Medium | Planned |
| **OpenAlex coverage gaps** — ~25% of papers have missing country/institution/reference data | Low | Addressed via missingness flags |

---

## Future Work

- **First-year citations target** — redefine target using `first_year_citations` (already in dataset) to remove citation maturity bias at the model level - citation rate in the first year after the publication
- **Venue quality features** — CORE conference ranking, Scimago journal quartile
- **Abstract text features** — OpenAlex returns abstract inverted index, not yet used
- **Author reputation features** — h-index at time of publication (requires historical snapshot data)
- **Referenced works citation quality** — blocked by scale (1.7M unique referenced work IDs in dataset)

---

## Data Source

All data pulled from [OpenAlex](https://openalex.org/) — an open, fully free index of global research outputs. Volume statistics cited from the [Stanford AI Index 2025 Report](https://hai.stanford.edu/ai-index/2025-ai-index-report/research-and-development).

## Author

**Anna Igumnova**
Architect transitioning into data science.
This project was built combining an interest in academic research systems with applied ML.

[LinkedIn](https://www.linkedin.com/in/anna-igumnova-1248055b/) • [GitHub](https://github.com/AnnaIgumnova)
