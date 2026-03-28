# Modelling Overview — AI Paper Impact Predictor

## Objective
Binary classification to predict whether an AI research paper will reach the
top 10% of citations within its subfield and year, using only metadata available
at publication time.

---

## Model 1 — First Model (Notebook 04)
**Feature set:** 30 features (7 numerical + 23 OHE)
**Data:** 293,045 papers, 80/20 train/test split, stratified, random_state=42
**Selection:** Top performing model from PyCaret `compare_models()` ranked by F1

### Results
| Metric | Value |
|---|---|
| F1 | 0.54 |
| AUC | 0.84 |
| Recall | 0.61 |
| Precision | 0.49 |
| True Positives | 6,421 |
| False Negatives | 4,139 |

### Decision
GBC selected as top performing model by F1. No tuning — enriched data pull
planned first.

---

## Model 2 — Enriched Model (Notebook 08)
**New features added:** unique_authors_count, unique_institutions_count,
institution_edu_count, funder_count, award_count, sdg_count, sdg_max_score,
sdg_1–sdg_17
**Feature set:** 54 features (31 numerical + 23 OHE)
**Selection:** Top performing model from PyCaret `compare_models()` ranked by F1

### Results
| Metric | GBC | XGBoost |
|---|---|---|
| F1 | 0.53 | 0.48 |
| AUC | 0.84 | 0.85 |
| Recall | 0.53 | 0.39 |
| Precision | 0.53 | 0.63 |

### Key Findings
- Recall dropped from 0.61 → 0.53 — enriched features added noise
- AUC unchanged at 0.84 — discriminative power not affected
- XGBoost feature importance distorted — sdg_count dominant despite ρ=0.04
- funder_count and sdg_count confirmed as new signals in GBC

### Decision
No tuning on enriched model — feature trimming needed first to remove noise.
XGBoost ruled out — precision-heavy and feature importance unreliable.

---

## Model 3 — Trimmed Model (Notebook 10)

### Feature Trimming (Notebook 09)
Two categories of features dropped based on feature importance analysis
across GBC and XGBoost:

**Correlated pairs — weaker feature dropped:**
| Dropped | Kept | Correlation |
|---|---|---|
| `authorship_count` | `unique_authors_count` | 0.99 |
| `institution_edu_count` | `unique_institutions_count` | 0.88 |
| `award_count` | `funder_count` | 0.83 |
| `sdg_max_score` | `sdg_count` | 0.92 |

**Near-zero importance across both models:**
`sdg_1`–`sdg_17` (excl. `sdg_4`), `primary_topic_score`,
`keyword_count`, `is_oa`

**Missingness flags added:**
25% of papers have zeros in three key features due to OpenAlex coverage gaps.
Manual inspection confirmed these are indexing gaps, not genuine zeros.
Statistical tests confirmed flags have strongest association with target
of all binary features (Cramér's V 0.19–0.22).

| Flag | Coverage |
|---|---|
| `references_missing` | 24.9% of papers |
| `countries_missing` | 25.4% of papers |
| `institutions_missing` | 25.1% of papers |

**Final feature set: 54 → 34 features**

### Model Comparison
PyCaret `compare_models()` ranked by F1 — GBC top performing model.
Three additional models evaluated in depth:

**XGBoost and CatBoost** — investigated due to highest AUC (0.85) across
all runs, suggesting strong discriminative potential. Both ruled out:
XGBoost and CatBoost default to precision-heavy predictions (recall 0.38-0.40)
unsuitable for the publisher use case, and feature importance is distorted
in both models.

**AdaBoost** — investigated due to consistently highest recall (0.64) across
all three compare_models() runs, making it a candidate for the recall-focused
publisher use case. Ruled out: generates 9,319 false positives vs GBC's 6,396
— too noisy to be operationally useful. Feature importance distorted —
`sdg_count` dominant despite near-zero statistical association with target.

| Model | F1 | AUC | Recall | Precision | Selected |
|---|---|---|---|---|---|
| GBC | 0.54 | 0.84 | 0.60 | 0.50 | ✅ |
| AdaBoost | 0.52 | 0.82 | 0.64 | 0.42 | ❌ Too many false positives |
| XGBoost | 0.49 | 0.85 | 0.40 | 0.61 | ❌ Precision-heavy, distorted FI |
| CatBoost | 0.47 | 0.85 | 0.38 | 0.62 | ❌ Precision-heavy, distorted FI |

### Model Selection — GBC confirmed
GBC selected as the model to tune based on three independent lines of evidence:

**1. Performance**
Best F1 (0.54) and best precision/recall balance across all runs.
Recall recovered from 0.53 → 0.60 after trimming.

**2. Feature importance consistency**
Only model whose feature rankings are consistent with statistical evidence:
- `referenced_works_count` dominant (importance 0.40) — confirmed by
  EDA (ρ=0.39), Mann-Whitney effect size (0.573), and GBC importance (0.40)
- `sdg_count` correctly ranked low — consistent with Cramér's V (0.03)
  and Mann-Whitney effect size (0.050)
- XGBoost, AdaBoost and CatBoost all over-weight `sdg_count` despite
  near-zero statistical association with target

**3. Business case alignment**
AdaBoost higher recall (0.64) but generates 9,319 false positives vs GBC's
6,396 — too noisy for publisher and funder use cases.
CatBoost highest precision (0.62) but recall only 0.38 — misses too many
high-impact papers.

---

## Model 4 — Tuned GBC (Notebook 10) ✅ COMPLETE
**Base model:** Top performing model from PyCaret `compare_models()` —
GBC trimmed + missingness flags (34 features)
**Tuning:** Bayesian hyperparameter search, n_iter=50, optimize='F1'
**Search library:** scikit-optimize
**Runtime:** ~5 hours
**Saved:** `models/gbc_tuned.pkl`

### Tuning Results — CV

| Metric | Pre-Tuning | Tuned |
|---|---|---|
| F1 | 0.5405 | 0.5523 |
| AUC | 0.8406 | 0.8462 |
| Recall | 0.5859 | 0.6084 |
| Precision | 0.5016 | 0.4867 |

### Final Model Comparison — All GBC Runs (Test Set)

| Metric | First Model | Enriched | Trimmed+Flags | Tuned GBC |
|---|---|---|---|---|
| F1 | 0.5428 | 0.5257 | 0.5405 | 0.5519 |
| AUC | 0.8413 | 0.8412 | 0.8406 | 0.8446 |
| Recall | 0.5961 | 0.5195 | 0.5859 | 0.6442 |
| Precision | 0.4982 | 0.5323 | 0.5016 | 0.4827 |
| True Positives | 6,421 | 5,563 | 6,305 | 6,803 |
| False Negatives | 4,139 | 4,997 | 4,255 | 3,757 |

Tuned GBC is the best model across all runs — highest F1, AUC, recall
and true positives. Catches 382 more high-impact papers than the first
model and 498 more than trimmed+flags pre-tuning.

### Model Selection Confirmed
GBC selected as the final model based on:
1. Best F1 and recall across all runs
2. Feature importance consistent with statistical evidence —
   `referenced_works_count` dominant across EDA, Mann-Whitney,
   GBC importance and SHAP
3. Best precision/recall balance for publisher and funder use cases
4. Good generalisation — train/test gap ~0.02 across all metrics

### Known Limitation — Citation Maturity Bias
`publication_year` is the second most important feature but reflects
bias in the target variable — cumulative citations disadvantage recent
papers. Performance degrades gradually from F1 0.64 (2015-2016) to
F1 0.49 (2024). Model remains useful for recent papers — catching 59%
of high-impact 2024 papers at publication time.

### Future Work
- Redefine target using first-year citations — removes citation maturity
  bias, `first_year_citations` already available in dataset
- Temporal train/test split (2015-2021 train, 2022-2024 test) — more
  realistic evaluation, best done alongside target redefinition
- Venue quality features (CORE ranking, Scimago journal quartile)
- Abstract text features
- Author reputation features (h-index at time of publication)
- Referenced works citation quality (blocked by 1.7M unique ID scale)