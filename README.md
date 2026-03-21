# ai-paper-impact-predictor
Machine learning classifier to predict research paper citation impact using OpenAlex data

# AI Paper Impact Predictor

> Predicting whether an AI research paper will become highly cited using machine learning.

## Overview

This project builds a machine learning pipeline to predict whether an AI research paper will become highly cited — defined as reaching the top 10% of citations within its subfield and year.

The dataset contains ~293,000 AI research papers published between 2015–2024, pulled from the [OpenAlex API](https://openalex.org/) across 10 topics including NLP, Neural Networks, Quantum Computing and Privacy-Preserving Technologies.

## Pipeline

| Stage | Description |
|---|---|
| Data Pull | OpenAlex API via `pyalex` |
| Data Cleaning | Deduplication, junk title removal, type consolidation |
| EDA | Univariate, bivariate analysis, correlation matrix |
| Feature Engineering | Encoding, train/test split, artefact saving |
| Modelling | PyCaret model comparison + SHAP explainability |

## Use Cases

- **Academic publishers** — identify high-impact papers early for indexing priority
- **Research funders** — allocate grants to emerging fields before they peak
- **University strategy offices** — identify growing AI subfields for faculty hiring

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

## Project Structure
```
├── data/
│   ├── OpenAlex/        # Raw and cleaned data
│   └── features/        # Train/test feature matrices
├── models/              # Saved model artefacts
├── notebooks/           # Jupyter notebooks
└── project_log.md       # Detailed log of all steps
```
