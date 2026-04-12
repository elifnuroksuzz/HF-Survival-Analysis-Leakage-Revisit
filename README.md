# Heart Failure Survival Prediction: A Methodological Revisit

This repository provides a rigorous re-evaluation of Heart Failure (HF) survival prediction models. Beyond standard performance metrics, this project focuses on **data leakage identification**, **survival-specific modeling**, and **cross-dataset external validation**.

## 1. Project Overview
Most heart failure datasets include features that inadvertently introduce "time leakage," leading to over-optimistic model performance. This project:
- Replicates baseline studies to establish a ground truth.
- Formally proves data leakage in the `time` feature.
- Implements specialized survival models (Cox PH, Random Survival Forests, DeepSurv).
- Validates findings using the **MIMIC-III** database for external robustness.
- Explains model decisions using **SHAP** and assesses algorithmic fairness.

## 2. Repository Structure
The project is organized into modular Jupyter Notebooks for clarity and reproducibility:

```text
/HF_Revisit_Project/
├── 00_setup_and_data.ipynb        # Data ingestion, cleaning, and EDA
├── 01_replication_chicco.ipynb    # Replication of the original benchmark study
├── 02_time_leakage_proof.ipynb    # Mathematical/Empirical proof of time leakage
├── 03_survival_models.ipynb       # Training CoxPH, RSF, and DeepSurv models
├── 04_external_validation.ipynb   # Generalization test on MIMIC-III dataset
├── 05_shap_and_fairness.ipynb     # Model interpretability (XAI) and fairness audits
├── utils.py                       # Shared helper functions for metrics and plotting
└── /results/                      # Model weights, figures, and evaluation logs
