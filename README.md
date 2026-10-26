# Disease Prediction from Medical Data Using Machine Learning

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/code%20license-MIT-green) ![Purpose](https://img.shields.io/badge/purpose-educational-orange)

> ⚠️ **Disclaimer:** This project is for **educational purposes only**. It is not a medical device or diagnostic system and must not be used to make health decisions.

## Overview and Objective
A beginner-friendly machine learning project that compares four supervised classification algorithms on a public medical dataset, with a focus on evaluating them in a way that suits medical problems (not just accuracy).

## Problem Statement
Given measurements computed from digitised images of cell nuclei from a breast mass, predict whether the tumour is **malignant** (disease = 1) or **benign** (0).

## Dataset
**Breast Cancer Wisconsin (Diagnostic)** - UCI Machine Learning Repository, 569 samples, 30 numeric features, target `diagnosis` (M/B).
- Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (DOI 10.24432/C5DW2B), licence CC BY 4.0
- Creators: W. Wolberg, O. Mangasarian, N. Street, W. Street
- Included as `data/disease_dataset.csv` (exported from the copy bundled with scikit-learn; see [`data/DATA_SOURCE.md`](data/DATA_SOURCE.md) for details and transformations).
- Quality checks: 0 missing values, 0 duplicates, no invalid values; class split 357 benign / 212 malignant (moderate imbalance, handled with a stratified split and recall/F1 metrics).
- Contains no patient-identifying information.

## Technologies Used
Python, pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer), XGBoost, matplotlib, seaborn, Jupyter, joblib.

## Algorithms Implemented
1. Logistic Regression
2. Support Vector Machine (RBF kernel)
3. Random Forest
4. XGBoost

Preprocessing (median imputation, standard scaling for LR/SVM) lives inside scikit-learn `Pipeline`s, so it is fitted on training data only (no data leakage). Random seed: 42.

## Project Structure
```
disease-prediction-ml/
├── data/
│   ├── disease_dataset.csv
│   └── DATA_SOURCE.md
├── notebooks/
│   └── disease_prediction.ipynb
├── src/
│   └── train_model.py
├── models/
│   └── .gitkeep
├── reports/
│   └── model_comparison.csv
├── images/                # charts saved by the notebook
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Installation
```bash
git clone <your-repo-url>
cd disease-prediction-ml
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run the Notebook
From the project root:
```bash
jupyter notebook notebooks/disease_prediction.ipynb
```
Then **Kernel → Restart & Run All**. Charts are saved to `images/`.

## How to Run the Python Script
From the project root:
```bash
python src/train_model.py
```
This trains all models, prints a comparison table, and writes `reports/model_comparison.csv`, `reports/best_model_report.txt` and `models/best_model.joblib`.

## Evaluation Metrics and Results
Test set = 20% stratified hold-out (114 samples: 72 benign, 42 malignant). Positive class = malignant. `CV_*` columns are 5-fold cross-validation means on the training set only. Rows are ordered by the selection rule (CV recall, then CV F1).

| Model               |   Accuracy |   Precision |   Recall |     F1 |   ROC_AUC |   False_Negatives |   False_Positives |   CV_Recall_Mean |   CV_F1_Mean |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|------------------:|------------------:|-----------------:|-------------:|
| Logistic Regression |     0.9649 |      0.9750 |   0.9286 | 0.9512 |    0.9960 |                 3 |                 1 |           0.9529 |       0.9640 |
| XGBoost             |     0.9561 |      1.0000 |   0.8810 | 0.9367 |    0.9937 |                 5 |                 0 |           0.9529 |       0.9614 |
| SVM (RBF)           |     0.9737 |      1.0000 |   0.9286 | 0.9630 |    0.9947 |                 3 |                 0 |           0.9471 |       0.9610 |
| Random Forest       |     0.9737 |      1.0000 |   0.9286 | 0.9630 |    0.9944 |                 3 |                 0 |           0.9353 |       0.9461 |

(Produced by `src/train_model.py`; the notebook reproduces the same values.)

## Key Findings
- All four models separate the classes well (ROC-AUC about 0.99).
- Accuracy alone is misleading: models with similar accuracy still miss different numbers of malignant cases (see `False_Negatives`). No model reached 100% recall.
- In medical screening a **false negative** (missed disease) is usually costlier than a **false positive** (extra tests), so recall is emphasised. Models were selected by cross-validated recall then F1, not by accuracy.
- Logistic Regression and XGBoost tie on cross-validated recall (0.9529); Logistic Regression wins the F1 tie-break. Differences are tiny and the test set is small (a single patient changes metrics noticeably), so **the ranking is not conclusive**.

## Limitations and Ethical Considerations
- Small single-source dataset, no external validation; results may not generalise.
- Uses pre-extracted measurements, not raw images or clinical history.
- Minimal hyperparameter tuning; default 0.5 decision threshold (lowering it would trade precision for recall).
- Not clinically validated, not calibrated for real-world prevalence, no fairness/bias audit.
- Medical AI requires regulatory approval, clinical validation and human oversight. Do not use this for real patients.

## Testing Performed
Files verified; CSV loads; notebook executed top-to-bottom without errors; script executed successfully; both were run in a fresh virtual environment installed from `requirements.txt`. Not tested: Windows/macOS, other Python versions.

## License and Attribution
Code: MIT License (see `LICENSE`). Dataset: CC BY 4.0 by its original authors; cite Wolberg, Mangasarian, Street & Street (1993), UCI Machine Learning Repository, https://doi.org/10.24432/C5DW2B.
