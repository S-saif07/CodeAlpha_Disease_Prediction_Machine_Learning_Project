"""Train and compare classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset.

Run from the project root:
    python src/train_model.py

Outputs:
    reports/model_comparison.csv   - metrics for every model (test set + cross-validation)
    reports/best_model_report.txt  - classification report + confusion matrix of the selected model
    models/best_model.joblib       - the fitted selected pipeline (git-ignored)

EDUCATIONAL USE ONLY - not a medical diagnostic tool.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # XGBoost is optional
    XGBClassifier = None

RANDOM_STATE = 42
# Paths are built relative to this file, so the script works on any computer.
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "disease_dataset.csv"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"


def load_data():
    df = pd.read_csv(DATA_PATH)
    y = (df["diagnosis"] == "M").astype(int)  # 1 = malignant (the "disease" class)
    X = df.drop(columns=["diagnosis"])
    return X, y


def build_models(X):
    """Each model is a Pipeline so imputing/scaling are fitted on training data only."""
    num_cols = X.select_dtypes(include="number").columns.tolist()

    def prep(scale):
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        return ColumnTransformer([("num", Pipeline(steps), num_cols)])

    models = {
        "Logistic Regression": Pipeline([("prep", prep(True)),
                                         ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
        "SVM (RBF)": Pipeline([("prep", prep(True)),
                               ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline([("prep", prep(False)),
                                   ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline([("prep", prep(False)),
                                      ("clf", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                                            eval_metric="logloss", random_state=RANDOM_STATE))])
    return models


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for name, pipe in build_models(X).items():
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["recall", "f1"])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC_AUC": roc_auc_score(y_test, proba),
            "False_Negatives": int(fn), "False_Positives": int(fp),
            "CV_Recall_Mean": cv_res["test_recall"].mean(),
            "CV_F1_Mean": cv_res["test_f1"].mean(),
        })
        fitted[name] = pipe

    results = pd.DataFrame(rows).round(4)
    # Model selection uses CROSS-VALIDATION on the training data only (never the test set),
    # prioritising recall (fewer missed cases), then F1. Accuracy is not used for selection.
    results = results.sort_values(["CV_Recall_Mean", "CV_F1_Mean"], ascending=False).reset_index(drop=True)
    best = results.loc[0, "Model"]

    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    pred = fitted[best].predict(X_test)
    report = (f"Selected model: {best}\n(selected by cross-validated recall, then F1)\n\n"
              f"{classification_report(y_test, pred, target_names=['Benign', 'Malignant'])}\n"
              f"Confusion matrix [rows=true, cols=predicted] (Benign, Malignant):\n"
              f"{confusion_matrix(y_test, pred)}\n\nEducational use only - not a diagnostic tool.\n")
    (REPORTS_DIR / "best_model_report.txt").write_text(report)
    joblib.dump(fitted[best], MODELS_DIR / "best_model.joblib")

    print(results.to_string(index=False))
    print("\n" + report)


if __name__ == "__main__":
    main()
