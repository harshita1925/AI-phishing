"""
model_engine.py

INFERENCE / LEARNING ENGINE
============================
This is where the paper's actual AI method lives: train Logistic
Regression as a baseline and a Random Forest Classifier as the main
model (Section III-C), evaluate both (Section IV-A/B), then run Root
Cause Analysis via Random Forest feature importances (Section III-D)
and turn that into a plain-English explanation for a single website
(Explainable AI, Section IV-C).

No Streamlit / UI code appears in this file -- it is pure ML logic and
can be reused, unit-tested, or run from the command line independently
of the app.
"""

from typing import Dict, List, TypedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from feature_schema import FEATURES, ROOT_CAUSE_EXPLANATIONS

RANDOM_STATE = 42


class TrainingResult(TypedDict):
    lr_accuracy: float
    rf_accuracy: float
    rf_model: RandomForestClassifier
    confusion_matrix: np.ndarray
    feature_importances: Dict[str, float]
    n_train: int
    n_test: int


def train_models(df: pd.DataFrame) -> TrainingResult:
    """Train Logistic Regression (baseline) + Random Forest (main model)
    on an 80/20 split, exactly as described in the paper's Section III-B.
    """
    X = df[FEATURES]
    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_accuracy = accuracy_score(y_test, lr.predict(X_test))

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    cm = confusion_matrix(y_test, rf_pred)

    importances = dict(zip(FEATURES, rf.feature_importances_))

    return TrainingResult(
        lr_accuracy=lr_accuracy,
        rf_accuracy=rf_accuracy,
        rf_model=rf,
        confusion_matrix=cm,
        feature_importances=importances,
        n_train=len(X_train),
        n_test=len(X_test),
    )


class Explanation(TypedDict):
    verdict: str            # "PHISHING" or "LEGITIMATE"
    confidence: float       # 0-100
    top_reasons: List[Dict[str, object]]  # [{feature, importance, explanation}]


def explain_prediction(rf_model: RandomForestClassifier,
                        feature_importances: Dict[str, float],
                        sample: Dict[str, int]) -> Explanation:
    """Classify ONE website and explain why -- the Root Cause Analysis /
    Explainable AI step described in the paper's Section V.
    """
    x = pd.DataFrame([sample])[FEATURES]
    pred = rf_model.predict(x)[0]
    proba = rf_model.predict_proba(x)[0]
    confidence = float(proba[list(rf_model.classes_).index(pred)]) * 100

    verdict = "PHISHING" if pred == 1 else "LEGITIMATE"

    top_reasons: List[Dict[str, object]] = []
    if pred == 1:
        suspicious = [f for f in FEATURES if sample[f] == 1]
        suspicious.sort(key=lambda f: feature_importances[f], reverse=True)
        for f in suspicious[:5]:
            top_reasons.append({
                "feature": f,
                "importance": feature_importances[f],
                "explanation": ROOT_CAUSE_EXPLANATIONS[f],
            })

    return Explanation(verdict=verdict, confidence=confidence, top_reasons=top_reasons)
