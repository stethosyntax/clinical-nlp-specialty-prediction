"""
src/train.py
------------
Model training and evaluation utilities for clinical NLP pipeline.
Author: Aruna Kunche
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score


def build_lr_pipeline(max_features: int = 10000) -> Pipeline:
    """
    Build TF-IDF + Logistic Regression pipeline.

    Parameters
    ----------
    max_features : int
        Maximum number of TF-IDF features.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            C=1.0
        ))
    ])


def build_svc_pipeline(max_features: int = 15000) -> Pipeline:
    """
    Build TF-IDF + Linear SVC pipeline.

    Parameters
    ----------
    max_features : int
        Maximum number of TF-IDF features.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ('clf', LinearSVC(
            C=1.0,
            class_weight='balanced',
            max_iter=2000,
            random_state=42
        ))
    ])


def evaluate_pipeline(pipeline: Pipeline, X_test, y_test,
                       model_name: str = 'Model') -> dict:
    """
    Evaluate a trained pipeline and print results.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline.
    X_test : array-like
        Test features.
    y_test : array-like
        True labels.
    model_name : str
        Display name for output.

    Returns
    -------
    dict with accuracy, weighted_f1, predictions
    """
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='weighted')

    print(f'\n{"=" * 55}')
    print(f'{model_name.upper()}')
    print(f'{"=" * 55}')
    print(f'Accuracy    : {acc:.4f}  ({acc*100:.1f}%)')
    print(f'Weighted F1 : {f1:.4f}')
    print(f'\nClassification Report:')
    print(classification_report(y_test, y_pred))

    return {'accuracy': acc, 'weighted_f1': f1, 'predictions': y_pred}


def compare_models(results: dict) -> pd.DataFrame:
    """
    Return a DataFrame comparing model results.

    Parameters
    ----------
    results : dict
        {model_name: {'accuracy': float, 'weighted_f1': float}}

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    best_acc = max(v['accuracy'] for v in results.values())
    for name, metrics in results.items():
        rows.append({
            'Model'       : name,
            'Accuracy'    : round(metrics['accuracy'],    4),
            'Weighted_F1' : round(metrics['weighted_f1'], 4),
            'Best'        : '✓' if metrics['accuracy'] == best_acc else ''
        })
    return pd.DataFrame(rows)
