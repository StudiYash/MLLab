"""
Breast Cancer Prediction project utilities.

This module provides:
- a function to load the breast cancer dataset as a pandas DataFrame,
- a function to train a CatBoostClassifier with a standard train/test split,
- a function to perform prediction from a partial feature dictionary
  (missing features are filled with mean values).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier

@dataclass
class BreastCancerModelBundle:
    """Container for the trained model and its metadata."""
    model: CatBoostClassifier
    feature_names: List[str]
    metrics: Dict[str, float]
    X_reference: pd.DataFrame  # used for mean-based imputation in inference

def load_breast_cancer_dataset() -> pd.DataFrame:
    """
    Loads the breast cancer dataset from sklearn.
    Returns a DataFrame with all features plus a 'target' column.
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def train_breast_cancer_model(test_size: float = 0.2, random_state: int = 42) -> BreastCancerModelBundle:
    """
    Trains a CatBoostClassifier on the breast cancer dataset.
    Returns a bundle containing the model, metrics, and reference data.
    """
    df = load_breast_cancer_dataset()
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_state,
        verbose=False
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    return BreastCancerModelBundle(
        model=model,
        feature_names=list(X.columns),
        metrics=metrics,
        X_reference=X_train  # Using training data as reference
    )

def predict_from_partial_features(bundle: BreastCancerModelBundle, input_data: Dict[str, float]) -> Dict[str, object]:
    """
    Performs inference using a partial feature dictionary.
    Missing features are imputed with the mean from the reference data.
    """
    # Start with mean values
    feature_means = bundle.X_reference.mean()
    input_df = pd.DataFrame([feature_means])
    
    # Override with provided values
    for feature, value in input_data.items():
        if feature in input_df.columns:
            input_df[feature] = value
        else:
            raise ValueError(f"Unknown feature: {feature}")

    # Ensure column order matches training data
    input_df = input_df[bundle.feature_names]

    # Predict
    pred_class = bundle.model.predict(input_df)[0]
    pred_proba = bundle.model.predict_proba(input_df)[0]
    
    # Map prediction to label (0 = malignant, 1 = benign in sklearn dataset)
    # Note: sklearn load_breast_cancer target_names are ['malignant', 'benign']
    # So 0 -> malignant, 1 -> benign.
    
    label = "benign" if pred_class == 1 else "malignant"
    probability = pred_proba[pred_class]

    return {
        "raw_prediction": int(pred_class),
        "label": label,
        "probability": float(probability)
    }
