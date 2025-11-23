"""
Diabetes Prediction project utilities.

This module provides:
- functions to load and preprocess the diabetes dataset,
- a function to train multiple classifiers (RFC, DT, XGB, SVC),
- a function to run inference for a single patient.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

@dataclass
class DiabetesModelBundle:
    """
    Container for the trained diabetes models and metadata.
    """
    models: Dict[str, Any]               # e.g. {"rfc": RandomForestClassifier, "xgb": XGBClassifier, ...}
    feature_names: List[str]             # column names, in order
    metrics: Dict[str, Dict[str, float]] # per-model metrics (accuracy, maybe precision/recall if easy)
    feature_means: Dict[str, float]      # for defaulting missing fields in inference

def get_diabetes_csv_path() -> Path:
    """
    Returns the path to diabetes.csv inside data/raw/predicting_diabetes.
    """
    root = Path(__file__).resolve().parents[2]  # repo root
    return root / "data" / "raw" / "predicting_diabetes" / "diabetes.csv"

def load_diabetes_dataset() -> pd.DataFrame:
    """
    Loads diabetes.csv and returns a pandas DataFrame.
    Raises FileNotFoundError with a clear message if missing.
    """
    path = get_diabetes_csv_path()
    if not path.exists():
        raise FileNotFoundError(f"Diabetes dataset not found at {path}")
    return pd.read_csv(path)

def preprocess_diabetes_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
    """
    Cleans and prepares the diabetes dataset for modeling.

    Steps (closely based on the original notebook logic):
      - Treat 0 as missing in the following columns:
        ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
      - Replace those 0 values with NaN.
      - Impute:
          * Glucose, BloodPressure -> mean
          * SkinThickness, Insulin, BMI -> median
      - (Do NOT drop rows; just impute.)
      - Split into:
          * X: all columns except "Outcome"
          * y: "Outcome"
      - Return:
          * X (DataFrame of numeric features)
          * y (Series of labels)
          * feature_means: dict(feature_name -> mean of that feature) 
            for later use as default values in inference.
    """
    df_clean = df.copy()
    
    # Columns where 0 means missing
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    # Replace 0 with NaN
    for col in zero_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        
    # Impute
    # Glucose, BloodPressure -> mean
    for col in ["Glucose", "BloodPressure"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            
    # SkinThickness, Insulin, BMI -> median
    for col in ["SkinThickness", "Insulin", "BMI"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    # Separate X and y
    if "Outcome" not in df_clean.columns:
        raise ValueError("Dataset missing 'Outcome' column")
        
    X = df_clean.drop(columns=["Outcome"])
    y = df_clean["Outcome"]
    
    # Compute feature means for inference defaults
    # Note: Using mean for all features as a generic fallback, 
    # even though we used median for imputation during training.
    # This is for *missing input* during inference, which is slightly different.
    feature_means = X.mean().to_dict()
    
    return X, y, feature_means

def train_diabetes_models(
    test_size: float = 0.33,
    random_state: int = 7,
) -> DiabetesModelBundle:
    """
    Trains multiple classifiers on the diabetes dataset.

    Models to include (simple versions, similar to the notebook):
      - RandomForestClassifier
      - DecisionTreeClassifier
      - XGBClassifier
      - SVC

    Split:
      - Use train_test_split with stratify=y.

    For each model:
      - fit on train
      - compute accuracy on test

    Return:
      - DiabetesModelBundle with:
          * models dict {"rfc": ..., "dt": ..., "xgb": ..., "svc": ...}
          * feature_names
          * metrics per model {"rfc": {"accuracy": ...}, ...}
          * feature_means from preprocess_diabetes_data()
    """
    df = load_diabetes_dataset()
    X, y, feature_means = preprocess_diabetes_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize models
    models = {
        "rfc": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "dt": DecisionTreeClassifier(random_state=random_state),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
        "svc": SVC(random_state=random_state, probability=True) # Default SVC does not have probability=True by default
    }
    
    metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        metrics[name] = {"accuracy": acc}
        
    return DiabetesModelBundle(
        models=models,
        feature_names=X.columns.tolist(),
        metrics=metrics,
        feature_means=feature_means
    )

def predict_diabetes(
    bundle: DiabetesModelBundle,
    input_data: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """
    Run inference for a single patient using all models.

    - input_data is a mapping from feature_name -> value.
    - Use bundle.feature_names as canonical order of features.
    - For each feature:
        * if present in input_data, use the given value
        * else, fall back to bundle.feature_means[feature]
    - Build a single-row feature vector and feed it to each model.

    For each model, return:
      {
        "probability_diabetes": float,   # P(Outcome=1) if model supports predict_proba, else None
        "predicted_label": int,          # 0 or 1
      }

    Overall return structure:
      {
        "rfc": {...},
        "dt": {...},
        "xgb": {...},
        "svc": {...},
      }
    """
    # Build input vector
    input_values = []
    for feature in bundle.feature_names:
        if feature in input_data:
            input_values.append(float(input_data[feature]))
        else:
            input_values.append(float(bundle.feature_means.get(feature, 0.0)))
            
    # Reshape for prediction (1, n_features)
    # Note: Some models warn if feature names are missing when trained with DF.
    # To be safe and consistent, we can create a DataFrame.
    X_input = pd.DataFrame([input_values], columns=bundle.feature_names)
    
    results = {}
    
    for name, model in bundle.models.items():
        # Predict label
        pred_label = int(model.predict(X_input)[0])
        
        # Predict probability if supported
        prob_diabetes = None
        if hasattr(model, "predict_proba"):
            try:
                # predict_proba returns [prob_0, prob_1]
                prob_diabetes = float(model.predict_proba(X_input)[0][1])
            except (AttributeError, IndexError):
                pass
                
        results[name] = {
            "probability_diabetes": prob_diabetes,
            "predicted_label": pred_label
        }
        
    return results
