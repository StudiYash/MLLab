"""
Titanic Survival Prediction project utilities.

This module provides:
- functions to load and preprocess the Titanic dataset,
- a function to train a RandomForestClassifier,
- a function to run inference for a single passenger.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@dataclass
class TitanicModelBundle:
    """
    Container for the trained Titanic model and metadata.
    """
    model: RandomForestClassifier
    feature_names: List[str]           # final set of features used by the model
    feature_means: Dict[str, float]    # for numeric defaults at inference time
    feature_modes: Dict[str, Any]      # for categorical defaults at inference time
    metrics: Dict[str, float]          # e.g. {"accuracy": ...}

def get_titanic_train_path() -> Path:
    """
    Returns the path to train.csv inside data/raw/titanic_survival_prediction.
    """
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "raw" / "titanic_survival_prediction" / "train.csv"

def load_titanic_train() -> pd.DataFrame:
    """
    Loads train.csv and returns a pandas DataFrame.
    Raises FileNotFoundError with a clear message if missing.
    """
    path = get_titanic_train_path()
    if not path.exists():
        raise FileNotFoundError(f"Titanic train dataset not found at {path}")
    return pd.read_csv(path)

def preprocess_titanic_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float], Dict[str, Any]]:
    """
    Cleans and prepares the Titanic dataset for modeling.

    Steps:
    1. Drop 'Cabin', 'Ticket', 'PassengerId', 'Name'.
    2. Impute 'Age' with median.
    3. Impute 'Fare' with median.
    4. Impute 'Embarked' with mode.
    5. Map 'Sex': {'male': 0, 'female': 1}.
    6. Map 'Embarked': {'S': 0, 'C': 1, 'Q': 2}.
    7. Select features: ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"].
    
    Returns:
        X (DataFrame of numeric features)
        y (Series of labels)
        feature_means (dict for numeric defaults)
        feature_modes (dict for categorical defaults)
    """
    df_clean = df.copy()
    
    # 1. Drop columns
    cols_to_drop = ["Cabin", "Ticket", "PassengerId", "Name"]
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])
    
    # 2. Impute Age (median)
    if "Age" in df_clean.columns:
        df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())
        
    # 3. Impute Fare (median)
    if "Fare" in df_clean.columns:
        df_clean["Fare"] = df_clean["Fare"].fillna(df_clean["Fare"].median())
        
    # 4. Impute Embarked (mode)
    if "Embarked" in df_clean.columns:
        mode_embarked = df_clean["Embarked"].mode()[0]
        df_clean["Embarked"] = df_clean["Embarked"].fillna(mode_embarked)
        
    # 5. Map Sex
    if "Sex" in df_clean.columns:
        df_clean["Sex"] = df_clean["Sex"].map({"male": 0, "female": 1})
        
    # 6. Map Embarked
    if "Embarked" in df_clean.columns:
        df_clean["Embarked"] = df_clean["Embarked"].map({"S": 0, "C": 1, "Q": 2})
        
    # 7. Select features
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    
    # Ensure all exist
    missing = [c for c in feature_cols if c not in df_clean.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
        
    X = df_clean[feature_cols]
    
    if "Survived" in df_clean.columns:
        y = df_clean["Survived"]
    else:
        # If processing test set without labels, y is None or empty
        y = pd.Series()
        
    # Compute defaults for inference
    feature_means = {}
    feature_modes = {}
    
    # For this simple set, all are numeric after mapping, so we can use means/medians.
    # But let's stick to the logic: Sex and Embarked were categorical.
    # We already mapped them to numeric, so we can treat them as numeric for default purposes (e.g. mode mapped to int).
    
    for col in X.columns:
        feature_means[col] = float(X[col].mean())
        # For categorical-like, mode might be better, but mean is safe for fallback if we round it or just use it.
        # Let's store modes for Sex and Embarked specifically if we want 'categorical' defaults, 
        # but since they are already int, mean is fine.
        # Actually, let's compute mode for Sex/Embarked/Pclass just in case.
        feature_modes[col] = X[col].mode()[0]

    return X, y, feature_means, feature_modes

def train_titanic_model(
    test_size: float = 0.2,
    random_state: int = 0,
) -> TitanicModelBundle:
    """
    Trains a RandomForestClassifier on the Titanic dataset.
    Returns a TitanicModelBundle with model and metadata.
    """
    df = load_titanic_train()
    X, y, feature_means, feature_modes = preprocess_titanic_data(df)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return TitanicModelBundle(
        model=model,
        feature_names=X.columns.tolist(),
        feature_means=feature_means,
        feature_modes=feature_modes,
        metrics={"accuracy": accuracy}
    )

def predict_survival(
    bundle: TitanicModelBundle,
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict survival from a partial feature dict.
    
    - input_data: dict of features.
      - Sex: "male" or "female" (will be mapped)
      - Embarked: "S", "C", "Q" (will be mapped)
      - Others: numeric
      
    Returns:
        {
            "predicted_label": int (0 or 1),
            "predicted_class_name": str ("Not Survived" or "Survived"),
            "probability_survived": float
        }
    """
    # Prepare input row
    # We need to handle the mapping here because input_data might have raw strings ("male", "S")
    # But our bundle.feature_names expects mapped integers.
    
    # Helper to map input
    mapped_input = input_data.copy()
    
    if "Sex" in mapped_input and isinstance(mapped_input["Sex"], str):
        mapped_input["Sex"] = {"male": 0, "female": 1}.get(mapped_input["Sex"].lower(), 0) # Default to 0 (male) if unknown
        
    if "Embarked" in mapped_input and isinstance(mapped_input["Embarked"], str):
        mapped_input["Embarked"] = {"S": 0, "C": 1, "Q": 2}.get(mapped_input["Embarked"].upper(), 0) # Default to 0 (S)
        
    input_values = []
    for feature in bundle.feature_names:
        if feature in mapped_input:
            input_values.append(float(mapped_input[feature]))
        else:
            # Fallback
            # Use mode for categorical-ish, mean for continuous
            if feature in ["Pclass", "Sex", "Embarked", "SibSp", "Parch"]:
                input_values.append(float(bundle.feature_modes.get(feature, 0)))
            else:
                input_values.append(float(bundle.feature_means.get(feature, 0.0)))
                
    # Reshape
    X_input = pd.DataFrame([input_values], columns=bundle.feature_names)
    
    # Predict
    pred_label = int(bundle.model.predict(X_input)[0])
    prob_survived = float(bundle.model.predict_proba(X_input)[0][1])
    
    return {
        "predicted_label": pred_label,
        "predicted_class_name": "Survived" if pred_label == 1 else "Not Survived",
        "probability_survived": prob_survived
    }

if __name__ == "__main__":
    print("Training Titanic model...")
    bundle = train_titanic_model()
    print(f"Model trained. Validation Accuracy: {bundle.metrics['accuracy']:.4f}")
    
    print("Testing prediction...")
    example = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }
    result = predict_survival(bundle, example)
    print("Prediction result:", result)
