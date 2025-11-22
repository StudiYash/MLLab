"""
Iris Flower Classification project utilities.

This module provides:
- functions to load and preprocess the Iris dataset from CSV,
- a function to train a RandomForest classifier,
- a function to run inference from a (possibly partial) feature dictionary.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class IrisModelBundle:
    """
    Container for the trained Iris model and its metadata.
    """
    model: Any
    feature_names: List[str]              # numeric feature names in order
    classes: List[str]                    # species names in class-index order
    label_encoder: LabelEncoder
    metrics: Dict[str, float]             # e.g. {"accuracy": 0.95}
    numeric_means: Dict[str, float]       # mean of each numeric feature (for inference defaults)


def get_iris_csv_path() -> Path:
    """
    Returns the path to IrisFlowerClassification.csv inside
    data/raw/iris_flower_classification.
    """
    # app/projects/iris_flowers.py -> repo root is parents[2]
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "raw" / "iris_flower_classification" / "IrisFlowerClassification.csv"


def load_iris_dataset() -> pd.DataFrame:
    """
    Load IrisFlowerClassification.csv and return a cleaned DataFrame.

    Expected columns (Kaggle-style):
        Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
    """
    path = get_iris_csv_path()
    if not path.exists():
        raise FileNotFoundError(f"Iris dataset not found at {path}")

    df = pd.read_csv(path)

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Drop rows with any missing values (if present)
    df = df.dropna()

    return df


def preprocess_iris_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, Dict[str, float]]:
    """
    Preprocess Iris data:

    - Drop 'Id' if present.
    - Use the 4 numeric feature columns:
        ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    - Encode 'Species' -> integers via LabelEncoder.
    - Compute mean of numeric features for later inference defaults.

    Returns:
        X             : DataFrame of numeric features
        y_encoded     : numpy array of encoded labels
        label_encoder : fitted LabelEncoder instance
        numeric_means : dict mapping feature_name -> mean value
    """
    df_clean = df.copy()

    # Drop Id column if present
    if "Id" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Id"])

    # Numeric feature columns (standard Iris schema)
    numeric_features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

    # Ensure all required columns are present
    missing = [col for col in numeric_features + ["Species"] if col not in df_clean.columns]
    if missing:
        raise ValueError(f"Missing expected columns in Iris dataset: {missing}")

    X = df_clean[numeric_features].astype(float)
    y = df_clean["Species"]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Means for default values in inference
    numeric_means = X.mean().to_dict()

    return X, y_encoded, label_encoder, numeric_means


def train_iris_model(
    test_size: float = 0.2,
    random_state: int = 42,
) -> IrisModelBundle:
    """
    Train a RandomForestClassifier on the Iris dataset.

    Returns an IrisModelBundle with model, metrics, and metadata.
    """
    df = load_iris_dataset()
    X, y_encoded, label_encoder, numeric_means = preprocess_iris_data(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    feature_names = list(X.columns)
    classes = label_encoder.classes_.tolist()
    metrics = {"accuracy": float(accuracy)}

    return IrisModelBundle(
        model=model,
        feature_names=feature_names,
        classes=classes,
        label_encoder=label_encoder,
        metrics=metrics,
        numeric_means=numeric_means,
    )


def predict_from_features(
    bundle: IrisModelBundle,
    input_data: Dict[str, float],
) -> Dict[str, object]:
    """
    Predict Iris species from a (possibly partial) feature dict.

    - Uses bundle.feature_names (the numeric columns) as the canonical feature order.
    - For each feature:
        - if present in input_data, use that value
        - else, fall back to bundle.numeric_means[feature]
    - Returns:
        {
            "predicted_class": int,
            "predicted_label": str,
            "confidence": float,
            "probabilities": List[float],
        }
    """
    # Build feature vector with defaults from numeric_means
    values = []
    for feature in bundle.feature_names:
        if feature in input_data:
            values.append(float(input_data[feature]))
        else:
            # fall back to mean if available, else 0.0
            default_val = bundle.numeric_means.get(feature, 0.0)
            values.append(float(default_val))

    # Shape (1, n_features)
    X_input = np.array(values, dtype=float).reshape(1, -1)

    # Predict probabilities
    probs = bundle.model.predict_proba(X_input)[0]
    pred_idx = int(np.argmax(probs))
    predicted_label = bundle.classes[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "predicted_class": pred_idx,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probs.tolist(),
    }
