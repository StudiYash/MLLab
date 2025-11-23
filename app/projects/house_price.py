"""
House Price Prediction project utilities.

This module provides:
- functions to load and preprocess the housing dataset,
- a function to train multiple regression models,
- a function to run inference for a single house.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


@dataclass
class HousePriceModelBundle:
    """
    Container for the trained house price models and metadata.
    """
    models: Dict[str, Any]                 # e.g. {"svr": svr_model, "rf": rf_model, "lr": lr_model}
    feature_names: List[str]               # final feature names after encoding
    metrics: Dict[str, Dict[str, float]]   # e.g. {"svr": {"mae": ..., "mape": ...}, ...}
    encoder: OneHotEncoder                 # fitted encoder for categorical vars
    numeric_columns: List[str]
    categorical_columns: List[str]
    numeric_means: Dict[str, float]        # mean of numeric columns (for inference defaults)


def get_house_price_excel_path() -> Path:
    """
    Returns the path to HousePricePrediction.xlsx inside data/raw/house_price_prediction.
    """
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "raw" / "house_price_prediction" / "HousePricePrediction.xlsx"


def load_house_price_dataset() -> pd.DataFrame:
    """
    Loads the house price dataset from the Excel file and returns a DataFrame.
    """
    path = get_house_price_excel_path()
    if not path.exists():
        raise FileNotFoundError(f"House price dataset not found at {path}")
    df = pd.read_excel(path)
    return df


def preprocess_house_price_data(
    df: pd.DataFrame,
    target_column: str = "SalePrice",
) -> Tuple[pd.DataFrame, pd.Series, OneHotEncoder, List[str], List[str]]:
    """
    Cleans and encodes the dataset.

    Steps:
    - Drop Id column if present.
    - If target has NaNs, fill with mean.
    - Drop remaining rows with missing values.
    - Identify categorical (object) vs numeric columns.
    - One-hot encode categorical columns via OneHotEncoder(handle_unknown='ignore').
    - Return:
      - X (encoded features as DataFrame)
      - y (target)
      - fitted encoder
      - numeric_columns
      - categorical_columns
    """
    df_clean = df.copy()

    # Drop Id if present
    if "Id" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Id"])

    # Fill target NaNs with mean
    if df_clean[target_column].isnull().any():
        df_clean[target_column] = df_clean[target_column].fillna(df_clean[target_column].mean())

    # Drop remaining rows with missing values
    df_clean = df_clean.dropna()

    # Identify columns
    categorical_columns = df_clean.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = (
        df_clean.select_dtypes(include=["number"])
        .drop(columns=[target_column])
        .columns
        .tolist()
    )

    # Separate X and y
    X_raw = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    # Encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    if categorical_columns:
        X_cat = encoder.fit_transform(X_raw[categorical_columns])
        cat_feature_names = encoder.get_feature_names_out(categorical_columns).tolist()
        X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names, index=X_raw.index)
    else:
        # No categorical columns – keep an empty DF for consistency
        X_cat_df = pd.DataFrame(index=X_raw.index)

    # Numeric part
    X_num_df = X_raw[numeric_columns]

    # Concatenate numeric + categorical encodings
    X = pd.concat([X_num_df, X_cat_df], axis=1)

    return X, y, encoder, numeric_columns, categorical_columns


def train_house_price_models(
    test_size: float = 0.2,
    random_state: int = 0,
) -> HousePriceModelBundle:
    """
    Trains SVR, RandomForestRegressor, and LinearRegression on the house price dataset.
    Returns a HousePriceModelBundle with models and metrics.
    """
    df = load_house_price_dataset()
    X, y, encoder, numeric_cols, cat_cols = preprocess_house_price_data(df)

    # Compute numeric means for inference defaults
    numeric_means = X[numeric_cols].mean().to_dict()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize models
    svr = SVR()
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    lr = LinearRegression()

    models_map = {
        "svr": svr,
        "rf": rf,
        "lr": lr,
    }

    metrics: Dict[str, Dict[str, float]] = {}

    for name, model in models_map.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        mae = mean_absolute_error(y_valid, y_pred)
        mape = mean_absolute_percentage_error(y_valid, y_pred)

        metrics[name] = {"mae": mae, "mape": mape}

    return HousePriceModelBundle(
        models=models_map,
        feature_names=X.columns.tolist(),
        metrics=metrics,
        encoder=encoder,
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        numeric_means=numeric_means,
    )


def predict_house_price(
    bundle: HousePriceModelBundle,
    input_data: Dict[str, Any],
) -> Dict[str, float]:
    """
    Runs inference for a single house feature dictionary using all models.

    - For numeric columns:
        uses the value from input_data if provided,
        otherwise falls back to the mean from training data.
    - For categorical columns:
        uses the value from input_data if provided,
        otherwise uses the string "Unknown" (handled by handle_unknown='ignore').

    Returns a dict of model_name -> predicted price.
    """
    input_row: Dict[str, Any] = {}

    # Numeric: use provided or mean from training
    for col in bundle.numeric_columns:
        if col in input_data:
            input_row[col] = input_data[col]
        else:
            input_row[col] = bundle.numeric_means.get(col, 0.0)

    # Categorical: use provided or a generic "Unknown"
    for col in bundle.categorical_columns:
        input_row[col] = input_data.get(col, "Unknown")

    # Create DataFrame for a single row
    input_df_raw = pd.DataFrame([input_row])

    # Encode categorical part
    if bundle.categorical_columns:
        X_cat = bundle.encoder.transform(input_df_raw[bundle.categorical_columns])
        cat_feature_names = bundle.encoder.get_feature_names_out(
            bundle.categorical_columns
        ).tolist()
        X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names)
    else:
        X_cat_df = pd.DataFrame()

    # Numeric part
    X_num_df = input_df_raw[bundle.numeric_columns]

    # Concatenate numeric + encoded categorical
    X_input = pd.concat([X_num_df, X_cat_df], axis=1)

    # Ensure columns match the training feature order
    X_input = X_input.reindex(columns=bundle.feature_names, fill_value=0)

    results: Dict[str, float] = {}
    for name, model in bundle.models.items():
        pred = model.predict(X_input)[0]
        results[name] = float(pred)

    return results
