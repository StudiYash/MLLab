"""
Stock Price Prediction project utilities.

This module provides:
- functions to load and preprocess the Tesla stock dataset,
- a function to train a RandomForestClassifier for next-day movement prediction,
- a function to run inference for the next day's movement.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

@dataclass
class StockPriceModelBundle:
    """
    Container for the trained Stock Price model and metadata.
    """
    model: RandomForestClassifier
    scaler: StandardScaler
    feature_names: List[str]
    metrics: Dict[str, float]
    label_map: Dict[int, str]  # {0: "down", 1: "up"}

def get_tesla_csv_path() -> Path:
    """
    Returns the path to Tesla.csv inside data/raw/stock_price_prediction.
    """
    root = Path(__file__).resolve().parents[2]  # repo root
    return root / "data" / "raw" / "stock_price_prediction" / "Tesla.csv"

def load_stock_dataset() -> pd.DataFrame:
    """
    Loads the Tesla dataset and returns a pandas DataFrame.
    """
    path = get_tesla_csv_path()
    if not path.exists():
        raise FileNotFoundError(f"Stock dataset not found at {path}")
    return pd.read_csv(path)

def preprocess_stock_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses the stock data for training.
    
    Steps:
    1. Parse date and sort.
    2. Ensure 'Close' column exists.
    3. Create target: next day movement (1 if Close[t+1] > Close[t], else 0).
    4. Engineer rolling features.
    5. Drop NaNs.
    
    Returns:
        X: DataFrame of features
        y: Series of labels (0 or 1)
    """
    df_clean = df.copy()
    
    # 1. Date handling
    date_col = None
    for col in ["Date", "date", "DATE"]:
        if col in df_clean.columns:
            date_col = col
            break
            
    if date_col:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        df_clean = df_clean.sort_values(by=date_col)
        
    # 2. Ensure Close
    if "Close" not in df_clean.columns:
        if "Adj Close" in df_clean.columns:
            df_clean["Close"] = df_clean["Adj Close"]
        else:
            raise ValueError("Dataset missing 'Close' column")
            
    # 3. Create target (Next Day Movement)
    # 1 = UP, 0 = DOWN/FLAT
    df_clean["Target"] = (df_clean["Close"].shift(-1) > df_clean["Close"]).astype(int)
    
    # The last row has no target (shift(-1) is NaN comparison -> False -> 0, but logically invalid)
    # We will drop the last row later after feature engineering to be safe, 
    # but strictly speaking shift(-1) > Close is False for NaN > val. 
    # Let's explicitly mark the last row as invalid or just drop it at the end.
    # Actually, shift(-1) creates a NaN in a temporary column if we did that.
    # Let's just remember to drop the last row of the dataset eventually.
    
    # 4. Feature Engineering
    # Base features
    base_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df_clean.columns]
    
    # Rolling features
    # Return
    df_clean["return_1d"] = df_clean["Close"].pct_change()
    # Moving Averages
    df_clean["ma_5"] = df_clean["Close"].rolling(window=5).mean()
    df_clean["ma_10"] = df_clean["Close"].rolling(window=10).mean()
    # Volatility
    df_clean["volatility_5"] = df_clean["Close"].rolling(window=5).std()
    
    # Select features
    feature_cols = base_cols + ["return_1d", "ma_5", "ma_10", "volatility_5"]
    
    # 5. Drop NaNs
    # This drops the initial rows (due to rolling) AND the last row (if we consider target validity, 
    # though our target calc above didn't produce NaNs, it produced 0s for the last row).
    # Wait, (NaN > val) is False. So the last row has Target=0. But we shouldn't train on it.
    # Let's explicitly drop the last row since we don't know the future.
    df_clean = df_clean.iloc[:-1] 
    
    # Now drop NaNs from rolling windows
    df_clean = df_clean.dropna(subset=feature_cols)
    
    if df_clean.empty:
        raise ValueError("Dataset is empty after preprocessing (too short for rolling windows?)")
        
    X = df_clean[feature_cols]
    y = df_clean["Target"]
    
    return X, y

def train_stock_movement_model(
    test_size: float = 0.2,
    random_state: int = 42,
) -> StockPriceModelBundle:
    """
    Trains a RandomForestClassifier to predict next-day stock movement.
    Uses a time-based split.
    """
    df = load_stock_dataset()
    X, y = preprocess_stock_data(df)
    
    # Time-based split
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0)
    }
    
    return StockPriceModelBundle(
        model=model,
        scaler=scaler,
        feature_names=X.columns.tolist(),
        metrics=metrics,
        label_map={0: "down", 1: "up"}
    )

def prepare_last_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the last valid feature row from the dataframe for inference.
    """
    df_clean = df.copy()
    
    # Date handling (same as preprocess)
    date_col = None
    for col in ["Date", "date", "DATE"]:
        if col in df_clean.columns:
            date_col = col
            break
    if date_col:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        df_clean = df_clean.sort_values(by=date_col)
        
    if "Close" not in df_clean.columns:
        if "Adj Close" in df_clean.columns:
            df_clean["Close"] = df_clean["Adj Close"]
        else:
            raise ValueError("Dataset missing 'Close' column")
            
    # Feature Engineering (same as preprocess)
    base_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df_clean.columns]
    
    df_clean["return_1d"] = df_clean["Close"].pct_change()
    df_clean["ma_5"] = df_clean["Close"].rolling(window=5).mean()
    df_clean["ma_10"] = df_clean["Close"].rolling(window=10).mean()
    df_clean["volatility_5"] = df_clean["Close"].rolling(window=5).std()
    
    feature_cols = base_cols + ["return_1d", "ma_5", "ma_10", "volatility_5"]
    
    # We want the very last row, assuming it has valid data
    # If the dataset is too short, this might be NaN
    last_row = df_clean.iloc[[-1]][feature_cols]
    
    if last_row.isnull().values.any():
        # Try to find the last non-NaN row? 
        # Or just raise error if the latest data isn't sufficient for rolling features.
        # For a prediction app, we usually want the LATEST available data.
        # If today's data is incomplete, we can't predict tomorrow based on today.
        raise ValueError("Last row contains NaNs (insufficient history for rolling features?)")
        
    return last_row

def predict_next_day_movement(
    bundle: StockPriceModelBundle,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Predicts whether the stock will go UP or DOWN on the next day,
    based on the latest data in df.
    """
    X_last = prepare_last_window_features(df)
    
    # Scale
    X_last_scaled = bundle.scaler.transform(X_last)
    
    # Predict
    pred_class = int(bundle.model.predict(X_last_scaled)[0])
    probs = bundle.model.predict_proba(X_last_scaled)[0] # [prob_0, prob_1]
    prob_up = float(probs[1])
    
    label = bundle.label_map[pred_class]
    confidence = prob_up if pred_class == 1 else 1.0 - prob_up
    
    return {
        "predicted_class": pred_class,
        "predicted_label": label,
        "probability_up": prob_up,
        "confidence": confidence
    }

if __name__ == "__main__":
    print("Training stock movement model on Tesla.csv ...")
    bundle = train_stock_movement_model()
    
    print("Validation metrics:", bundle.metrics)
    
    # Load raw df again for a "next day" type prediction
    df = load_stock_dataset()
    result = predict_next_day_movement(bundle, df)
    print("Next-day movement prediction:", result)
