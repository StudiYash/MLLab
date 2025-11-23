"""
Handwritten Digit Detection project utilities.

This module provides:
- functions to load and preprocess the digit dataset from a CSV file,
- a function to train a simple neural network model,
- a function to run inference on a single 28x28 image.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, IO
from PIL import Image

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import pickle
import os
from utils.config import MODELS_DIR

@dataclass
class DigitModelBundle:
    """
    Container for the trained handwritten digit model and metadata.
    """
    model: Sequential
    input_shape: Tuple[int, int, int]
    num_classes: int
    class_labels: np.ndarray  # e.g. array([0,1,2,...,9])
    X_val: np.ndarray         # validation images
    y_val: np.ndarray         # validation labels (one-hot)
    val_accuracy: float
    metrics: Dict[str, float]  # e.g. train/val accuracy & loss
    n_train: int
    n_val: int

def get_digits_csv_path() -> Path:
    """
    Returns the path to HandwrittenDigitDetection.csv inside data/raw/handwritten_digit_detection.
    """
    # app/projects/handwritten_digits.py -> go two levels up to repo root
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "raw" / "handwritten_digit_detection" / "HandwrittenDigitDetection.csv"

def load_and_preprocess_digits(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int], int, np.ndarray]:
    """
    Loads the dataset, preprocesses it, and splits it into train/val sets.
    """
    csv_path = get_digits_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)
    
    # First column is label, rest are pixels
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    # Coerce to numeric and fill NaNs
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0).values

    # Normalize
    X = X / 255.0

    # Reshape to (num_samples, 28, 28, 1)
    # The dataset has 784 pixels (28*28)
    X = X.reshape(-1, 28, 28, 1)

    # One-hot encode labels
    num_classes = 10
    y_cat = to_categorical(y, num_classes=num_classes)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

    input_shape = (28, 28, 1)
    class_labels = np.arange(num_classes)

    return X_train, X_val, y_train, y_val, input_shape, num_classes, class_labels

def build_digit_model(input_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    """
    Builds and compiles the Keras Sequential model.
    """
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def train_digit_model(
    epochs: int = 10,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DigitModelBundle:
    """
    Trains the handwritten digit detection model.
    """
    # Try loading first
    try:
        return _load_digit_model()
    except (FileNotFoundError, OSError, ImportError):
        # Fallback to training
        pass

    X_train, X_val, y_train, y_val, input_shape, num_classes, class_labels = load_and_preprocess_digits(
        test_size=test_size, 
        random_state=random_state
    )

    model = build_digit_model(input_shape, num_classes)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0  # Keep it quiet for the module
    )

    # Get validation accuracy
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    history_dict = history.history if hasattr(history, "history") else {}

    def _last_or_default(key: str, default: float = 0.0) -> float:
        values = history_dict.get(key, [])
        return float(values[-1]) if values else float(default)

    metrics = {
        "train_accuracy": _last_or_default("accuracy"),
        "val_accuracy": _last_or_default("val_accuracy"),
        "train_loss": _last_or_default("loss"),
        "val_loss": _last_or_default("val_loss"),
    }

    n_train = int(X_train.shape[0])
    n_val = int(X_val.shape[0])

    bundle = DigitModelBundle(
        model=model,
        input_shape=input_shape,
        num_classes=num_classes,
        class_labels=class_labels,
        X_val=X_val,
        y_val=y_val,
        val_accuracy=val_accuracy,
        metrics=metrics,
        n_train=n_train,
        n_val=n_val,
    )
    
    # Save for next time
    try:
        _save_digit_model(bundle)
    except Exception as e:
        print(f"Warning: Could not save digit model: {e}")
        
    return bundle

def _get_digits_model_paths() -> Tuple[Path, Path]:
    """
    Returns (model_file_path, metadata_file_path).
    """
    model_dir = MODELS_DIR / "handwritten_digits"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "digits_mlp.h5", model_dir / "digits_metadata.pkl"

def _save_digit_model(bundle: DigitModelBundle):
    """
    Saves the Keras model and metadata to disk.
    """
    model_path, meta_path = _get_digits_model_paths()
    
    # Save Keras model
    bundle.model.save(model_path)
    
    # Save metadata
    metadata = {
        "input_shape": bundle.input_shape,
        "num_classes": bundle.num_classes,
        "class_labels": bundle.class_labels,
        "X_val": bundle.X_val,
        "y_val": bundle.y_val,
        "val_accuracy": bundle.val_accuracy,
        "metrics": bundle.metrics,
        "n_train": bundle.n_train,
        "n_val": bundle.n_val,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

def _load_digit_model() -> DigitModelBundle:
    """
    Loads the model and metadata from disk.
    Raises FileNotFoundError if files are missing.
    """
    model_path, meta_path = _get_digits_model_paths()
    
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Model files not found")
        
    # Load Keras model
    model = load_model(model_path)
    
    # Load metadata
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
        
    return DigitModelBundle(
        model=model,
        input_shape=metadata["input_shape"],
        num_classes=metadata["num_classes"],
        class_labels=metadata["class_labels"],
        X_val=metadata["X_val"],
        y_val=metadata["y_val"],
        val_accuracy=metadata["val_accuracy"],
        metrics=metadata["metrics"],
        n_train=metadata["n_train"],
        n_val=metadata["n_val"],
    )

def predict_single_digit(bundle: DigitModelBundle, image_array: np.ndarray) -> Dict[str, object]:
    """
    Runs inference on a single image array.
    """
    # Ensure float32
    img = image_array.astype(np.float32)

    # Normalize if needed
    if img.max() > 1.0:
        img = img / 255.0

    # Reshape to (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)

    # Predict
    probs = bundle.model.predict(img, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    pred_label = int(bundle.class_labels[pred_class])
    confidence = float(probs[pred_class])

    return {
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "confidence": confidence,
        "probabilities": probs.tolist(),
    }

def preprocess_uploaded_digit_image(file_obj: IO[bytes]) -> np.ndarray:
    """
    Takes an uploaded image file (Streamlit's UploadedFile or any file-like object),
    converts it to a 28x28 grayscale image, normalizes it to [0, 1],
    applies a simple inversion heuristic if needed, and returns
    a flattened vector of shape (784,) suitable for the model.
    """
    # Load image as grayscale
    img = Image.open(file_obj).convert("L")  # "L" = 8-bit pixels, black and white

    # Resize to 28x28 (same as training)
    img = img.resize((28, 28))

    # Convert to numpy and normalize
    arr = np.array(img).astype("float32") / 255.0

    # Simple heuristic: if the image is mostly light (white background, dark digit),
    # keep as-is; if mostly dark, invert.
    if arr.mean() > 0.5:
        # Mostly white -> assume white background, dark digit
        # (this is typical MNIST-style; we might not need inversion here)
        pass
    else:
        # Mostly dark -> invert so background is light and digit is dark
        arr = 1.0 - arr

    # Flatten to a vector of 784 elements
    arr_flat = arr.reshape(-1)

    return arr_flat
