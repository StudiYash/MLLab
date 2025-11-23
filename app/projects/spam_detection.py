"""
Spam Email Detection project utilities.

This module provides:
- functions to load and preprocess the spam dataset,
- a function to train an LSTM-based spam classifier,
- a function to run inference for single or multiple emails.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

# TensorFlow imports
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import load_model
import pickle
import os
from utils.config import MODELS_DIR

@dataclass
class SpamModelBundle:
    """
    Container for the trained Spam Detection model and metadata.
    """
    model: Any                # Keras model
    tokenizer: Any            # Keras Tokenizer
    max_len: int              # max sequence length used for padding
    label_map: Dict[int, str] # {0: "ham", 1: "spam"}
    vocab_size: int           # size of vocabulary
    history: Optional[Dict[str, List[float]]] = None # training history

def get_spam_csv_path() -> Path:
    """
    Returns the path to spam_ham_dataset.csv inside data/raw/spam_email_detection.
    """
    root = Path(__file__).resolve().parents[2]  # repo root
    return root / "data" / "raw" / "spam_email_detection" / "spam_ham_dataset.csv"

def load_spam_dataset() -> Tuple[List[str], np.ndarray, Dict[int, str]]:
    """
    Loads the spam dataset and returns texts, labels, and label map.
    
    Returns:
        texts: List of email strings
        labels: numpy array of 0s and 1s
        label_map: {0: "ham", 1: "spam"} (or similar)
    """
    path = get_spam_csv_path()
    if not path.exists():
        raise FileNotFoundError(f"Spam dataset not found at {path}")
        
    df = pd.read_csv(path)
    
    # 1. Detect text column
    text_candidates = ["text", "message", "EmailText", "email", "body"]
    text_col = None
    for col in text_candidates:
        if col in df.columns:
            text_col = col
            break
            
    if not text_col:
        raise ValueError(f"Could not find text column. Available columns: {df.columns.tolist()}")
        
    # 2. Detect label column
    label_candidates = ["label", "Category", "class", "spam", "target", "label_num"]
    label_col = None
    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break
            
    if not label_col:
        raise ValueError(f"Could not find label column. Available columns: {df.columns.tolist()}")
        
    # 3. Normalize labels
    # Check if numeric
    if pd.api.types.is_numeric_dtype(df[label_col]):
        # Assume 0 is ham, 1 (or >0) is spam
        labels = (df[label_col] >= 1).astype(int).values
        label_map = {0: "ham", 1: "spam"}
    else:
        # String labels
        # Normalize
        s_labels = df[label_col].astype(str).str.lower().str.strip()
        
        # Map common patterns
        # "spam" -> 1, everything else -> 0
        labels = (s_labels == "spam").astype(int).values
        label_map = {0: "ham", 1: "spam"}
        
    texts = df[text_col].astype(str).tolist()
    
    return texts, labels, label_map

def train_spam_model(
    max_words: int = 10000,
    max_len: int = 150,
    embedding_dim: int = 64,
    batch_size: int = 32,
    epochs: int = 3,
    validation_split: float = 0.2,
) -> SpamModelBundle:
    """
    Trains an LSTM model on the spam dataset.
    """
    # Try loading first
    try:
        return _load_spam_model()
    except (FileNotFoundError, OSError, ImportError):
        pass

    texts, labels, label_map = load_spam_dataset()
    
    # Tokenization
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    y = labels
    
    # Model
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    # Train
    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=0
    )
    
    bundle = SpamModelBundle(
        model=model,
        tokenizer=tokenizer,
        max_len=max_len,
        label_map=label_map,
        vocab_size=max_words,
        history=history.history
    )
    
    # Save
    try:
        _save_spam_model(bundle)
    except Exception as e:
        print(f"Warning: Could not save spam model: {e}")
        
    return bundle

def _get_spam_model_paths() -> Tuple[Path, Path]:
    model_dir = MODELS_DIR / "spam_detection"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "spam_lstm.h5", model_dir / "spam_metadata.pkl"

def _save_spam_model(bundle: SpamModelBundle):
    model_path, meta_path = _get_spam_model_paths()
    bundle.model.save(model_path)
    
    metadata = {
        "tokenizer": bundle.tokenizer,
        "max_len": bundle.max_len,
        "label_map": bundle.label_map,
        "vocab_size": bundle.vocab_size,
        "history": bundle.history
    }
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

def _load_spam_model() -> SpamModelBundle:
    model_path, meta_path = _get_spam_model_paths()
    
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Spam model files not found")
        
    model = load_model(model_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
        
    return SpamModelBundle(
        model=model,
        tokenizer=metadata["tokenizer"],
        max_len=metadata["max_len"],
        label_map=metadata["label_map"],
        vocab_size=metadata["vocab_size"],
        history=metadata.get("history")
    )

def preprocess_emails(bundle: SpamModelBundle, emails: List[str]) -> np.ndarray:
    """
    Turns a list of raw email strings into padded integer sequences.
    """
    if not emails:
        return np.empty((0, bundle.max_len))
        
    sequences = bundle.tokenizer.texts_to_sequences(emails)
    padded = pad_sequences(sequences, maxlen=bundle.max_len, padding="post", truncating="post")
    return padded

def predict_spam_batch(
    bundle: SpamModelBundle,
    emails: List[str],
) -> List[Dict[str, Any]]:
    """
    Predicts spam/ham for a list of emails.
    """
    if not emails:
        return []
        
    padded = preprocess_emails(bundle, emails)
    
    # Predict
    predictions = bundle.model.predict(padded, verbose=0)
    
    results = []
    for i, email in enumerate(emails):
        prob = float(predictions[i]) if predictions.ndim == 1 else float(predictions[i][0])
        
        pred_class = 1 if prob >= 0.5 else 0
        label = bundle.label_map.get(pred_class, "unknown")
        
        confidence = prob if pred_class == 1 else 1.0 - prob
        
        results.append({
            "email": email,
            "predicted_class": pred_class,
            "predicted_label": label,
            "probability_spam": prob,
            "confidence": confidence
        })
        
    return results

def predict_spam(
    bundle: SpamModelBundle,
    email: str,
) -> Dict[str, Any]:
    """
    Convenience wrapper for a single email.
    """
    results = predict_spam_batch(bundle, [email])
    if results:
        return results[0]
    else:
        return {
            "email": email,
            "predicted_class": -1,
            "predicted_label": "error",
            "probability_spam": 0.0,
            "confidence": 0.0
        }

if __name__ == "__main__":
    print("Training spam detection model (this may take a bit)...")
    # 1 epoch for quick sanity check in __main__, but default is 3
    bundle = train_spam_model(epochs=1)
    
    test_emails = [
        "Congratulations! You've won a $1000 gift card. Click here to claim your prize now!!!",
        "Hi, just checking in about our meeting tomorrow at 10 AM.",
    ]
    
    results = predict_spam_batch(bundle, test_emails)
    for r in results:
        print("-" * 60)
        print("Email:", r["email"])
        print(f"Predicted: {r['predicted_label']} "
              f"(class={r['predicted_class']}, "
              f"prob_spam={r['probability_spam']:.4f}, "
              f"confidence={r['confidence']:.4f})")
