"""
Sentiment Analysis on Movie Reviews project utilities.

This module provides:
- functions to load a pre-trained LSTM model and tokenizer,
- a function to preprocess text (tokenization + padding),
- functions to run inference on single or multiple texts.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import numpy as np

# TensorFlow imports
# Note: These might be slow, so we import them at module level but they are used inside functions usually.
# However, for type hinting 'Any' is used for model/tokenizer to avoid strict dependency on import time if we wanted lazy loading,
# but here we just import them.
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model

@dataclass
class SentimentModelBundle:
    """
    Container for the trained Sentiment model and metadata.
    """
    model: Any                # Keras model
    tokenizer: Any            # Keras tokenizer
    max_len: int              # max sequence length used for padding
    label_map: Dict[int, str] # {0: "negative", 1: "positive"}

from utils.config import MODELS_DIR

def get_sentiment_model_dir() -> Path:
    """
    Returns the path to models/sentiment_analysis_movie_reviews.
    """
    return MODELS_DIR / "sentiment_analysis_movie_reviews"

def get_sentiment_model_path() -> Path:
    return get_sentiment_model_dir() / "imdb_lstm_sentiment.h5"

def get_sentiment_tokenizer_path() -> Path:
    return get_sentiment_model_dir() / "imdb_tokenizer.json"

def load_sentiment_model(max_len: int = 200) -> SentimentModelBundle:
    """
    Loads the pre-trained IMDB LSTM model and tokenizer from disk and
    returns a SentimentModelBundle.

    - max_len: sequence length used for padding/truncating inputs.
    """
    model_path = get_sentiment_model_path()
    tokenizer_path = get_sentiment_tokenizer_path()
    
    if not model_path.exists():
        raise FileNotFoundError(f"Sentiment model not found at {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Sentiment tokenizer not found at {tokenizer_path}")
        
    # Load model
    model = load_model(model_path)
    
    # Load tokenizer
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        json_string = f.read()
    tokenizer = tokenizer_from_json(json_string)
    
    label_map = {0: "negative", 1: "positive"}
    
    return SentimentModelBundle(
        model=model,
        tokenizer=tokenizer,
        max_len=max_len,
        label_map=label_map
    )

def preprocess_texts(bundle: SentimentModelBundle, texts: List[str]) -> np.ndarray:
    """
    Turns a list of raw text strings into padded integer sequences ready for the model.
    """
    if not texts:
        return np.empty((0, bundle.max_len))
        
    # Convert to sequences
    sequences = bundle.tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded = pad_sequences(sequences, maxlen=bundle.max_len, padding="post", truncating="post")
    
    return padded

def predict_sentiments(
    bundle: SentimentModelBundle,
    texts: List[str]
) -> List[Dict[str, Any]]:
    """
    Predicts sentiment for a list of texts.

    Returns a list of dicts, one per input text:
        {
            "text": original text,
            "predicted_label": "positive" or "negative",
            "predicted_class": 0 or 1,
            "confidence": float in [0, 1]
        }
    """
    if not texts:
        return []
        
    padded = preprocess_texts(bundle, texts)
    
    # Predict
    # Output shape (N, 1) or (N,)
    predictions = bundle.model.predict(padded, verbose=0)
    
    results = []
    for i, text in enumerate(texts):
        prob = float(predictions[i]) if predictions.ndim == 1 else float(predictions[i][0])
        
        # Threshold at 0.5
        pred_class = 1 if prob >= 0.5 else 0
        label = bundle.label_map[pred_class]
        
        # Confidence: if class 1, it's prob. If class 0, it's 1 - prob.
        confidence = prob if pred_class == 1 else 1.0 - prob
        
        results.append({
            "text": text,
            "predicted_label": label,
            "predicted_class": pred_class,
            "confidence": confidence
        })
        
    return results

def predict_sentiment(
    bundle: SentimentModelBundle,
    text: str
) -> Dict[str, Any]:
    """
    Convenience wrapper over predict_sentiments for a single text.
    """
    results = predict_sentiments(bundle, [text])
    if results:
        return results[0]
    else:
        # Should not happen if text is provided
        return {
            "text": text,
            "predicted_label": "unknown",
            "predicted_class": -1,
            "confidence": 0.0
        }

if __name__ == "__main__":
    print("Loading sentiment model...")
    try:
        bundle = load_sentiment_model(max_len=200)
        print("Model loaded.")
        
        sample_texts = [
            "This movie was absolutely fantastic. I loved every minute of it!",
            "It was a terrible film. I will never watch it again."
        ]
        
        print("Running predictions...")
        results = predict_sentiments(bundle, sample_texts)
        
        for r in results:
            print(f"Text: {r['text']}")
            print(f"  -> Predicted: {r['predicted_label']} "
                  f"(class={r['predicted_class']}, confidence={r['confidence']:.4f})")
                  
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure the model files are present in models/sentiment_analysis_movie_reviews/")
    except Exception as e:
        print(f"An error occurred: {e}")
