import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import backends
try:
    from app.projects.breast_cancer import train_breast_cancer_model, predict_from_partial_features
    from app.projects.handwritten_digits import train_digit_model, predict_single_digit
    from app.projects.house_price import train_house_price_models, predict_house_price
    from app.projects.iris_flowers import train_iris_model, predict_from_features as iris_predict
    from app.projects.diabetes import train_diabetes_models, predict_diabetes
    from app.projects.titanic_survival import train_titanic_model, predict_survival
    from app.projects.sentiment_analysis import load_sentiment_model, predict_sentiment
    from app.projects.spam_detection import train_spam_model, predict_spam
    from app.projects.stock_price import train_stock_movement_model, load_stock_dataset, predict_next_day_movement
    from app.projects.movie_recommender import build_movie_recommender, safe_recommend
except ImportError as e:
    print(f"[CRITICAL] Import failed: {e}")
    sys.exit(1)

def test_breast_cancer():
    try:
        print("Testing Breast Cancer...", end=" ")
        bundle = train_breast_cancer_model()
        # Dummy input
        input_data = {k: 10.0 for k in bundle.feature_names}
        predict_from_partial_features(bundle, input_data)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

def test_digits():
    try:
        print("Testing Handwritten Digits...", end=" ")
        # Train/load with few epochs
        bundle = train_digit_model(epochs=1)
        # Dummy image
        dummy_img = np.zeros((28, 28))
        predict_single_digit(bundle, dummy_img)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

def test_house_price():
    try:
        print("Testing House Price...", end=" ")
        bundle = train_house_price_models()
        # Dummy input
        input_data = {k: 0 for k in bundle.feature_names}
        predict_house_price(bundle, input_data)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

def test_iris():
    try:
        print("Testing Iris...", end=" ")
        bundle = train_iris_model()
        input_data = {k: 1.0 for k in bundle.feature_names}
        iris_predict(bundle, input_data)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

def test_diabetes():
    try:
        print("Testing Diabetes...", end=" ")
        bundle = train_diabetes_models()
        input_data = {k: 1.0 for k in bundle.feature_names}
        predict_diabetes(bundle, input_data)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

def test_titanic():
    try:
        print("Testing Titanic...", end=" ")
        bundle = train_titanic_model()
        input_data = {k: 0 for k in bundle.feature_names}
        predict_survival(bundle, input_data)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

def test_sentiment():
    try:
        print("Testing Sentiment...", end=" ")
        # This might fail if model not found, but we want to test the try/except logic
        try:
            bundle = load_sentiment_model()
            predict_sentiment(bundle, "This is a test.")
            print("OK")
        except FileNotFoundError:
            print("SKIPPED (Model missing)")
    except Exception as e:
        print(f"FAIL: {e}")

def test_spam():
    try:
        print("Testing Spam...", end=" ")
        bundle = train_spam_model(epochs=1)
        predict_spam(bundle, "Test email")
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

def test_stock():
    try:
        print("Testing Stock...", end=" ")
        bundle = train_stock_movement_model()
        df = load_stock_dataset()
        predict_next_day_movement(bundle, df)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

def test_movies():
    try:
        print("Testing Movies...", end=" ")
        bundle = build_movie_recommender()
        safe_recommend(bundle, "Star Wars (1977)")
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    print("=== MLLab Smoke Test ===")
    test_breast_cancer()
    test_digits()
    test_house_price()
    test_iris()
    test_diabetes()
    test_titanic()
    test_sentiment()
    test_spam()
    test_stock()
    test_movies()
    print("=== Done ===")
