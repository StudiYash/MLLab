# MLLab – Interactive Machine Learning Lab

## 🚀 Overview

MLLab is a unified home for multiple small ML/DL projects (regression, classification, NLP, recommender systems, time-series/stock movement, etc.).

Originally, each project was a standalone notebook; now they are being refactored into a single, organized, Streamlit-based lab.

The goal is both: learning (understanding different ML tasks) and portfolio (presenting them as one cohesive app).

## 🧩 Included Mini Projects

- Breast Cancer Prediction (binary classification with CatBoost)
- Handwritten Digit Detection (deep learning on image pixels)
- House Price Prediction (regression with multiple models)
- Iris Flower Classification (EDA + classical ML)
- Movie Recommendation System (correlation-based recommender)
- Predicting Diabetes (classification with multiple models)
- Sentiment Analysis on Movie Reviews (LSTM-based NLP)
- Spam Email Detection (LSTM-based NLP)
- Stock Price Prediction (up/down movement classification)
- Titanic Survival Prediction (Kaggle-style pipeline)

## 🗂️ Repository Structure

```text
MLLab/
  app/
    main.py          # Streamlit entry point
    projects/        # Per-project logic & UI modules
  notebooks/         # Original Jupyter notebooks for each mini-project
  models/            # Saved trained models (pkl, h5, etc.)
  data/
    raw/             # Original datasets (CSV, Excel, etc.)
    processed/       # Cleaned / feature-engineered datasets
  utils/
    common.py        # Shared helpers (to be implemented)
  requirements.txt
  README.md
```

## 🛠 Tech Stack

- Python 3.x
- Streamlit
- scikit-learn
- TensorFlow / Keras (for DL projects)
- XGBoost, CatBoost (where used)
- Pandas, NumPy, Matplotlib, Seaborn

## ▶️ How to Run

1. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run app/main.py
   ```

## 📌 Roadmap

- Bring all 10 notebooks and datasets into this repo
- Refactor each project into `train_model() / load_model() / predict()` modules
- Build per-project Streamlit pages with Code + Interaction tabs
- Add model comparison summaries and metrics dashboards
- Polish UI/UX and documentation
