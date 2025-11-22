# app/main.py

# --- Bootstrap so "app" package imports work with `streamlit run app/main.py`
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]  # repo root (folder that contains "app")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ==== Import project backends ====

# ==== Import project backends ====

from app.projects.breast_cancer import (
    train_breast_cancer_model,
    predict_from_partial_features,
)

from app.projects.handwritten_digits import (
    train_digit_model,
    DigitModelBundle,
    predict_single_digit,
    preprocess_uploaded_digit_image,
)

from app.projects.house_price import (
    train_house_price_models,
    predict_house_price,
)

from app.projects.iris_flowers import (
    train_iris_model,
    predict_from_features as iris_predict_from_features,
)

from app.projects.diabetes import (
    train_diabetes_models,
    predict_diabetes,
)

from app.projects.titanic_survival import (
    train_titanic_model,
    predict_survival,
)

from app.projects.sentiment_analysis import (
    load_sentiment_model,
    predict_sentiment,
)

from app.projects.spam_detection import (
    train_spam_model,
    predict_spam,
)

from app.projects.stock_price import (
    train_stock_movement_model,
    load_stock_dataset,
    predict_next_day_movement,
)

from app.projects.movie_recommender import (
    build_movie_recommender,
    safe_recommend,
)

from utils.common import (
    get_repo_root,
    show_notebook_viewer,
    open_notebook_file,
)

# =======================================================================================
#                                  CACHED LOADERS
# =======================================================================================

@st.cache_resource
def get_breast_cancer_bundle():
    return train_breast_cancer_model()


@st.cache_resource
def get_digit_bundle() -> DigitModelBundle:
    # a small model, but still better to cache
    return train_digit_model(epochs=3)  # keep epochs modest for interactivity


@st.cache_resource
def get_house_price_bundle():
    return train_house_price_models()


@st.cache_resource
def get_iris_bundle():
    return train_iris_model()


@st.cache_resource
def get_diabetes_bundle():
    return train_diabetes_models()


@st.cache_resource
def get_titanic_bundle():
    return train_titanic_model()


@st.cache_resource
def get_sentiment_bundle():
    # IMDB LSTM model from disk
    return load_sentiment_model(max_len=200)


@st.cache_resource
def get_spam_bundle():
    # Train once per session
    return train_spam_model(epochs=1)


@st.cache_resource
def get_stock_bundle():
    return train_stock_movement_model()


@st.cache_resource
def get_stock_dataframe():
    return load_stock_dataset()


@st.cache_resource
def get_movie_recommender_bundle():
    return build_movie_recommender(min_ratings_default=50)


# =======================================================================================
#                                  PAGE IMPLEMENTATIONS
# =======================================================================================

def page_overview():
    # ------------------------------------------
    # Display centered logo at the top
    # ------------------------------------------
    root = get_repo_root()
    logo_path = root / "assets" / "logo.png"

    # Small CSS helper just to center the block
    st.markdown(
        """
        <style>
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if logo_path.exists():
        st.markdown("<div class='logo-container'>", unsafe_allow_html=True)
        # width is in pixels; tweak 900 → 1000 if you want even bigger
        st.image(str(logo_path), width=900)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"Logo not found at: {logo_path}")

    # ------------------------------------------
    # Centered Title (without "🧪 MLLab – ")
    # ------------------------------------------
    st.markdown(
        """
        <h2 style='text-align: center; margin-bottom: 1.5rem;'>
            Interactive Machine Learning Lab
        </h2>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------
    # Description text
    # ------------------------------------------
    st.markdown(
        """
MLLab is a unified **Streamlit lab** that bundles multiple ML/DL mini-projects
under one roof.  
Each project started as a standalone notebook and is now being refactored
into a **reusable backend + interactive UI** pattern.

---

### 📚 Included Mini Projects

1. **Breast Cancer Prediction** – Binary classification (CatBoost, sklearn dataset)  
2. **Handwritten Digit Detection** – 28×28 image classifier (Keras Dense NN)  
3. **House Price Prediction** – Regression with multiple models (SVR, RF, LR)  
4. **Iris Flower Classification** – Classic ML on the Iris dataset  
5. **Predicting Diabetes** – Multiple classifiers (RFC, DT, XGB, SVC)  
6. **Titanic Survival Prediction** – Kaggle-style survival model  
7. **Sentiment Analysis** – Pre-trained IMDB LSTM model  
8. **Spam Email Detection** – LSTM-based spam vs ham classifier  
9. **Stock Price Prediction** – Tesla next-day movement (up/down) classifier  
10. **Movie Recommendation System** – Item–item correlation-based recommender

Use the **sidebar** to jump into any project and play with:
- Minimal inputs (rest are auto-filled from dataset statistics),
- Trained models behind the scenes,
- Quick metrics to understand how each model performs.
"""
    )


def page_breast_cancer():
    st.title("🎗️ Breast Cancer Prediction")
    
    col_left, col_right = st.columns([1.1, 1.4])
    
    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_breast_cancer_bundle()

        st.subheader("Model Performance")

        # Safely extract metrics with defaults in case keys are missing
        raw_metrics = bundle.metrics if isinstance(bundle.metrics, dict) else {}
        accuracy  = float(raw_metrics.get("accuracy", 0.0))
        precision = float(raw_metrics.get("precision", 0.0))
        recall    = float(raw_metrics.get("recall", 0.0))
        f1_score  = float(raw_metrics.get("f1", 0.0))

        # --- KPI cards (top row) ---
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Accuracy", f"{accuracy * 100:.2f}%")
        with mcol2:
            st.metric("Precision", f"{precision * 100:.2f}%")
        with mcol3:
            st.metric("Recall", f"{recall * 100:.2f}%")
        with mcol4:
            st.metric("F1-score", f"{f1_score * 100:.2f}%")

        # --- Detailed table ---
        st.markdown("#### Detailed Scores")
        metrics_df = pd.DataFrame(
            [
                {"Metric": "Accuracy",  "Score": accuracy},
                {"Metric": "Precision", "Score": precision},
                {"Metric": "Recall",    "Score": recall},
                {"Metric": "F1-score",  "Score": f1_score},
            ]
        )
        st.table(metrics_df.style.format({"Score": "{:.4f}"}))

        # --- Bar chart for visual comparison ---
        st.markdown("#### Visual Comparison")
        chart_df = metrics_df.copy()
        chart_df["Score (%)"] = chart_df["Score"] * 100
        chart_df = chart_df.set_index("Metric")
        st.bar_chart(chart_df["Score (%)"])

        # Existing explanation text (keep this below the visuals)
        st.markdown(
            """
        We use the **sklearn breast cancer dataset**.  
        There are many features; here we expose a few important ones and fill the rest
        with **mean values** from the training data.
        """
        )

        col1, col2 = st.columns(2)

        with col1:
            mean_radius = st.number_input("mean radius", min_value=0.0, max_value=50.0, value=14.0)
            mean_texture = st.number_input("mean texture", min_value=0.0, max_value=50.0, value=20.0)

        with col2:
            mean_smoothness = st.number_input(
                "mean smoothness", min_value=0.0, max_value=1.0, value=0.10
            )

        if st.button("Predict Tumor Type"):
            input_data = {
                "mean radius": mean_radius,
                "mean texture": mean_texture,
                "mean smoothness": mean_smoothness,
            }
            result = predict_from_partial_features(bundle, input_data)

            st.success(f"Prediction: **{result['label'].upper()}**")
            st.write(f"Raw class: `{result['raw_prediction']}`")
            st.write(f"Confidence: `{result['probability']:.4f}`")

    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/BreastCancerPrediction.ipynb", height=1100)

        if st.button("Open Jupyter File", key="open_bc_nb"):
            open_notebook_file("notebooks/BreastCancerPrediction.ipynb")


def page_handwritten_digits():
    st.title("✏️ Handwritten Digit Detection")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_digit_bundle()

        # Short description
        st.markdown(
            """
            This model is trained on a CSV version of a handwritten digit dataset
            where each row is a flattened **28×28 grayscale image** (784 pixel values)
            fed into a dense neural network.
            """
        )

        # --- Metrics panel ---
        st.subheader("Validation Metrics & Dataset Size")

        metrics = getattr(bundle, "metrics", {}) or {}
        n_train = getattr(bundle, "n_train", None)
        n_val = getattr(bundle, "n_val", None)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        train_acc = metrics.get("train_accuracy")
        val_acc = metrics.get("val_accuracy")
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")

        with col_m1:
            if train_acc is not None:
                st.metric("Train Accuracy", f"{train_acc * 100:.2f}%")
        with col_m2:
            if val_acc is not None:
                st.metric("Val Accuracy", f"{val_acc * 100:.2f}%")
        with col_m3:
            if train_loss is not None:
                st.metric("Train Loss", f"{train_loss * 100:.4f}%")
        with col_m4:
            if val_loss is not None:
                st.metric("Val Loss", f"{val_loss * 100:.4f}%")

        # Dataset sizes
        lines = []
        if n_train is not None:
            lines.append(f"- Training set size: **{n_train}** images")
        if n_val is not None:
            lines.append(f"- Validation set size: **{n_val}** images")
        if lines:
            st.markdown("\n".join(lines))

        st.markdown("---")

        # --- Random validation image demo ---
        st.subheader("Quick Demo – Random Validation Image")

        if st.button("Run prediction on a random validation image"):
            import numpy as np

            idx = int(np.random.randint(0, bundle.X_val.shape[0]))
            img = bundle.X_val[idx]
            true_label = int(np.argmax(bundle.y_val[idx]))

            result = predict_single_digit(bundle, img)

            st.write(f"Random index: `{idx}`")
            st.write(f"True label: `{true_label}`")
            st.write(
                f"Predicted label: **{result['predicted_label']}** "
                f"(class={result['predicted_class']}, confidence={result['confidence']:.4f})"
            )

        st.markdown("---")

        # --- User-uploaded digit demo ---
        st.subheader("Try Your Own Digit")

        uploaded_file = st.file_uploader(
            "Upload a handwritten digit image (PNG/JPG, roughly 28×28 or square):",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded digit", width=150)

            if st.button("Predict Uploaded Digit"):
                # We already imported preprocess_uploaded_digit_image at top-level
                # but just to be safe/explicit or if user wanted it inside:
                # from app.projects.handwritten_digits import preprocess_uploaded_digit_image

                try:
                    processed = preprocess_uploaded_digit_image(uploaded_file)
                    result = predict_single_digit(bundle, processed)

                    st.write(
                        f"Predicted label: **{result['predicted_label']}** "
                        f"(class={result['predicted_class']}, "
                        f"confidence={result['confidence']:.4f})"
                    )
                except Exception as e:
                    st.error(f"Could not process uploaded image: {e}")
            
    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/HandwrittenDigitDetection.ipynb", height=850)

        if st.button("Open Jupyter File", key="open_digits_nb"):
            open_notebook_file("notebooks/HandwrittenDigitDetection.ipynb")


def page_house_price():
    st.title("🏡 House Price Prediction")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_house_price_bundle()

        st.subheader("Per-Model Metrics")

        raw_metrics = getattr(bundle, "metrics", {}) or {}

        rows = []
        name_map = {
            "svr": "SVR (Support Vector Regression)",
            "rf": "Random Forest Regressor",
            "lr": "Linear Regression",
        }

        for key, values in raw_metrics.items():
            mae = float(values.get("mae", 0.0))
            mape = float(values.get("mape", 0.0)) * 100.0  # convert from fraction to %
            rows.append(
                {
                    "Model": name_map.get(key, key.upper()),
                    "MAE (₹)": mae,
                    "MAPE (%)": mape,
                }
            )

        if rows:
            metrics_df = pd.DataFrame(rows)

            # Nicely formatted table
            st.markdown("#### Model Performance Summary")
            st.table(
                metrics_df.style.format(
                    {
                        "MAE (₹)": "{:,.2f}",
                        "MAPE (%)": "{:.2f}",
                    }
                )
            )

            # Simple bar chart for MAPE comparison
            st.markdown("#### MAPE Comparison (%)")
            chart_df = metrics_df.set_index("Model")[["MAPE (%)"]]
            st.bar_chart(chart_df)
        else:
            st.info("Model metrics are not available.")

        st.markdown(
            """
        We trained **SVR**, **RandomForestRegressor**, and **LinearRegression** on the
        HousePricePrediction dataset.

        You can pass a few important numeric fields; the rest will default to **0** or
        dummy values (since we don't have full UI for all engineered features yet).
        """
        )

        st.markdown(
            """
            #### Input Features

            Adjust a few key characteristics of the house below. The remaining engineered
            features are filled with default values behind the scenes.
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            overall_qual = st.slider(
                "Overall house quality (1–10)",
                min_value=1,
                max_value=10,
                value=7,
                help="Overall material and finish quality of the house. 1 = Very poor, 10 = Excellent.",
            )

            gr_liv_area = st.number_input(
                "Above-ground living area (sq ft)",
                min_value=300,
                max_value=6000,
                value=1500,
                help="Total living area above ground level, in square feet.",
            )

        with col2:
            garage_cars = st.slider(
                "Number of car spaces in garage",
                min_value=0,
                max_value=4,
                value=2,
                help="How many cars can comfortably fit in the garage.",
            )

            total_bsmt_sf = st.number_input(
                "Basement area (sq ft)",
                min_value=0,
                max_value=4000,
                value=800,
                help="Total basement area (finished + unfinished), in square feet.",
            )

        if st.button("Predict SalePrice with all models"):
            input_data = {
                "OverallQual": overall_qual,
                "GrLivArea": gr_liv_area,
                "GarageCars": garage_cars,
                "TotalBsmtSF": total_bsmt_sf,
            }

            preds = predict_house_price(bundle, input_data)

            st.subheader("Predicted Sale Price (by model)")

            # Friendly names for display
            name_map = {
                "svr": "SVR – Support Vector Regression",
                "rf": "Random Forest Regressor",
                "lr": "Linear Regression",
            }

            # Build a small table
            rows = []
            for key, price in preds.items():
                rows.append(
                    {
                        "Model": name_map.get(key.lower(), key.upper()),
                        "Estimated Price (₹)": float(price),
                    }
                )

            if rows:
                pred_df = pd.DataFrame(rows)

                # Nicely formatted table
                st.table(
                    pred_df.style.format({"Estimated Price (₹)": "₹ {:,.0f}"})
                )

                # Optional: highlight which model was best during training (lowest MAE)
                raw_metrics = getattr(bundle, "metrics", {}) or {}
                best_model_key = None
                best_mae = float("inf")

                for key, vals in raw_metrics.items():
                    mae = float(vals.get("mae", float("inf")))
                    if mae < best_mae:
                        best_mae = mae
                        best_model_key = key

                if best_model_key is not None and best_model_key in preds:
                    best_label = name_map.get(best_model_key.lower(), best_model_key.upper())
                    best_price = preds[best_model_key]
                    st.info(
                        f"Based on training performance (lowest MAE), "
                        f"**{best_label}** is the most reliable estimate here: "
                        f"≈ **₹ {best_price:,.0f}**"
                    )
            else:
                st.warning("No predictions were returned by the models.")


    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/HousePricePrediction.ipynb", height=1400)

        if st.button("Open Jupyter File", key="open_house_nb"):
            open_notebook_file("notebooks/HousePricePrediction.ipynb")


def page_iris():
    st.title("🌸 Iris Flower Classification")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_iris_bundle()

        st.subheader("Model Performance")
        metrics = getattr(bundle, "metrics", {}) or {}
        accuracy = float(metrics.get("accuracy", 0.0))

        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.metric("Accuracy", f"{accuracy * 100:.2f}%", help="Share of flowers correctly classified on the test set.")

        if metrics:
            st.markdown("#### Detailed metrics")
            metrics_df = pd.DataFrame(
                [{"Metric": k, "Value": float(v)} for k, v in metrics.items()]
            )
            st.table(metrics_df.style.format({"Value": "{:.4f}"}))

        if metrics and len(metrics) > 1:
            chart_df = metrics_df.set_index("Metric").copy()
            st.bar_chart(chart_df["Value"])

        st.markdown(
            """
            **Feature guide**

            - **Sepal length (cm)** – length of the outer flower part (sepal)
            - **Sepal width (cm)** – width of the sepal
            - **Petal length (cm)** – length of the inner petal
            - **Petal width (cm)** – width of the petal
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            sepal_len = st.number_input(
                "Sepal length (cm)",
                min_value=0.0, max_value=10.0, value=5.1
            )
            sepal_wid = st.number_input(
                "Sepal width (cm)",
                min_value=0.0, max_value=10.0, value=3.5
            )
        with col2:
            petal_len = st.number_input(
                "Petal length (cm)",
                min_value=0.0, max_value=10.0, value=1.4
            )
            petal_wid = st.number_input(
                "Petal width (cm)",
                min_value=0.0, max_value=10.0, value=0.2
            )

        if st.button("Predict Species"):
            sample = {
                "SepalLengthCm": sepal_len,
                "SepalWidthCm": sepal_wid,
                "PetalLengthCm": petal_len,
                "PetalWidthCm": petal_wid,
            }
            result = iris_predict_from_features(bundle, sample)
            predicted = result.get("predicted_label", "Unknown")
            st.success(f"The model predicts this flower is **{predicted}**.")

            probs = result.get("probabilities", None)
            if probs is not None:
                # Case 1: dict of class_name -> prob
                if isinstance(probs, dict):
                    prob_items = [{"Species": k, "Probability": float(v)} for k, v in probs.items()]
                else:
                    # Fallback: if it's a list/ndarray, try to pair with a 'class_labels' attribute on the bundle
                    labels = getattr(bundle, "class_labels", None)
                    if labels is not None and len(labels) == len(probs):
                        prob_items = [
                            {"Species": str(lbl), "Probability": float(p)}
                            for lbl, p in zip(labels, probs)
                        ]
                    else:
                        prob_items = []

                if prob_items:
                    st.markdown("#### Class probabilities")
                    prob_df = pd.DataFrame(prob_items)
                    # Show as percentages in the table
                    st.table(prob_df.style.format({"Probability": "{:.2%}"}))

                    # Also show a bar chart for quick comparison
                    chart_df = prob_df.set_index("Species")
                    st.bar_chart(chart_df["Probability"])
                else:
                    # Last-resort fallback: just print the raw object
                    st.write("Probabilities:", probs)

    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/IrisFlowerClassification.ipynb", height=850)

        if st.button("Open Jupyter File", key="open_iris_nb"):
            open_notebook_file("notebooks/IrisFlowerClassification.ipynb")


def page_diabetes():
    st.title("💉 Diabetes Prediction")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_diabetes_bundle()

        # ------------------------------------------------------------------
        # 1) MODEL METRICS – KPI cards + table + bar chart
        # ------------------------------------------------------------------
        st.subheader("Model Performance")

        raw_metrics = getattr(bundle, "metrics", {}) or {}

        # Map short keys to human-readable model names
        name_map = {
            "rfc": "Random Forest (RFC)",
            "dt": "Decision Tree (DT)",
            "xgb": "XGBoost (XGB)",
            "svc": "Support Vector Classifier (SVC)",
        }

        rows = []
        for key, inner in raw_metrics.items():
            if not isinstance(inner, dict):
                continue
            acc = inner.get("accuracy")
            if acc is None:
                continue
            pretty_name = name_map.get(key, key.upper())
            rows.append(
                {
                    "Model": pretty_name,
                    "Accuracy": float(acc),
                    "Accuracy (%)": float(acc) * 100.0,
                }
            )

        if rows:
            metrics_df = pd.DataFrame(rows)

            # KPI row
            kpi_cols = st.columns(len(rows))
            for col, row in zip(kpi_cols, rows):
                with col:
                    st.metric(
                        row["Model"],
                        f"{row['Accuracy (%)']:.2f} %",
                    )

            # Detailed table
            st.markdown("#### Detailed Metrics")
            st.table(
                metrics_df[["Model", "Accuracy (%)"]].style.format(
                    {"Accuracy (%)": "{:.2f}"}
                )
            )

            # Simple bar chart
            st.markdown("#### Accuracy Comparison")
            chart_df = metrics_df.set_index("Model")["Accuracy (%)"]
            st.bar_chart(chart_df)

        # ------------------------------------------------------------------
        # 2) INPUT FORM – same variables, better explanations
        # ------------------------------------------------------------------
        st.markdown(
            """
We treat **0 values as missing** in several medical fields and impute them
smartly during training.  
You can fill in the values you know – missing ones will fall back to typical
dataset averages.
"""
        )

        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=2,
                help="Number of times you have been pregnant (0 if never).",
            )
            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=0.0,
                max_value=300.0,
                value=120.0,
                help="Blood sugar level measured in a glucose tolerance test.",
            )
            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                help="Diastolic blood pressure (the lower number in a BP reading).",
            )
        with col2:
            bmi = st.number_input(
                "BMI (kg/m²)",
                min_value=0.0,
                max_value=70.0,
                value=30.0,
                help="Body Mass Index – weight relative to height.",
            )
            age = st.number_input(
                "Age (years)",
                min_value=10,
                max_value=120,
                value=35,
                help="Your current age in years.",
            )

        if st.button("Predict Diabetes Risk"):
            input_data = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "BMI": bmi,
                "Age": age,
            }

            results = predict_diabetes(bundle, input_data)

            # --------------------------------------------------------------
            # 3) OVERALL RISK SUMMARY
            # --------------------------------------------------------------
            st.subheader("Overall Risk Summary")

            probs = [
                r.get("probability_diabetes")
                for r in results.values()
                if r.get("probability_diabetes") is not None
            ]

            if probs:
                avg_prob = float(sum(probs) / len(probs))
                avg_percent = avg_prob * 100.0

                # Simple thresholding for a human-friendly label
                if avg_prob < 0.25:
                    emoji = "🟢"
                    level = "Low risk"
                    note = "Most models think diabetes is unlikely."
                elif avg_prob < 0.60:
                    emoji = "🟡"
                    level = "Moderate risk"
                    note = "Some models show signs of possible diabetes."
                else:
                    emoji = "🔴"
                    level = "High risk"
                    note = "Models see a high probability of diabetes."

                st.markdown(
                    f"""
{emoji} **{level}**

Estimated probability of diabetes (averaged across models):  
**{avg_percent:.1f}%**

_{note}_  
"""
                )
            else:
                st.info(
                    "Models did not return diabetes probabilities. "
                    "Per-model predictions are shown below."
                )

            # --------------------------------------------------------------
            # 4) PER-MODEL BREAKDOWN (readable)
            # --------------------------------------------------------------
            st.subheader("Per-Model Predictions")

            pretty_rows = []
            for key, r in results.items():
                prob = r.get("probability_diabetes")
                label = r.get("predicted_label")

                model_name = name_map.get(key, key.upper())

                if prob is not None:
                    prob_percent = prob * 100.0
                else:
                    prob_percent = None

                if label == 1:
                    base_verdict = "Model leans towards **Diabetes**"
                else:
                    base_verdict = "Model leans towards **No Diabetes**"

                # Risk tag based on probability
                if prob is None:
                    risk_tag = "Risk level unknown"
                elif prob < 0.25:
                    risk_tag = "Low risk"
                elif prob < 0.60:
                    risk_tag = "Moderate risk"
                else:
                    risk_tag = "High risk"

                pretty_rows.append(
                    {
                        "Model": model_name,
                        "Verdict": base_verdict.replace("**", ""),
                        "Risk level": risk_tag,
                        "P(diabetes) (%)": prob_percent,
                    }
                )

            if pretty_rows:
                pretty_df = pd.DataFrame(pretty_rows)
                st.table(
                    pretty_df.style.format({"P(diabetes) (%)": "{:.1f}"})
                )

            # --------------------------------------------------------------
            # 5) MEDICAL DISCLAIMER
            # --------------------------------------------------------------
            st.markdown(
                """
> ⚠️ **Important:**  
> This tool is for **educational and demonstration purposes only** and does **not**
> provide medical advice, diagnosis, or treatment.  
> Always consult a qualified healthcare professional for any health-related decisions.
"""
            )

    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/PredictingDiabetes.ipynb", height=1900)

        if st.button("Open Jupyter File", key="open_diabetes_nb"):
            open_notebook_file("notebooks/PredictingDiabetes.ipynb")


def page_titanic():
    st.title("🚢 Titanic Survival Prediction")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_titanic_bundle()

        # -------------------------------
        # Model performance (no raw JSON)
        # -------------------------------
        st.subheader("Model Performance")

        raw_metrics = getattr(bundle, "metrics", {}) or {}
        accuracy = float(raw_metrics.get("accuracy", 0.0))
        precision_survived = float(raw_metrics.get("precision_survived", 0.0))
        recall_survived = float(raw_metrics.get("recall_survived", 0.0))
        f1_survived = float(raw_metrics.get("f1_survived", 0.0))

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Accuracy", f"{accuracy * 100:.1f}%")
        with mcol2:
            st.metric("Precision (Survived)", f"{precision_survived * 100:.1f}%")
        with mcol3:
            st.metric("Recall (Survived)", f"{recall_survived * 100:.1f}%")
        with mcol4:
            st.metric("F1-score (Survived)", f"{f1_survived * 100:.1f}%")

        # Optional small text explanation
        st.markdown(
            """
            These metrics are computed on a validation split of the **Kaggle Titanic dataset**,
            focusing on the **Survived (1)** class.
            """
        )

        st.markdown("---")

        # -------------------------------
        # Passenger configuration with friendly labels
        # -------------------------------
        st.markdown(
            """
            **Configure a hypothetical passenger:**  

            - **Ticket Class** – 1 = First class, 2 = Second class, 3 = Third class  
            - **Siblings/Spouses Onboard** – close family travelling with them (SibSp)  
            - **Parents/Children Onboard** – parents or children travelling with them (Parch)  
            - **Fare** – approximate ticket cost (as in the original dataset)  
            - **Port of Embarkation** – S = Southampton, C = Cherbourg, Q = Queenstown  
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox(
                "Ticket Class (1 = First, 2 = Second, 3 = Third)",
                options=[1, 2, 3],
                index=2,
            )
            sex = st.selectbox(
                "Passenger Sex",
                options=["male", "female"],
                index=0,
            )
            age = st.number_input(
                "Age (years)",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
            )
        with col2:
            sibsp = st.number_input(
                "Siblings/Spouses Onboard",
                min_value=0,
                max_value=10,
                value=0,
            )
            parch = st.number_input(
                "Parents/Children Onboard",
                min_value=0,
                max_value=10,
                value=0,
            )
            fare = st.number_input(
                "Ticket Fare",
                min_value=0.0,
                max_value=600.0,
                value=32.2,
            )
            embarked = st.selectbox(
                "Port of Embarkation (S, C, Q)",
                options=["S", "C", "Q"],
                index=0,
            )

        if st.button("Predict Survival"):
            example = {
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": embarked,
            }
            result = predict_survival(bundle, example)

            prob_survived = float(result["probability_survived"])
            prob_percent = prob_survived * 100.0
            label = result["predicted_class_name"]  # "Survived" or "Not Survived"

            if label.lower().startswith("survived"):
                st.success(
                    f"🛟 The model estimates a **{prob_percent:.1f}% chance** that this passenger "
                    f"**would survive**.\n\n"
                    f"Because this probability is above 50%, the final prediction is **Survived**."
                )
            else:
                st.error(
                    f"⚠️ The model estimates a **{prob_percent:.1f}% chance** that this passenger "
                    f"**would survive**.\n\n"
                    f"Because this probability is below 50%, the final prediction is **Not Survived**."
                )

            st.caption(
                "This is a statistical estimate based on the Kaggle Titanic dataset – "
                "it explains what the model has learned, not a real-life guarantee."
            )

    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/TitanicSurvivalPrediction.ipynb", height=1100)

        if st.button("Open Jupyter File", key="open_titanic_nb"):
            open_notebook_file("notebooks/TitanicSurvivalPrediction.ipynb")


def page_sentiment():
    st.title("🎭 Sentiment Analysis")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_sentiment_bundle()

        st.markdown(
            """
        This uses a **pre-trained LSTM model** on the IMDB dataset.

        Type a review and we’ll classify it as **positive** or **negative**.
        """
        )

        text = st.text_area(
            "Enter a movie review:",
            value="The plot was weak but the acting was great.",
            height=150,
        )

        if st.button("Analyze Sentiment"):
            result = predict_sentiment(bundle, text)
            st.write(f"Text: {result['text']}")
            st.success(
                f"Predicted: **{result['predicted_label']}** "
                f"(class={result['predicted_class']}, confidence={result['confidence']:.4f})"
            )

    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/SentimentAnalysisonMovieReviews.ipynb", height=600)

        if st.button("Open Jupyter File", key="open_sentiment_nb"):
            open_notebook_file("notebooks/SentimentAnalysisonMovieReviews.ipynb")


def page_spam():
    st.title("📧 Spam Email Detection")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_spam_bundle()

        st.markdown(
            """
        This model is an **LSTM-based spam classifier** trained on the spam/ham dataset.

        Enter email content below and check whether the model thinks it's spam.
        """
        )

        email = st.text_area(
            "Email text",
            value="Congratulations! You've won a $1000 gift card. Click here to claim your prize now!!!",
            height=150,
        )

        if st.button("Check if Spam"):
            result = predict_spam(bundle, email)
            st.write("Email:", result["email"])
            st.success(
                f"Predicted: **{result['predicted_label']}** "
                f"(class={result['predicted_class']}, prob_spam={result['probability_spam']:.4f}, "
                f"confidence={result['confidence']:.4f})"
            )

    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/SpamEmailDetection.ipynb", height=600)

        if st.button("Open Jupyter File", key="open_spam_nb"):
            open_notebook_file("notebooks/SpamEmailDetection.ipynb")


def page_stock():
    st.title("📈 Stock Price Movement – Tesla")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_stock_bundle()
        df = get_stock_dataframe()

        st.subheader("Model Validation Metrics")
        st.write(bundle.metrics)

        st.markdown(
            """
        We train a **RandomForestClassifier** on Tesla historical data and predict
        whether the **next day** closes **up** or **down**.
        """
        )

        if st.button("Predict Next-Day Movement (using latest data)"):
            try:
                result = predict_next_day_movement(bundle, df)
                st.success(
                    f"Prediction: **{result['predicted_label'].upper()}** "
                    f"(class={result['predicted_class']}, "
                    f"P(up)={result['probability_up']:.3f}, "
                    f"confidence={result['confidence']:.3f})"
                )
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/StockPricePrediction.ipynb", height=600)

        if st.button("Open Jupyter File", key="open_stock_nb"):
            open_notebook_file("notebooks/StockPricePrediction.ipynb")


def page_movie_recommender():
    st.title("🎬 Movie Recommendation System")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_movie_recommender_bundle()

        st.subheader("Dataset Overview")
        st.write(f"Number of movies: `{len(bundle.movie_stats)}`")
        st.write(bundle.movie_stats.head())

        st.markdown(
            """
        The recommender is **item–item correlation based**.  
        Give it a movie title and it returns similar movies with high rating correlation
        and sufficient rating counts.
        """
        )

        default_movie = bundle.movie_stats.sort_values(
            by="rating_count", ascending=False
        ).head(1).index.tolist()[0]

        movie_title = st.text_input("Movie title", value=default_movie)

        min_ratings = st.number_input(
            "Minimum number of ratings", min_value=1, max_value=500, value=bundle.min_ratings_default
        )

        top_n = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5)

        if st.button("Get Recommendations"):
            result = safe_recommend(bundle, movie_title, top_n=top_n, min_ratings=min_ratings)

            if "error" in result and result["error"]:
                st.error(result["error"])
            else:
                st.subheader(f"Recommendations for: {result['query']}")
                rows = result.get("results", [])
                if not rows:
                    st.write("No recommendations found. Try lowering min_ratings.")
                else:
                    for r in rows:
                        st.write(
                            f"**{r['title']}** – corr={r['correlation']:.3f}, "
                            f"count={r['rating_count']}, mean_rating={r['rating_mean']:.2f}"
                        )

    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/MovieRecommendationSystem.ipynb", height=600)

        if st.button("Open Jupyter File", key="open_movie_nb"):
            open_notebook_file("notebooks/MovieRecommendationSystem.ipynb")


# =======================================================================================
#                                       MAIN
# =======================================================================================

def main():
    st.set_page_config(
        page_title="MLLab – Interactive Machine Learning Lab",
        page_icon="🧪",
        layout="wide",
    )

    pages = {
        "Overview": page_overview,
        "Breast Cancer Prediction": page_breast_cancer,
        "Handwritten Digit Detection": page_handwritten_digits,
        "House Price Prediction": page_house_price,
        "Iris Flower Classification": page_iris,
        "Diabetes Prediction": page_diabetes,
        "Titanic Survival Prediction": page_titanic,
        "Sentiment Analysis": page_sentiment,
        "Spam Email Detection": page_spam,
        "Stock Price Prediction": page_stock,
        "Movie Recommendation System": page_movie_recommender,
    }

    st.sidebar.title("🔭 MLLab Projects")
    
    # Changed from selectbox to radio for vertical list
    choice = st.sidebar.radio(
        "Choose a mini-project",
        list(pages.keys())
    )

    # Render selected page
    pages[choice]()


if __name__ == "__main__":
    main()
