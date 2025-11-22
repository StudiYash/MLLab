# app/main.py

# --- Bootstrap so "app" package imports work with `streamlit run app/main.py`
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]  # repo root (folder that contains "app")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ==== Import project backends ====

from app.projects.breast_cancer import (
    train_breast_cancer_model,
    predict_from_partial_features,
)

from app.projects.handwritten_digits import (
    train_digit_model,
    DigitModelBundle,
    predict_single_digit,
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

        st.subheader("Model Metrics")
        st.write(bundle.metrics)

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
        show_notebook_viewer("notebooks/BreastCancerPrediction.ipynb", height=600)

        if st.button("Open Jupyter File", key="open_bc_nb"):
            open_notebook_file("notebooks/BreastCancerPrediction.ipynb")


def page_handwritten_digits():
    st.title("✏️ Handwritten Digit Detection")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        st.markdown(
            """
        This model is trained on a CSV version of a digit dataset (784 pixel columns).  
        For now, we provide a quick **random-sample demo**: we pick one validation image
        from the trained model and run inference.
        """
        )

        bundle = get_digit_bundle()

        st.subheader("Validation Set Snapshot")
        st.write(f"Validation set size: {bundle.X_val.shape[0]} images")
        st.write(f"Validation accuracy (from training): `{bundle.val_accuracy:.4f}`")

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
            
    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/HandwrittenDigitDetection.ipynb", height=600)

        if st.button("Open Jupyter File", key="open_digits_nb"):
            open_notebook_file("notebooks/HandwrittenDigitDetection.ipynb")


def page_house_price():
    st.title("🏡 House Price Prediction")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_house_price_bundle()

        st.subheader("Per-Model Metrics")
        st.write(bundle.metrics)

        st.markdown(
            """
        We trained **SVR**, **RandomForestRegressor**, and **LinearRegression** on the
        HousePricePrediction dataset.

        You can pass a few important numeric fields; the rest will default to **0** or
        dummy values (since we don't have full UI for all engineered features yet).
        """
        )

        # We don't know all feature names without inspecting X,
        # but users can still experiment with a couple of obvious ones:
        # Typical Kaggle house price cols: 'OverallQual', 'GrLivArea', etc.
        col1, col2 = st.columns(2)
        with col1:
            overall_qual = st.number_input("OverallQual (1–10)", min_value=1, max_value=10, value=7)
            gr_liv_area = st.number_input("GrLivArea (sq ft)", min_value=300, max_value=6000, value=1500)
        with col2:
            garage_cars = st.number_input("GarageCars", min_value=0, max_value=4, value=2)
            total_bsmt_sf = st.number_input(
                "TotalBsmtSF (sq ft)", min_value=0, max_value=4000, value=800
            )

        if st.button("Predict SalePrice with all models"):
            input_data = {
                "OverallQual": overall_qual,
                "GrLivArea": gr_liv_area,
                "GarageCars": garage_cars,
                "TotalBsmtSF": total_bsmt_sf,
            }

            preds = predict_house_price(bundle, input_data)
            st.subheader("Predicted Prices")
            for model_name, price in preds.items():
                st.write(f"**{model_name.upper()}**: `{price:,.2f}`")

    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/HousePricePrediction.ipynb", height=600)

        if st.button("Open Jupyter File", key="open_house_nb"):
            open_notebook_file("notebooks/HousePricePrediction.ipynb")


def page_iris():
    st.title("🌸 Iris Flower Classification")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_iris_bundle()

        st.subheader("Model Metrics")
        st.write(bundle.metrics)

        st.markdown("Enter petal & sepal measurements in **cm**:")

        col1, col2 = st.columns(2)
        with col1:
            sepal_len = st.number_input("SepalLengthCm", min_value=0.0, max_value=10.0, value=5.1)
            sepal_wid = st.number_input("SepalWidthCm", min_value=0.0, max_value=10.0, value=3.5)
        with col2:
            petal_len = st.number_input("PetalLengthCm", min_value=0.0, max_value=10.0, value=1.4)
            petal_wid = st.number_input("PetalWidthCm", min_value=0.0, max_value=10.0, value=0.2)

        if st.button("Predict Species"):
            sample = {
                "SepalLengthCm": sepal_len,
                "SepalWidthCm": sepal_wid,
                "PetalLengthCm": petal_len,
                "PetalWidthCm": petal_wid,
            }
            result = iris_predict_from_features(bundle, sample)
            st.success(f"Predicted: **{result['predicted_label']}**")
            st.write(f"Probabilities: {result['probabilities']}")

    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/IrisFlowerClassification.ipynb", height=600)

        if st.button("Open Jupyter File", key="open_iris_nb"):
            open_notebook_file("notebooks/IrisFlowerClassification.ipynb")


def page_diabetes():
    st.title("💉 Diabetes Prediction")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_diabetes_bundle()

        st.subheader("Model Accuracies")
        st.write(bundle.metrics)

        st.markdown(
            """
        We treat zeros as missing in several medical fields and impute them cleverly
        for training. Here, you can supply a subset of features; missing fields default
        to dataset-level means.
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
            glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
            blood_pressure = st.number_input(
                "BloodPressure", min_value=0.0, max_value=200.0, value=70.0
            )
        with col2:
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0)
            age = st.number_input("Age", min_value=10, max_value=120, value=35)

        if st.button("Predict Diabetes Risk"):
            input_data = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "BMI": bmi,
                "Age": age,
            }

            results = predict_diabetes(bundle, input_data)
            st.subheader("Per-Model Predictions")
            for name, r in results.items():
                label = "Diabetes" if r["predicted_label"] == 1 else "No Diabetes"
                prob = r["probability_diabetes"]
                if prob is not None:
                    st.write(f"**{name.upper()}** → {label} (P(diabetes)={prob:.3f})")
                else:
                    st.write(f"**{name.upper()}** → {label} (probability N/A)")

    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/PredictingDiabetes.ipynb", height=600)

        if st.button("Open Jupyter File", key="open_diabetes_nb"):
            open_notebook_file("notebooks/PredictingDiabetes.ipynb")


def page_titanic():
    st.title("🚢 Titanic Survival Prediction")
    
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown("### 🧪 Interactive View")
        bundle = get_titanic_bundle()

        st.subheader("Validation Metrics")
        st.write(bundle.metrics)

        st.markdown("Configure a hypothetical passenger:")

        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Pclass", options=[1, 2, 3], index=2)
            sex = st.selectbox("Sex", options=["male", "female"], index=0)
            age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
        with col2:
            sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
            parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
            fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2)
            embarked = st.selectbox("Embarked", options=["S", "C", "Q"], index=0)

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

            st.write(
                f"Predicted: **{result['predicted_class_name']}** "
                f"(prob_survived={result['probability_survived']:.3f})"
            )

    with col_right:
        st.subheader("📔Notebook View")
        show_notebook_viewer("notebooks/TitanicSurvivalPrediction.ipynb", height=600)

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
