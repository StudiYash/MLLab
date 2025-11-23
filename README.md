# ![MLLab](https://github.com/StudiYash/MLLab/blob/main/assets/logo.png)

## 🛡️ Project Introduction

### Abstract 
**MLLab** is a unified home for multiple small ML/DL projects (regression, classification, NLP, recommender systems, time-series/stock movement, etc.).
Originally, each project was a standalone notebook; now they are being refactored into a single, organized, Streamlit-based lab.
The goal is both: learning (understanding different ML tasks) and portfolio (presenting them as one cohesive app).

### Project Timeline 

- **Start Date**: 20th November 2025  
- **End Date**: 23rd November 2025 
- **Total Time Required**: 4 days 

### My Introduction  

| Name                   | GitHub Profile | LinkedIn Profile |
|------------------------|----------------|------------------|
| **Yash Suhas Shukla**  | [GitHub](https://github.com/StudiYash) | [LinkedIn](https://www.linkedin.com/in/yash-shukla-2024aiguy/) |

<div align="center">
  <img src="https://github.com/StudiYash/MLLab/blob/main/assets/About%20Me.gif" alt="Introduction GIF" width="800" height="450">
</div>

---

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

---

## 🛠 Tech Stack

- Python 3.x
- Streamlit
- scikit-learn
- TensorFlow / Keras (for DL projects)
- XGBoost, CatBoost (where used)
- Pandas, NumPy, Matplotlib, Seaborn

---
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

   > **Note:** Some models (like Handwritten Digits, Spam Detection) may take a moment to train on the first run. They will be saved to the `models/` directory for faster loading in subsequent runs.

## ✨ Streamlit Interface

<p align="center">
    <img src="https://github.com/StudiYash/MLLab/blob/main/assets/demo_images/home_screen.png" alt="Streamlit UI Home Page" width="75%" />
    </p>

[![Application Interface Images](https://img.shields.io/badge/View-Application%20Interface%20Images-blue?style=for-the-badge&logo=github)](https://github.com/StudiYash/MLLab/tree/main/assets/demo_images)

---

## ⚠️ Disclaimer

This application is for **educational and demonstration purposes only**. 
- **Medical predictions** (Breast Cancer, Diabetes) are NOT medical advice.
- **Financial/Survival predictions** (Stock, Titanic) are based on historical data and are not guarantees of future outcomes.

## How You Can Help 🙌

Even though it's free to use, **MLLab** grows best when the community jumps in!

- 🧩 Add new mini-projects or ML experiments
- 🛠 Refactor code, clean notebooks, or optimize pipelines
- 📊 Improve visualizations, metrics, and explanations
- 🧪 Create exercises, challenges, or “try it yourself” sections
- 🐛 Report bugs or suggest new features via issues

> Just fork the repo, push your changes, and open a pull request.  
> Every small improvement makes MLLab a better lab for everyone. ⚗️

---

## License & Usage 📜

This project is governed by the **Creative Commons Zero (CC0 1.0 Universal)** license.

> **That means:**  
> 🔓 You can **use, modify, share, teach, remix, or build** on this work freely — even for **commercial** purposes.  
> 🎁 **No permission or attribution required.** This is my gift to the Python community.

🔗 [View Full License Text → CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

![License: CC0 1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)

---

## Final Words ❤️

> Machine Learning feels complex — but it doesn’t have to stay mysterious.  
> **MLLab** is here to turn scattered ML concepts into clear, hands-on mini-projects you can actually run, tweak, and learn from.

If you found this helpful, consider **starring the repository ⭐**,  
sharing it with a friend, or contributing a new experiment.

---

**Stay curious. Experiment boldly. Build ML projects that actually run. ⚗️🤖**
