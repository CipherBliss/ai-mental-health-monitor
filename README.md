# ai-mental-health-monitor

# 🧠 AI-Based Mental Health Monitoring System

An AI-powered web application that combines **Machine Learning** and **Natural Language Processing (NLP)** to assess mental health risk. The system uses a survey-based ML model to predict the likelihood of needing mental health treatment, paired with an NLP sentiment analyzer that evaluates a user's own written journal entries — combining both signals into an overall risk assessment.

🔗 **Live App:** https://ai-mental-health-monitor-cipherbliss.streamlit.app

---

---

## 📋 Overview

Mental health issues often go unnoticed until they become severe. This project aims to provide an early, accessible, AI-assisted way for individuals to reflect on their mental state through two complementary approaches:

1. **Structured questionnaire (ML)** — Answers a set of lifestyle and behavioral questions (stress levels, mood swings, coping ability, etc.) and predicts whether the person is likely to need treatment/support.
2. **Free-text journal entry (NLP)** — Analyzes the sentiment of the user's own words to detect mood in a more natural, unstructured way.
3. **Combined Risk Score** — Merges both signals into a single Low / Moderate / High risk indicator.

⚠️ **Disclaimer:** This is an educational/portfolio project and is **not a diagnostic tool**. It does not replace professional medical or psychological advice.

---

## 🗂️ Dataset

- **Source:** Mental Health Dataset (survey-based, ~292,000 records)
- **Features:** Gender, Country, Occupation, Self-employed status, Family history, Days spent indoors, Growing stress, Changes in habits, Mental health history, Mood swings, Coping struggles, Work interest, Social weakness, Mental health interview comfort, Care options awareness
- **Target:** `treatment` — whether the individual is likely to need mental health treatment (Yes/No)

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| ML Models | Scikit-learn (Logistic Regression, Decision Tree, Random Forest) |
| NLP | NLTK (VADER Sentiment Analyzer) |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Persistence | Joblib |
| Development | Google Colab |
| Deployment | GitHub + Streamlit Community Cloud |

---

## 🤖 Machine Learning Pipeline

1. **Data Preprocessing** — Handled missing values, label-encoded all categorical features.
2. **Train/Test Split** — 80/20 split, stratified on the target variable.
3. **Model Training** — Trained and evaluated three models:

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 70.33% |
| **Decision Tree (max_depth=8)** | **76.33%** |
| Random Forest (n_estimators=200) | 75.69% |

4. **Model Selection** — Compared training vs. test accuracy to check for overfitting before selecting the final model.
5. **Feature Importance** — Identified which questionnaire responses (e.g. Coping Struggles, Growing Stress) most influence the prediction.

---

## 💬 NLP Module

Since the dataset itself contains no free text, a **VADER Sentiment Analyzer** (NLTK) module was added to let users type a journal entry describing how they feel. The text is scored on a compound sentiment scale and classified into:

- Highly Negative
- Negative
- Neutral
- Positive
- Highly Positive

This mood label is combined with the ML questionnaire prediction to generate the final risk level.

---

## 🌐 Application Features

The Streamlit web app has three tabs:

1. **📋 Questionnaire (ML)** — Fill out the survey to get an ML-based treatment-need prediction.
2. **📝 Journal Entry (NLP)** — Write freely about how you're feeling to get an NLP-based mood analysis.
3. **📊 Combined Result** — View the merged risk assessment (Low / Moderate / High) based on both inputs.

---

## 📁 Project Structure

```
ai-mental-health-monitor/
│
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── mental_health_model.pkl    # Trained ML model
├── feature_encoders.pkl       # Label encoders for input features
├── target_encoder.pkl         # Label encoder for the target variable
├── feature_columns.pkl        # Ordered list of feature columns
├── README.md                  # Project documentation
└── screenshots/
    └── app_screenshot.png     # App preview image
```

---

## ⚙️ Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-mental-health-monitor.git
cd ai-mental-health-monitor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```


---

## 🚀 Deployment

This project is deployed on **Streamlit Community Cloud**, connected directly to this GitHub repository. Any update pushed to the `main` branch automatically redeploys the live app.

---

## 📈 Future Improvements

- Add cross-validation for more robust model comparison
- Upgrade NLP module to a transformer-based emotion classifier (e.g. DistilRoBERTa) for finer-grained mood detection
- Add mood-tracking history/dashboard for returning users
- Add multi-language support for the journal entry analysis

---



---

## 👤 Author: Ritusree Banerjee

Developed as an AI/ML + NLP portfolio project — combining survey-based classification with sentiment-based text analysis for mental health awareness.
