import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

@st.cache_resource
def load_sentiment_analyzer():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

sia = load_sentiment_analyzer()

@st.cache_resource
def load_ml_artifacts():
    model = joblib.load("mental_health_model.pkl")
    encoders = joblib.load("feature_encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, encoders, target_encoder, feature_columns

model, encoders, target_encoder, feature_columns = load_ml_artifacts()

st.set_page_config(page_title="AI Mental Health Monitor", page_icon="🧠", layout="centered")
st.title("🧠 AI-Based Mental Health Monitoring")
st.write(
    "This tool combines a survey-based ML risk model with NLP sentiment analysis "
    "of your own words. It is **not a diagnostic tool** — please seek professional "
    "help if you're struggling."
)

tab1, tab2, tab3 = st.tabs(["📋 Questionnaire (ML)", "📝 Journal Entry (NLP)", "📊 Combined Result"])

def safe_encode(col, value):
    le = encoders[col]
    if value not in le.classes_:
        return 0
    return le.transform([value])[0]

def analyze_journal_entry(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound <= -0.5:
        mood = "Highly Negative"
    elif compound <= -0.1:
        mood = "Negative"
    elif compound < 0.1:
        mood = "Neutral"
    elif compound < 0.5:
        mood = "Positive"
    else:
        mood = "Highly Positive"
    return mood, scores

def combined_risk_assessment(ml_pred, journal_mood):
    if ml_pred == "Yes" and journal_mood in ["Negative", "Highly Negative"]:
        return "High"
    elif ml_pred == "Yes" or journal_mood in ["Negative", "Highly Negative"]:
        return "Moderate"
    return "Low"

with tab1:
    st.subheader("Answer a few questions")
    gender = st.selectbox("Gender", ["Male", "Female"])
    country = st.text_input("Country", "United States")
    occupation = st.selectbox("Occupation", ["Corporate", "Student", "Business", "Housewife", "Others"])
    self_employed = st.selectbox("Self-employed?", ["Yes", "No"])
    family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
    days_indoors = st.selectbox("Days spent indoors recently",
                                 ["Go out Every day", "1-14 days", "15-30 days", "31-60 days", "More than 2 months"])
    growing_stress = st.selectbox("Do you feel growing stress?", ["Yes", "No", "Maybe"])
    changes_habits = st.selectbox("Noticed changes in habits?", ["Yes", "No", "Maybe"])
    mh_history = st.selectbox("Personal mental health history?", ["Yes", "No", "Maybe"])
    mood_swings = st.selectbox("Mood swings", ["Low", "Medium", "High"])
    coping_struggles = st.selectbox("Struggling to cope?", ["Yes", "No"])
    work_interest = st.selectbox("Lost interest in work?", ["Yes", "No", "Maybe"])
    social_weakness = st.selectbox("Feeling social weakness/withdrawal?", ["Yes", "No", "Maybe"])
    mh_interview = st.selectbox("Comfortable discussing mental health in an interview?", ["Yes", "No", "Maybe"])
    care_options = st.selectbox("Aware of care options at work/school?", ["Yes", "No", "Not sure"])

    if st.button("Predict Risk from Questionnaire"):
        raw_input = {
            "Gender": gender, "Country": country, "Occupation": occupation,
            "self_employed": self_employed, "family_history": family_history,
            "Days_Indoors": days_indoors, "Growing_Stress": growing_stress,
            "Changes_Habits": changes_habits, "Mental_Health_History": mh_history,
            "Mood_Swings": mood_swings, "Coping_Struggles": coping_struggles,
            "Work_Interest": work_interest, "Social_Weakness": social_weakness,
            "mental_health_interview": mh_interview, "care_options": care_options,
        }
        encoded_row = [safe_encode(col, raw_input[col]) for col in feature_columns]
        X_new = pd.DataFrame([encoded_row], columns=feature_columns)
        pred_encoded = model.predict(X_new)[0]
        pred_label = target_encoder.inverse_transform([pred_encoded])[0]
        st.session_state["ml_pred"] = pred_label
        st.success(f"Model prediction: **{pred_label}** likely to need treatment/support")

with tab2:
    st.subheader("Write how you're feeling today")
    journal_text = st.text_area("Journal entry", height=150,
                                 placeholder="e.g. I've been feeling really overwhelmed with work this week...")
    if st.button("Analyze Journal Entry"):
        if journal_text.strip():
            mood, scores = analyze_journal_entry(journal_text)
            st.session_state["journal_mood"] = mood
            st.info(f"Detected mood: **{mood}**")
            st.write("Sentiment scores:", scores)
        else:
            st.warning("Please write something first.")

with tab3:
    st.subheader("Combined Monitoring Result")
    ml_pred = st.session_state.get("ml_pred")
    journal_mood = st.session_state.get("journal_mood")
    if ml_pred and journal_mood:
        risk = combined_risk_assessment(ml_pred, journal_mood)
        st.metric("Questionnaire prediction", ml_pred)
        st.metric("Journal mood (NLP)", journal_mood)
        color = {"Low": "green", "Moderate": "orange", "High": "red"}[risk]
        st.markdown(f"### Overall risk level: :{color}[{risk}]")
        if risk == "High":
            st.error("This combined result suggests elevated risk. Please consider talking "
                      "to a mental health professional or a trusted person soon.")
        elif risk == "Moderate":
            st.warning("Keep an eye on how you're feeling, and consider talking to someone you trust.")
        else:
            st.success("No strong signs of elevated risk right now — keep taking care of yourself.")
    else:
        st.info("Complete both the Questionnaire tab and the Journal Entry tab to see a combined result.")

st.divider()
st.caption("⚠️ This app is a student/portfolio project for educational purposes and is NOT a "
           "substitute for professional diagnosis or treatment.")
