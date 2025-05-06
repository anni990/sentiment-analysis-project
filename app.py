import streamlit as st
import joblib
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- Configure Streamlit page ---
st.set_page_config(page_title="Emotion AI", page_icon="💬", layout="centered")

# --- Load the model and vectorizer ---
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- UI styling ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1578662996442-48f60103fc96?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #22C1C3;'>💬 Emotion AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #FFCB59;'> Detect whether a sentence expresses <b>Positive</b> or <b>Negative</b> sentiment using a Logistic Regression model trained on the IMDB dataset.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Input section ---
user_input = st.text_area("Enter a message to analyze:", "", height=100, max_chars=500)

# --- Button to trigger sentiment analysis ---
if st.button("🔍 Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("Running sentiment analysis..."):
            # Preprocess input
            input_clean = " ".join([w for w in user_input.lower().split() if w not in set(stopwords.words('english'))])
            input_vec = vectorizer.transform([input_clean])

            # Predict
            prediction = model.predict(input_vec)[0]
            prob = model.predict_proba(input_vec)[0][prediction]

            label = "POSITIVE" if prediction == 1 else "NEGATIVE"
            color = "#4CAF50" if label == "POSITIVE" else "#F44336"

            # Show result
            st.markdown(f"<h3 style='color:{color}; text-align:center;'>Sentiment: {label}</h3>", unsafe_allow_html=True)
            st.progress(prob)
            st.markdown(f"**Confidence Score:** `{prob * 100:.2f}%`")

# --- Footer ---
st.markdown("---")
st.markdown("<center><sub>🔗 Developed by Aniket Kumar Mishra | Made with ❤️ in Streamlit</sub></center>", unsafe_allow_html=True)
