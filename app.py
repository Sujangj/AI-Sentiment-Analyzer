import streamlit as st
from transformers import pipeline

# Load AI model once
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

st.title("🤖 AI Sentiment Analyzer")
st.write("Analyze text sentiment using AI.")

user_input = st.text_area("Enter text:", "I love learning AI!")

if st.button("Analyze Sentiment"):
    with st.spinner("Analyzing..."):
        model = load_model()
        result = model(user_input)[0]
        sentiment = result['label']
        confidence = result['score']
        st.success(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")
