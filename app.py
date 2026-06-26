import streamlit as st
import joblib

# Load model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Email Spam Detection System")
st.write("Detect whether a message is Spam or Not using Machine Learning (NLP + TF-IDF).")

msg = st.text_area("Enter Email Message")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message")
    else:
        data = vectorizer.transform([msg])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
