import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("📧 Email Spam Detector")

data = {
    "text": [
        "Win money now",
        "Hello friend",
        "Free offer claim now",
        "Meeting tomorrow"
    ],
    "label": [1, 0, 1, 0]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

msg = st.text_area("Enter message")

if st.button("Predict"):
    if msg:
        X_input = vectorizer.transform([msg])
        pred = model.predict(X_input)

        if pred[0] == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
