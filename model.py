import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Sample dataset (you can replace with real CSV later)
data = {
    "text": [
        "Win money now",
        "Hello, how are you?",
        "Free offer just for you",
        "Meeting scheduled tomorrow",
        "Claim your prize"
    ],
    "label": [1, 0, 1, 0, 1]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Convert text to features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved successfully!")
