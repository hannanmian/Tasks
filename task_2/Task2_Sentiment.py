"""
Task 2: Sentiment Analysis
--------------------------
This script builds a simple sentiment analysis model using a small IMDB-like dataset.

Steps:
1. Load dataset (CSV with 1000 reviews).
2. Preprocess text (lowercase, remove special characters).
3. Convert text to features using CountVectorizer.
4. Train Logistic Regression model.
5. Evaluate with accuracy.
6. Predict sentiment for new reviews.
"""

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("imdb_1000_reviews.csv")

# 2. Preprocessing function
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    return text

df["clean_review"] = df["review"].apply(preprocess_text)

# 3. Convert text to features
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Evaluate accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# 6. Predict new reviews
sample_reviews = [
    "I really enjoyed this movie, so good!",
    "It was awful, I hated every moment.",
    "Not bad, but could have been better."
]
sample_clean = [preprocess_text(r) for r in sample_reviews]
sample_features = vectorizer.transform(sample_clean)
preds = model.predict(sample_features)

for review, pred in zip(sample_reviews, preds):
    print(f"Review: '{review}' => Sentiment: {'Positive' if pred == 1 else 'Negative'}")
