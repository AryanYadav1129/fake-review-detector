import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data (replace with your actual dataset)
reviews = ["This product is great!", "Worst purchase ever!", "Highly recommended!", "Fake and terrible quality!"]
labels = [1, 0, 1, 0]  # 1 = real, 0 = fake

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)

# Train a simple Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)

# Save the trained model
joblib.dump(model, "fake_review_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
