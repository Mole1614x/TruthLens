
import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load datasets
folder = PROJECT_ROOT / "data" / "processed" / "splits"

train = pd.read_csv(folder / "train.csv")
validation = pd.read_csv(folder / "validation.csv")

print("Training records:", len(train))
print("Validation records:", len(validation))

# Separate text and labels
X_train = train["combined_text"].fillna("")
y_train = train["label"].astype(int)

X_validation = validation["combined_text"].fillna("")
y_validation = validation["label"].astype(int)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_validation_tfidf = vectorizer.transform(X_validation)

print("TF-IDF conversion completed")
print("Number of features:", len(vectorizer.get_feature_names_out()))

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("Model training completed")

# Evaluate on validation data
y_pred = model.predict(X_validation_tfidf)

print("\nValidation Accuracy:", accuracy_score(y_validation, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_validation,
    y_pred,
    target_names=["Fake", "Real"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_validation, y_pred))

# Save model and vectorizer
models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(model, models_dir / "logistic_regression.pkl")
joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")

