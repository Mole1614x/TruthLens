
import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report

# Load validation data
data = pd.read_csv("data/processed/splits/validation.csv")

X = data["combined_text"].fillna("")
y = data["label"].astype(int)

# Load the fitted vectorizer and tuned model
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/tuned_linear_svm.pkl")

# Make predictions
X_tfidf = vectorizer.transform(X)
predictions = model.predict(X_tfidf)

# Display evaluation results
print("Confusion Matrix:")
print(confusion_matrix(y, predictions))

print("\nClassification Report:")
print(classification_report(
    y,
    predictions,
    target_names=["Fake", "Real"]
))

# Find incorrectly classified articles
data["actual_label"] = y
data["predicted_label"] = predictions

errors = data[data["actual_label"] != data["predicted_label"]].copy()

errors["actual_label"] = errors["actual_label"].map({
    0: "Fake",
    1: "Real"
})

errors["predicted_label"] = errors["predicted_label"].map({
    0: "Fake",
    1: "Real"
})

# Save incorrect predictions for inspection
errors[[
    "title",
    "actual_label",
    "predicted_label",
    "combined_text"
]].to_csv("models/misclassified_articles.csv", index=False)

print("\nTotal validation articles:", len(data))
print("Incorrect predictions:", len(errors))
print("Misclassified articles saved to models/misclassified_articles.csv")

print("\nSample incorrect predictions:")
print(errors[[
    "title",
    "actual_label",
    "predicted_label"
]].head(10).to_string(index=False))
