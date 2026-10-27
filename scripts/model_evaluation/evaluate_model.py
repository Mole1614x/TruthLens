
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load the saved model and TF-IDF vectorizer
model = joblib.load("models/logistic_regression.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load the separate test dataset
test_data = pd.read_csv("data/processed/splits/test.csv")

# Separate text and actual labels
X_test = test_data["combined_text"].fillna("")
y_test = test_data["label"].astype(int)

# Convert test text using the already-fitted vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Predict the labels
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
