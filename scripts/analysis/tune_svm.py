
import time
import joblib
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load training and validation datasets
train_data = pd.read_csv("data/processed/splits/train.csv")
validation_data = pd.read_csv("data/processed/splits/validation.csv")

X_train = train_data["combined_text"].fillna("")
y_train = train_data["label"].astype(int)

X_validation = validation_data["combined_text"].fillna("")
y_validation = validation_data["label"].astype(int)

print("Training records:", len(X_train))
print("Validation records:", len(X_validation))

# Load the vectorizer fitted on training data
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

X_train = vectorizer.transform(X_train)
X_validation = vectorizer.transform(X_validation)

# Values of C to test
c_values = [0.1, 0.5, 1.0, 2.0]

results = []
best_f1 = -1
best_model = None

for c in c_values:
    print("\nTesting C =", c)

    model = LinearSVC(C=c)

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    predictions = model.predict(X_validation)

    accuracy = accuracy_score(y_validation, predictions)
    precision = precision_score(
        y_validation, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_validation, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_validation, predictions, average="weighted", zero_division=0
    )

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("Training time:", round(training_time, 2), "seconds")

    results.append({
        "C": c,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Training Time": training_time
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# Save tuning results
results_df = pd.DataFrame(results)
results_df.to_csv("models/svm_tuning_results.csv", index=False)

# Save the best model separately
joblib.dump(best_model, "models/tuned_linear_svm.pkl")

print("\nTUNING RESULTS")
print(results_df.round(4).to_string(index=False))

print("\nBest C:", best_model.C)
print("Best validation F1-score:", round(best_f1, 4))
print("Best model saved: models/tuned_linear_svm.pkl")
