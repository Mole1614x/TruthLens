import time
import pandas as pd
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,classification_report,precision_score,recall_score,f1_score
)

# Load training and validation data
train_data = pd.read_csv("data/processed/splits/train.csv")
validation_data = pd.read_csv("data/processed/splits/validation.csv")

x_train = train_data["combined_text"].fillna("")
y_train = train_data["label"].astype(int)

x_validation = validation_data["combined_text"].fillna("")
y_validation = validation_data["label"].astype(int)

print("Training records:", len(x_train))
print("Validation records:", len(x_validation))

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

x_train = vectorizer.transform(x_train)
x_validation = vectorizer.transform(x_validation)

models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}

results = []

for name,model in models.items():

    print("\n" + "=" * 50)
    print("Model:",name)

    start_time = time.time()

    model.fit(x_train,y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(x_validation)

    accuracy = accuracy_score(y_validation,y_pred)

    precision = precision_score(y_validation,y_pred,zero_division=0)

    recall = recall_score(
        y_validation, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_validation, y_pred, zero_division=0
    )

    print("Training time:", round(training_time, 2), "seconds")

    print("\nClassification Report:")
    print(
        classification_report(
            y_validation,
            y_pred,
            labels=[0, 1],
            target_names=["Fake", "Real"],
            zero_division=0
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Training Time (seconds)": training_time
    })

# Display comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
)

print("\n\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False, float_format="%.4f"))


# Save comparison results
results_df.to_csv(
    "models/model_comparison_results.csv",
    index=False
)

print("\nComparison results saved to:")
print("models/model_comparison_results.csv")
