from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_DIR / "reports"

comparison = [
    {
        "model": "DistilBERT",
        "test_samples": 7213,
        "accuracy": 0.992514,
        "weighted_f1": 0.9925,
        "correct_predictions": 7159,
        "incorrect_predictions": 54,
    },
    {
        "model": "CNN",
        "test_samples": 7213,
        "accuracy": 0.979897,
        "weighted_f1": 0.9799,
        "correct_predictions": 7068,
        "incorrect_predictions": 145,
    },
    {
        "model": "BiLSTM",
        "test_samples": 7213,
        "accuracy": 0.975184,
        "weighted_f1": 0.9752,
        "correct_predictions": 7034,
        "incorrect_predictions": 179,
    },
    {
        "model": "ANN",
        "test_samples": 7213,
        "accuracy": 0.965618,
        "weighted_f1": 0.9656,
        "correct_predictions": 6965,
        "incorrect_predictions": 248,
    },
    {
        "model": "Tuned Linear SVM",
        "test_samples": 7213,
        "accuracy": 0.954665,
        "weighted_f1": 0.9547,
        "correct_predictions": 6886,
        "incorrect_predictions": 327,
    },
    {
        "model": "LSTM",
        "test_samples": 7213,
        "accuracy": 0.916401,
        "weighted_f1": 0.9164,
        "correct_predictions": 6610,
        "incorrect_predictions": 603,
    },
]

df = pd.DataFrame(comparison)

df = df.sort_values(
    by="accuracy",
    ascending=False
).reset_index(drop=True)

df.insert(0, "rank", range(1, len(df) + 1))

REPORT_DIR.mkdir(parents=True, exist_ok=True)

output_path = REPORT_DIR / "final_model_comparison.csv"

df.to_csv(output_path, index=False)

print("\n========== FINAL MODEL COMPARISON ==========")
print(df.to_string(index=False))

print("\nSaved comparison report to:")
print(output_path)
