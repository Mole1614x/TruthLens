
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

REPORT_DIR = Path("reports")

MODELS = [
    "ann",
    "lstm",
    "bilstm",
    "cnn",
    "distilbert"
]


# --------------------------------------------------
# 2. Load and interpret model reports
# --------------------------------------------------

for model_name in MODELS:

    report_path = REPORT_DIR / f"{model_name}_test_report.csv"

    if not report_path.exists():
        print(f"Report not found: {report_path}")
        continue

    report = pd.read_csv(
        report_path,
        index_col=0
    )

    # Remove extra spaces from class names.
    report.index = report.index.astype(str).str.strip()

    print("\n" + "=" * 50)
    print(f"MODEL: {model_name.upper()}")
    print("=" * 50)

    print("\nClassification Report:")
    print(report)

    # Support both report naming conventions.
    if "Fake News" in report.index:
        fake_label = "Fake News"
        real_label = "Real News"
    else:
        fake_label = "Fake"
        real_label = "Real"

    fake_recall = report.loc[
        fake_label,
        "recall"
    ]

    real_recall = report.loc[
        real_label,
        "recall"
    ]

    print("\nInterpretation:")

    print(
        f"Fake News Recall: {fake_recall:.2%}"
    )

    print(
        f"Real News Recall: {real_recall:.2%}"
    )

    if fake_recall > real_recall:

        print(
            "The model identifies Fake News "
            "more effectively."
        )

    elif real_recall > fake_recall:

        print(
            "The model identifies Real News "
            "more effectively."
        )

    else:

        print(
            "The model has equal recall "
            "for both classes."
        )
