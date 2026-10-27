# TruthLens — Fake News Text Classification

TruthLens is a machine learning application that classifies news articles as **Fake** or **Real** based on their textual content. It combines traditional machine learning and deep learning models with an interactive Streamlit dashboard for predictions, model comparison, and explainability.

**GitHub Repository:** https://github.com/Mole1614x/TruthLens

> **Important:** TruthLens is a text-classification system, not an independent fact-checking service. Its predictions are based on patterns learned from the training dataset and do not prove whether an article or claim is factually true.

## Features

- Interactive Streamlit dashboard
- News classification using six trained models
- Model performance comparison on a held-out test set
- LIME-based explanations for individual predictions
- Model insights and discussion of limitations
- Organized scripts for data auditing, preparation, model building, training, evaluation, analysis, and prediction

## Models

TruthLens evaluates the following models:

1. Artificial Neural Network (ANN)
2. Long Short-Term Memory (LSTM)
3. Bidirectional LSTM (BiLSTM)
4. Convolutional Neural Network (CNN)
5. DistilBERT
6. Linear Support Vector Machine (Linear SVM)

## Dataset

The project uses the WELFake dataset for fake-news text classification.

The prepared dataset was divided into the following splits:

| Split | Number of articles |
|---|---:|
| Training | 57,708 |
| Validation | 7,213 |
| Test | 7,213 |
| **Total** | **72,134** |

Target labels:

- `0` — Fake
- `1` — Real

The raw dataset and processed data files are excluded from version control because of their size.

## Test Results

The following results were obtained on the held-out test set of 7,213 articles.

| Model | Test Accuracy | Correct Predictions | Incorrect Predictions |
|---|---:|---:|---:|
| ANN | 96.56% | 6,965 | 248 |
| LSTM | 91.64% | 6,610 | 603 |
| BiLSTM | 97.52% | 7,034 | 179 |
| CNN | 97.99% | 7,068 | 145 |
| DistilBERT | 99.25% | 7,159 | 54 |
| Linear SVM | 95.47% | 6,886 | 327 |

DistilBERT achieved the highest test accuracy among the evaluated models. These results describe performance on this dataset and do not guarantee equivalent accuracy on new, real-world news.

## Technology Stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Deep learning:** TensorFlow, Keras, Keras Hub
- **Machine learning:** scikit-learn
- **Data processing:** NumPy, pandas, SciPy
- **Explainability:** LIME

## Project Structure

```text
TruthLens/
├── app.py
├── requirements.txt
├── .gitignore
├── data/
├── models/
├── reports/
├── scripts/
│   ├── analysis/
│   ├── data_audit/
│   ├── data_preparation/
│   ├── model_building/
│   ├── model_evaluation/
│   ├── model_training/
│   └── prediction/
└── src/
    └── preprocessing/
        └── text_cleaner.py
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Mole1614x/TruthLens.git
cd TruthLens
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Provide the required model artifacts

The trained models, vectorizers, and large datasets are not included in the Git repository. The application requires the model files and supporting artifacts at their expected paths.

Make these artifacts available locally before launching the dashboard.

### 5. Run the dashboard

```bash
streamlit run app.py
```

Streamlit will display a local URL for opening the application in your browser.

## Reports and Analysis

The `reports/` directory contains evaluation reports, test summaries, model comparison results, and LIME explanation outputs.

The `scripts/` directory organizes utilities for:

- Data auditing and duplicate analysis
- Data preparation and splitting
- Model building and training
- Model evaluation
- Error analysis and model comparison
- Prediction

## Limitations

- Predictions reflect patterns learned from the training data, not independent verification of facts.
- Performance may vary on unfamiliar topics, new sources, or changing writing styles.
- Dataset biases and misleading correlations may affect predictions.
- LIME highlights text features that influence a model's output; it does not establish that the output is correct.
- Predictions should not be the sole basis for judging whether a news article is trustworthy.

## Project Status

Model development, evaluation, comparison, and the Streamlit dashboard have been completed. Repository documentation and publication preparation are in progress.
