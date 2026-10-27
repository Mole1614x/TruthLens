
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf
import keras_hub
from lime.lime_text import LimeTextExplainer
from scipy.special import expit


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MISCLASSIFIED_PATH = MODELS_DIR / "misclassified_articles.csv"

PRESET = "distil_bert_base_en_uncased"
MAX_LENGTH = 256
SEQUENCE_LENGTH = 300

LABELS = {
    0: "Fake News",
    1: "Real News",
}

MODEL_OPTIONS = [
    "DistilBERT",
    "ANN",
    "LSTM",
    "BiLSTM",
    "CNN",
    "Linear SVM",
]
DEFAULT_MODEL = "DistilBERT"

MODEL_PATHS = {
    "DistilBERT": MODELS_DIR / "truthlens_distilbert.keras",
    "ANN": MODELS_DIR / "truthlens_ann.keras",
    "LSTM": MODELS_DIR / "truthlens_lstm.keras",
    "BiLSTM": MODELS_DIR / "truthlens_bilstm_best.keras",
    "CNN": MODELS_DIR / "truthlens_cnn_best.keras",
    "Linear SVM": MODELS_DIR / "tuned_linear_svm.pkl",
}
VECTORIZER_PATH = MODELS_DIR / "ann_text_vectorizer.keras"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

MODEL_DESCRIPTIONS = {
    "DistilBERT": "Transformer model using its KerasHub tokenizer and attention mask.",
    "ANN": "Dense neural network using the saved 300-token text vectorizer.",
    "LSTM": "LSTM sequence model using the shared saved 300-token text vectorizer.",
    "BiLSTM": "Bidirectional LSTM using the shared saved 300-token text vectorizer.",
    "CNN": "CNN text classifier using the shared saved 300-token text vectorizer.",
    "Linear SVM": "Linear support vector machine using the saved TF-IDF vectorizer.",
}

PAGES = [
    "Overview",
    "News Analyzer",
    "Model Insights",
    "About & Limitations",
]


# =========================================================
# 2. CYBERSECURITY THEME
# =========================================================

st.set_page_config(
    page_title="TruthLens | Intelligence Console",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg: #080b12;
        --panel: #101622;
        --panel2: #141d2b;
        --border: #263448;
        --cyan: #38e8ff;
        --purple: #a78bfa;
        --green: #4ade80;
        --red: #fb7185;
        --text: #edf4ff;
        --muted: #91a2bb;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, #172039 0%, transparent 38%),
            var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #0b101a;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: var(--text) !important;
        letter-spacing: -0.035em;
    }

    p, label, li {
        color: #c2cee0;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #141d2b, #0f1520);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 18px 20px;
        min-height: 112px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--cyan) !important;
        font-family: 'JetBrains Mono', monospace;
    }

    [data-testid="stMetricDelta"] {
        color: var(--green) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border) !important;
        border-radius: 13px;
    }

    .hero {
        background: linear-gradient(115deg, #141e30, #111522 65%, #1b1731);
        border: 1px solid #33445d;
        border-radius: 18px;
        padding: 30px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .eyebrow {
        color: var(--cyan);
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: clamp(28px, 4vw, 43px);
        font-weight: 800;
        color: #f4f8ff;
        line-height: 1.12;
        margin: 0 0 12px 0;
    }

    .hero-description {
        color: #aebdd2;
        font-size: 15px;
        max-width: 760px;
        line-height: 1.7;
    }

    .section-label {
        color: var(--cyan);
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 11px;
        margin: 22px 0 10px 0;
    }

    .panel {
        background: linear-gradient(145deg, #131b29, #0e1420);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .panel-title {
        color: #e9f1ff;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .muted {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.65;
    }

    .status-pill {
        display: inline-block;
        border: 1px solid #276b58;
        background: #102b26;
        color: var(--green);
        border-radius: 20px;
        padding: 6px 11px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
    }

    .cyan-text { color: var(--cyan); }
    .purple-text { color: var(--purple); }
    .green-text { color: var(--green); }
    .red-text { color: var(--red); }

    .stButton > button {
        border-radius: 9px;
        border: 1px solid #277d93;
        background: linear-gradient(100deg, #123447, #182647);
        color: #eafcff;
        font-weight: 700;
        min-height: 44px;
    }

    .stButton > button:hover {
        border-color: var(--cyan);
        color: white;
        background: #18374a;
    }

    .stTextArea textarea {
        background: #0b111c;
        color: #e7f1ff;
        border: 1px solid #304057;
        border-radius: 10px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    hr {
        border-color: var(--border);
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD MODEL RESOURCES
# =========================================================

@st.cache_resource
def load_model_resources(model_name):
    """Load the chosen trained model and its matching preprocessing assets."""
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model selection: {model_name}")

    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_name} model not found: {model_path}"
        )

    resources = {"name": model_name, "model": None}

    if model_name == "DistilBERT":
        tokenizer = keras_hub.models.DistilBertTokenizer.from_preset(PRESET)
        model = tf.keras.models.load_model(model_path, compile=False)
        resources.update({
            "model": model,
            "tokenizer": tokenizer,
            "cls_id": tokenizer.token_to_id("[CLS]"),
            "sep_id": tokenizer.token_to_id("[SEP]"),
            "pad_id": tokenizer.token_to_id("[PAD]"),
        })

    elif model_name in {"ANN", "LSTM", "BiLSTM", "CNN"}:
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"Shared text vectorizer not found: {VECTORIZER_PATH}"
            )
        model = tf.keras.models.load_model(model_path, compile=False)
        vectorizer_model = tf.keras.models.load_model(
            VECTORIZER_PATH, compile=False
        )
        resources.update({
            "model": model,
            "vectorizer": vectorizer_model.layers[0],
        })

    elif model_name == "Linear SVM":
        if not TFIDF_VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found: {TFIDF_VECTORIZER_PATH}"
            )
        resources.update({
            "model": joblib.load(model_path),
            "vectorizer": joblib.load(TFIDF_VECTORIZER_PATH),
        })

    resources["explainer"] = LimeTextExplainer(
        class_names=["Fake News", "Real News"],
        random_state=42,
    )
    return resources


# =========================================================
# 4. TEXT PREPARATION AND PREDICTION
# =========================================================

def prepare_distilbert_text(news_text, resources):
    tokenizer = resources["tokenizer"]
    tokens = tokenizer.tokenize(news_text)
    if hasattr(tokens, "numpy"):
        tokens = tokens.numpy()

    token_ids = np.asarray(tokens).reshape(-1).tolist()
    token_ids = token_ids[:MAX_LENGTH - 2]
    token_ids = (
        [resources["cls_id"]]
        + token_ids
        + [resources["sep_id"]]
    )
    attention_mask = [1] * len(token_ids)

    padding_length = MAX_LENGTH - len(token_ids)
    token_ids += [resources["pad_id"]] * padding_length
    attention_mask += [0] * padding_length

    return (
        np.asarray([token_ids], dtype=np.int32),
        np.asarray([attention_mask], dtype=np.int32),
    )


def predict_probabilities(texts, resources):
    """Return two class scores in [Fake, Real] order for the selected model.

    Neural model outputs are the model's class scores. Linear SVM uses a
    sigmoid-transformed decision margin solely to provide LIME with a two-column
    score matrix; those SVM values are not calibrated probabilities.
    """
    model_name = resources["name"]
    model = resources["model"]
    all_scores = []

    for raw_text in texts:
        text = str(raw_text)

        if model_name == "DistilBERT":
            input_ids, attention_mask = prepare_distilbert_text(text, resources)
            output = model.predict(
                {"token_ids": input_ids, "padding_mask": attention_mask},
                verbose=0,
            )[0]
            scores = np.asarray(output, dtype=float).reshape(-1)
            if len(scores) != 2:
                raise ValueError(
                    "DistilBERT is expected to return two class scores."
                )

        elif model_name in {"ANN", "LSTM", "BiLSTM", "CNN"}:
            token_ids = resources["vectorizer"](
                tf.constant([text]), training=False
            )
            real_score = float(model.predict(token_ids, verbose=0).reshape(-1)[0])
            real_score = float(np.clip(real_score, 0.0, 1.0))
            scores = np.asarray([1.0 - real_score, real_score])

        elif model_name == "Linear SVM":
            features = resources["vectorizer"].transform([text])
            decision = float(np.asarray(model.decision_function(features)).reshape(-1)[0])
            real_score = float(expit(decision))
            scores = np.asarray([1.0 - real_score, real_score])

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # LIME expects each row to be a two-class score distribution.
        # Normalize transformer outputs if they are logits rather than probabilities.
        scores = np.asarray(scores, dtype=float)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        scores = np.clip(scores, 0.0, None)
        total = float(scores.sum())
        if total <= 0:
            scores = np.asarray([0.5, 0.5])
        else:
            scores = scores / total
        all_scores.append(scores)

    return np.asarray(all_scores)


# =========================================================
# 5. LOAD PROJECT REPORTS
# =========================================================

@st.cache_data
def load_comparison():
    path = REPORTS_DIR / "final_model_comparison.csv"

    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()

    # Add LSTM from its separate saved summary if missing.
    lstm_path = REPORTS_DIR / "lstm_test_summary.csv"

    if lstm_path.exists() and not df.empty:
        lstm = pd.read_csv(lstm_path)

        if "model" in lstm.columns and "model" in df.columns:
            if not df["model"].astype(str).str.contains(
                "LSTM", case=False, regex=False
            ).any():
                df = pd.concat([df, lstm], ignore_index=True)

    if not df.empty and "accuracy" in df.columns:
        df = df.sort_values("accuracy", ascending=False)

    return df


@st.cache_data
def load_class_report():
    path = REPORTS_DIR / "distilbert_test_report.csv"

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Class"})

    return df


@st.cache_data
def load_misclassified():
    if not MISCLASSIFIED_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(MISCLASSIFIED_PATH)
    except Exception:
        return pd.DataFrame()


def percent(value):
    return f"{float(value) * 100:.2f}%"


def render_header(title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">TRUTHLENS / INTELLIGENCE CONSOLE</div>
            <div class="hero-title">{title}</div>
            <div class="hero-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    st.caption(
        "Responsible use: TruthLens classifies learned text patterns. "
        "Its scores are not calibrated probabilities of factual truth, "
        "and its predictions do not independently verify news claims."
    )


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 0 22px 0">
            <div style="font-size:30px">◈ <span style="color:#38e8ff">TL</span></div>
            <div style="font-size:22px;font-weight:800;color:#f0f6ff">
                TRUTHLENS
            </div>
            <div style="font-family:monospace;font-size:10px;
                        letter-spacing:2px;color:#91a2bb">
                NEWS INTELLIGENCE SYSTEM
            </div>
        </div>
        <div style="height:1px;background:#263448;margin-bottom:18px"></div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        PAGES,
        key="active_page",
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown("#### SYSTEM STATUS")

    st.markdown(
        """
        <div class="panel">
            <span class="status-pill">● MODEL PIPELINE</span>
            <p class="muted" style="margin-top:12px">
                Six selectable text classifiers<br>
                LIME local explanations<br>
                WELFake evaluation dataset
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("TRUTHLENS · DEEP LEARNING PROJECT")
    st.caption("Model scores are not fact verification.")


# =========================================================
# 7. LOAD MODEL WHEN NEEDED
# =========================================================

model_resources = None

# Overview only needs saved evaluation reports; the selected model is loaded
# lazily inside News Analyzer when the user runs an analysis.
if page == "Overview":
    try:
        model_resources = load_model_resources(DEFAULT_MODEL)
    except Exception as error:
        st.info(
            "The default model could not be loaded. Saved evaluation reports "
            f"can still be displayed. Details: {error}"
        )


# =========================================================
# 8. OVERVIEW
# =========================================================

if page == "Overview":
    render_header(
        "News Intelligence Overview",
        "Monitor model performance, compare trained architectures, "
        "and inspect the capabilities of the TruthLens text classifier.",
    )

    comparison = load_comparison()

    if comparison.empty:
        st.warning(
            "Model comparison report not found. Expected: "
            "reports/final_model_comparison.csv"
        )
    else:
        best = comparison.iloc[0]

        accuracy = float(best["accuracy"])
        test_samples = int(best.get("test_samples", 0))
        correct = int(best.get("correct_predictions", 0))
        incorrect = int(best.get("incorrect_predictions", 0))

        st.markdown(
            '<div class="section-label">SYSTEM PERFORMANCE</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Best Model", str(best["model"]))
        c2.metric("Test Accuracy", percent(accuracy))
        c3.metric("Correct Predictions", f"{correct:,}")
        c4.metric("Incorrect Predictions", f"{incorrect:,}")

        st.markdown(
            '<div class="section-label">MODEL MONITORING</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.5, 1])

        with left:
            st.markdown(
                '<div class="panel-title">Architecture Performance</div>',
                unsafe_allow_html=True,
            )

            chart_data = comparison[["model", "accuracy"]].copy()
            chart_data["accuracy"] = chart_data["accuracy"] * 100
            chart_data = chart_data.set_index("model")

            st.bar_chart(
                chart_data,
                y="accuracy",
                height=320,
                color="#38e8ff",
            )

            st.caption("Metric: test-set accuracy (%)")

        with right:
            st.markdown(
                '<div class="panel-title">Evaluation Snapshot</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="panel">
                    <div class="eyebrow">TEST PARTITION</div>
                    <div style="font-size:34px;font-weight:800;
                                color:#38e8ff">{test_samples:,}</div>
                    <div class="muted">Held-out test samples</div>
                    <hr>
                    <div class="eyebrow">TOP MODEL</div>
                    <div style="font-size:23px;font-weight:700;
                                color:#a78bfa">{best["model"]}</div>
                    <div class="muted">Ranked by saved test accuracy</div>
                    <hr>
                    <div class="eyebrow">PIPELINE</div>
                    <div style="font-size:15px;color:#4ade80">
                        DistilBERT + LIME
                    </div>
                    <div class="muted">
                        Classification with local explanations
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="section-label">MODEL LEADERBOARD</div>',
            unsafe_allow_html=True,
        )

        display = comparison.copy()

        for col in ["accuracy", "weighted_f1"]:
            if col in display.columns:
                display[col] = display[col].map(
                    lambda x: f"{float(x) * 100:.2f}%"
                )

        st.dataframe(display, hide_index=True, width="stretch")

    render_disclaimer()


# =========================================================
# 9. NEWS ANALYZER
# =========================================================

elif page == "News Analyzer":
    render_header(
        "News Analysis Workspace",
        "Choose a trained model, submit a headline or article, and inspect its "
        "predicted class, model scores, text characteristics, and LIME explanation.",
    )

    st.markdown(
        '<div class="section-label">MODEL SELECTION</div>',
        unsafe_allow_html=True,
    )

    with st.form("news_analysis_form"):
        selected_model = st.selectbox(
            "Choose classification model",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(DEFAULT_MODEL),
            key="selected_model",
            help="Each model uses the preprocessing assets saved for its training pipeline.",
        )
        st.caption(MODEL_DESCRIPTIONS[selected_model])

        news_text = st.text_area(
            "Paste news text",
            height=220,
            placeholder=(
                "Paste a news headline or article here for text "
                "classification..."
            ),
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "▶  RUN TEXT ANALYSIS",
            width="stretch",
        )

    if submitted:
        if not news_text.strip():
            st.warning("Enter some news text before running analysis.")
        else:
            try:
                with st.spinner(f"Running {selected_model} inference..."):
                    selected_resources = load_model_resources(selected_model)
                    scores = predict_probabilities(
                        [news_text], selected_resources
                    )[0]

                predicted_class = int(np.argmax(scores))
                st.session_state["last_news_text"] = news_text
                st.session_state["last_scores"] = scores.tolist()
                st.session_state["last_prediction"] = predicted_class
                st.session_state["last_model"] = selected_model
                st.session_state["last_lime"] = None
            except Exception as error:
                st.error(f"Could not run {selected_model}: {error}")

    # Show the result only while the dropdown still matches the model that
    # produced it, so changing the dropdown cannot mislabel a previous result.
    if (
        "last_scores" in st.session_state
        and st.session_state.get("last_model") == selected_model
    ):
        scores = np.asarray(st.session_state["last_scores"], dtype=float)
        predicted_class = int(st.session_state["last_prediction"])
        saved_text = st.session_state["last_news_text"]
        result_model = st.session_state["last_model"]

        st.markdown(
            '<div class="section-label">CLASSIFICATION OUTPUT</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Model used for this result: **{result_model}**")

        result_col, fake_col, real_col = st.columns([1.2, 1, 1])

        with result_col:
            result_color = "#fb7185" if predicted_class == 0 else "#4ade80"
            st.markdown(
                f"""
                <div class="panel">
                    <div class="eyebrow">PREDICTED CLASS</div>
                    <div style="font-size:25px;font-weight:800;color:{result_color}">
                        {LABELS[predicted_class]}
                    </div>
                    <div class="muted">Based on learned text patterns</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with fake_col:
            st.metric("Fake News Score", percent(scores[0]))
            st.progress(float(np.clip(scores[0], 0.0, 1.0)))

        with real_col:
            st.metric("Real News Score", percent(scores[1]))
            st.progress(float(np.clip(scores[1], 0.0, 1.0)))

        if result_model == "Linear SVM":
            st.info(
                "Linear SVM scores are sigmoid-transformed decision margins for "
                "display and explanation. They are not calibrated probabilities."
            )

        word_count = len(saved_text.split())
        char_count = len(saved_text)
        sentence_count = max(1, sum(saved_text.count(mark) for mark in ".!?"))

        st.markdown(
            '<div class="section-label">TEXT PROFILE</div>',
            unsafe_allow_html=True,
        )
        t1, t2, t3 = st.columns(3)
        t1.metric("Words", f"{word_count:,}")
        t2.metric("Characters", f"{char_count:,}")
        t3.metric("Sentence Endings", f"{sentence_count:,}")

        st.markdown(
            '<div class="section-label">EXPLAINABILITY / LIME</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Local word contribution</div>
                <div class="muted">
                    LIME perturbs the submitted text to estimate which words
                    influence the selected model class for this particular input.
                    These are local explanations, not evidence that the claim is true or false.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        explanation_class = st.selectbox(
            "Class to explain",
            options=[predicted_class, 1 - predicted_class],
            format_func=lambda x: LABELS[x],
            key="lime_explanation_class",
        )

        if st.button("Generate LIME Explanation", width="stretch"):
            try:
                with st.spinner(
                    f"Generating {result_model} word-level explanation..."
                ):
                    explanation_resources = load_model_resources(result_model)

                    def lime_predict(texts):
                        return predict_probabilities(texts, explanation_resources)

                    explanation = explanation_resources["explainer"].explain_instance(
                        saved_text,
                        lime_predict,
                        labels=[0, 1],
                        num_features=10,
                        num_samples=100,
                    )
                    st.session_state["last_lime"] = {
                        "model": result_model,
                        "class": explanation_class,
                        "items": explanation.as_list(label=explanation_class),
                    }
            except Exception as error:
                st.error(f"LIME explanation failed: {error}")

        lime_result = st.session_state.get("last_lime")
        if (
            lime_result
            and lime_result.get("model") == result_model
            and lime_result.get("class") == explanation_class
        ):
            items = lime_result["items"]
            if items:
                lime_df = pd.DataFrame(items, columns=["Word", "Weight"])
                st.markdown(f"**Explaining: {LABELS[explanation_class]}**")
                st.bar_chart(
                    lime_df.set_index("Word"),
                    y="Weight",
                    color="#a78bfa",
                    height=320,
                )

                for word, weight in items:
                    if weight > 0:
                        direction = "Supports selected class"
                    elif weight < 0:
                        direction = "Opposes selected class"
                    else:
                        direction = "Little local influence"
                    st.write(f"**{word}** — {weight:+.6f} · {direction}")

                st.caption(
                    "Positive weights support the selected class; negative weights "
                    "oppose it. Weight magnitudes are local explanation values, not probabilities."
                )
            else:
                st.info("LIME did not return word contributions.")
    elif "last_scores" in st.session_state:
        st.info(
            f"Your previous result was produced by {st.session_state.get('last_model', 'another model')}. "
            "Run the analysis again to get a result from the currently selected model."
        )

    render_disclaimer()


# =========================================================
# 10. MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":
    render_header(
        "Model Diagnostics",
        "Compare trained architectures and inspect class-level metrics "
        "and saved misclassification examples from the evaluation workflow.",
    )

    comparison = load_comparison()

    if comparison.empty:
        st.warning("The model comparison report could not be loaded.")
    else:
        st.markdown(
            '<div class="section-label">COMPARATIVE EVALUATION</div>',
            unsafe_allow_html=True,
        )

        chart_df = comparison[["model", "accuracy", "weighted_f1"]].copy()
        chart_df = chart_df.set_index("model") * 100

        st.bar_chart(
            chart_df,
            height=350,
            color=["#38e8ff", "#a78bfa"],
        )

        st.caption("Comparison metrics are shown as percentages.")

        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
        )

    st.markdown(
        '<div class="section-label">DISTILBERT CLASS METRICS</div>',
        unsafe_allow_html=True,
    )

    class_report = load_class_report()

    if class_report.empty:
        st.info("Class report not found.")
    else:
        st.dataframe(
            class_report,
            hide_index=True,
            width="stretch",
        )

    st.markdown(
        '<div class="section-label">DISTILBERT CONFUSION MATRIX</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="muted">
            Rows represent actual classes; columns represent predicted
            classes. This matrix uses the saved test-set evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cm = pd.DataFrame(
        [[3461, 42], [12, 3698]],
        index=["Actual Fake", "Actual Real"],
        columns=["Predicted Fake", "Predicted Real"],
    )

    st.dataframe(cm, width="stretch")

    st.caption(
        "Matrix layout: actual Fake → 3,461 Fake and 42 Real predictions; "
        "actual Real → 12 Fake and 3,698 Real predictions."
    )

    st.markdown(
        '<div class="section-label">MISCLASSIFICATION REVIEW</div>',
        unsafe_allow_html=True,
    )

    mistakes = load_misclassified()

    if mistakes.empty:
        st.info(
            "No readable misclassification file was found at "
            "models/misclassified_articles.csv."
        )
    else:
        st.write(
            "These examples come from the saved project file. "
            "Check its columns to identify actual and predicted labels."
        )

        st.dataframe(
            mistakes.head(100),
            width="stretch",
        )

        st.caption(
            "Showing at most 100 rows to keep the dashboard responsive."
        )

    render_disclaimer()


# =========================================================
# 11. ABOUT AND LIMITATIONS
# =========================================================

elif page == "About & Limitations":
    render_header(
        "System Information",
        "Project architecture, evaluation context, and responsible-use guidance.",
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="eyebrow">PROJECT</div>
                <div class="panel-title">TruthLens</div>
                <p class="muted">
                    A deep-learning text classification application
                    trained to classify news text into Fake News
                    and Real News categories.
                </p>
                <hr>
                <div class="panel-title">Core Components</div>
                <p class="muted">
                    • DistilBERT, ANN, LSTM, BiLSTM and CNN<br>
                    • Linear SVM classifier<br>
                    • Keras and TensorFlow inference<br>
                    • KerasHub tokenizer<br>
                    • LIME local interpretability<br>
                    • Streamlit dashboard<br>
                    • Saved CSV evaluation reports
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="eyebrow">EVALUATION</div>
                <div class="panel-title">Held-out test set</div>
                <p class="muted">
                    Model comparison uses the saved evaluation results
                    from the project's test split. The dashboard does
                    not train or fine-tune models.
                </p>
                <hr>
                <div class="panel-title">Responsible Use</div>
                <p class="muted">
                    Predictions may be wrong. A plausible article may
                    be classified as fake, and fabricated text may be
                    classified as real. Use independent sources to
                    verify claims.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-label">IMPORTANT LIMITATIONS</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "TruthLens is a learned text classifier, not an independent "
        "fact-checking system. It does not search for corroborating "
        "sources or establish the truth of an event."
    )

    st.markdown(
        """
        **Keep in mind:**

        1. Model scores reflect the classifier's output for the input.
        2. High scores do not guarantee factual correctness.
        3. Training data can contain historical, source, or topic biases.
        4. LIME describes local model behavior, not causal proof.
        5. Important claims should be checked against reliable,
           independent reporting and primary sources.
        """
    )

    st.markdown("---")
    st.caption(
        "TruthLens · Deep Learning and Explainable AI · Six models + LIME"
    )
