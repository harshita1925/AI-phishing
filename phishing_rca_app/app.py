"""
app.py

AI-BASED PHISHING DETECTION & ROOT CAUSE ANALYSIS ASSISTANT
==============================================================
A small AI prototype built for CIA-1 Part 5, implementing the core
pipeline of the paper "AI-Based Phishing Detection Using Machine
Learning and Root Cause Analysis":

    - a declarative KNOWLEDGE BASE of phishing indicators
      (feature_schema.py),
    - a synthetic dataset matching the UCI Phishing Websites schema
      (dataset.py),
    - a LEARNING ENGINE that trains Logistic Regression (baseline) and
      Random Forest (main model), evaluates accuracy, and performs
      Root Cause Analysis via feature importance (model_engine.py),
      and
    - this Streamlit UI, which trains the model live, reports its
      performance, and lets you classify a website (preset or custom)
      with a full Explainable-AI breakdown of WHY it was flagged.

IMPORTANT: This is an educational PROTOTYPE trained on a synthetic
dataset (see dataset.py for why), not the full 11,055-row UCI dataset
used in the paper. The pipeline, model choice, and Root Cause Analysis
method are otherwise identical to what the paper describes.
"""

import pandas as pd
import streamlit as st

from feature_schema import FEATURES, FEATURE_QUESTIONS, get_all_features_summary
from dataset import load_dataset, EXAMPLE_CASES
from model_engine import train_models, explain_prediction

# -----------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Phishing Detection & Root Cause Analysis",
    page_icon="\U0001F3A3",  # fishing pole
    layout="wide",
)

# -----------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------
if "training" not in st.session_state:
    st.session_state.training = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "analyzed_sample" not in st.session_state:
    st.session_state.analyzed_sample = None
if "is_demo" not in st.session_state:
    st.session_state.is_demo = False


@st.cache_resource(show_spinner=False)
def get_trained_models():
    """Train once per session and cache -- mirrors the paper's one-time
    training step; re-running the app doesn't retrain from scratch."""
    df = load_dataset()
    return train_models(df)


def run_analysis(sample: dict) -> None:
    training = st.session_state.training
    explanation = explain_prediction(
        training["rf_model"], training["feature_importances"], sample
    )
    st.session_state.explanation = explanation
    st.session_state.analyzed_sample = sample


def reset_app() -> None:
    st.session_state.explanation = None
    st.session_state.analyzed_sample = None
    st.session_state.is_demo = False


# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------
st.title("AI Phishing Detection & Root Cause Analysis")
st.caption("A Machine-Learning Prototype with Explainable AI, based on the CTI research paper")

st.info(
    "This prototype trains a Random Forest phishing classifier and explains "
    "**why** each website was flagged, using the same 10 indicators "
    "identified in the paper's Root Cause Analysis (Figure 2): "
    "**SSLfinal_State** and **URL_of_Anchor** are the strongest signals."
)

st.divider()

# -----------------------------------------------------------------------
# TRAIN MODELS (cached)
# -----------------------------------------------------------------------
with st.spinner("Training Logistic Regression + Random Forest..."):
    st.session_state.training = get_trained_models()
training = st.session_state.training

# -----------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------
st.sidebar.header("Analyze a Website")

mode = st.sidebar.radio(
    "Choose input mode:",
    ["Preset example", "Custom feature values"],
)

if mode == "Preset example":
    case_keys = list(EXAMPLE_CASES.keys())
    case_labels = [EXAMPLE_CASES[k]["label"] for k in case_keys]
    chosen = st.sidebar.selectbox("Pick a preset website:", options=case_labels)
    chosen_key = case_keys[case_labels.index(chosen)]
    current_sample = EXAMPLE_CASES[chosen_key]["features"]
else:
    st.sidebar.caption("For each indicator: -1 = looks legitimate, 0 = neutral, 1 = suspicious/phishing")
    current_sample = {}
    for f in FEATURES:
        current_sample[f] = st.sidebar.select_slider(
            FEATURE_QUESTIONS[f].replace("?", ""), options=[-1, 0, 1], value=-1, key=f"slider_{f}"
        )

col_a, col_b = st.sidebar.columns(2)
analyze_clicked = col_a.button("Analyze", use_container_width=True)
reset_clicked = col_b.button("Reset", use_container_width=True)

st.sidebar.divider()
demo_clicked = st.sidebar.button("\u25B6 Run Demo Mode (Phishing Example)", use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown(
    "**PEAS Description**\n\n"
    "- **Performance measure:** correct, explainable phishing/legitimate verdict\n"
    "- **Environment:** website feature values (SSL, anchors, traffic, etc.)\n"
    "- **Actuators:** displaying the verdict, confidence, and root causes\n"
    "- **Sensors:** the feature values selected/entered by the user"
)

if reset_clicked:
    reset_app()

if analyze_clicked:
    run_analysis(current_sample)
    st.session_state.is_demo = False

if demo_clicked:
    run_analysis(EXAMPLE_CASES["phishing_example"]["features"])
    st.session_state.is_demo = True

# -----------------------------------------------------------------------
# MAIN AREA
# -----------------------------------------------------------------------

# ---- Section 1: Problem statement --------------------------------------
st.header("1. Problem Statement")
st.write(
    "Traditional blacklists cannot keep up with thousands of new phishing URLs "
    "created daily, and most AI phishing detectors act as a 'black box' -- they "
    "predict a verdict but cannot explain it. This prototype implements the "
    "paper's solution: a high-accuracy Random Forest classifier paired with an "
    "automated Root Cause Analysis module, so every alert comes with a "
    "plain-English explanation a security analyst can trust."
)

# ---- Section 2: Dataset & knowledge base --------------------------------
st.header("2. Dataset & Knowledge Base")
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown(f"**Training samples:** {training['n_train']}  \n**Testing samples:** {training['n_test']}")
    st.markdown("Same 80/20 split and -1/0/1 feature encoding as the paper's UCI-based methodology.")
with col2:
    with st.expander("View the Knowledge Base (feature meanings)"):
        for line in get_all_features_summary():
            st.markdown(f"- {line}")

# ---- Section 3: Model training & evaluation -----------------------------
st.header("3. Model Training & Evaluation")
c1, c2 = st.columns(2)
c1.metric("Logistic Regression Accuracy", f"{training['lr_accuracy']*100:.2f}%")
c2.metric("Random Forest Accuracy", f"{training['rf_accuracy']*100:.2f}%")

cm = training["confusion_matrix"]
cm_df = pd.DataFrame(
    cm,
    index=["Actual: Legitimate (-1)", "Actual: Phishing (1)"],
    columns=["Predicted: Legitimate (-1)", "Predicted: Phishing (1)"],
)
st.markdown("**Confusion Matrix (Random Forest)**")
st.dataframe(cm_df, use_container_width=True)

# ---- Section 4: Root Cause Analysis (feature importance) ---------------
st.header("4. Root Cause Analysis (Feature Importance)")
st.write(
    "Extracted directly from the trained Random Forest -- the same technique "
    "used in the paper (Section III-D) to identify which indicators drive "
    "phishing predictions."
)
importances = pd.Series(training["feature_importances"]).sort_values(ascending=True)
st.bar_chart(importances)

# ---- Section 5: Live analysis / demo ------------------------------------
st.header("5. Analyze a Website (Live Demo)")
explanation = st.session_state.explanation

if st.session_state.is_demo:
    st.success(
        "**Demo Mode:** automatically classifying a typical phishing site -- "
        "Feature Input \u2192 Random Forest Prediction \u2192 Root Cause Analysis \u2192 Explanation."
    )

if explanation is None:
    st.write(
        "Choose a preset example or set custom feature values in the sidebar, "
        "then click **Analyze**, or click **Run Demo Mode** for an automatic walkthrough."
    )
else:
    sample = st.session_state.analyzed_sample
    flagged = [f for f in FEATURES if sample[f] == 1]

    st.markdown("**Feature values analyzed:**")
    feat_df = pd.DataFrame(
        [{"Feature": f, "Value": sample[f], "Flagged as suspicious": "Yes" if sample[f] == 1 else "No"} for f in FEATURES]
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("### Verdict")
    if explanation["verdict"] == "PHISHING":
        st.error(f"\u2717 PHISHING  (confidence: {explanation['confidence']:.1f}%)")
    else:
        st.success(f"\u2713 LEGITIMATE  (confidence: {explanation['confidence']:.1f}%)")

    # ---- Section 6: Explainable AI --------------------------------------
    st.header("6. Explainable AI: Root Cause Breakdown")
    if explanation["top_reasons"]:
        st.write("The verdict above was driven primarily by the following flagged indicators, ranked by the Random Forest's learned importance:")
        for reason in explanation["top_reasons"]:
            st.markdown(
                f"- **{reason['feature']}** (importance {reason['importance']:.3f}): {reason['explanation']}"
            )
    else:
        st.write("No major red flags were detected among the top 10 indicators -- this is consistent with a legitimate site.")

# ---- Section 7: System architecture -------------------------------------
st.header("7. System Architecture")
st.code(
    "User Input (feature values)\n"
    "    |\n"
    "    v\n"
    "Knowledge Base              (feature_schema.py)\n"
    "    |\n"
    "    v\n"
    "Dataset / Problem Formulation  (dataset.py)\n"
    "    |\n"
    "    v\n"
    "Learning Engine: Logistic Regression + Random Forest  (model_engine.py)\n"
    "    |\n"
    "    v\n"
    "Prediction (Phishing / Legitimate)\n"
    "    |\n"
    "    v\n"
    "Root Cause Analysis (Feature Importance)\n"
    "    |\n"
    "    v\n"
    "Explainable AI Output (this UI)",
    language="text",
)

# ---- Section 8: Limitations ----------------------------------------------
st.header("8. Limitations & Future Work")
st.warning(
    "This prototype trains on a **synthetic** dataset that mirrors the UCI "
    "Phishing Websites schema and the paper's reported feature-importance "
    "ranking, since this environment has no internet access to the original "
    "11,055-row dataset. Swap `dataset.load_dataset()` for a `pd.read_csv()` "
    "on the real UCI CSV to reproduce the paper's exact 96.70% figure. "
    "Future work (per the paper): real-time browser extension, NLP-based "
    "content analysis, and a non-technical-friendly dashboard."
)

st.divider()
st.caption(
    "AI Phishing Detection & Root Cause Analysis | Educational prototype for CIA-1 Part 5 | "
    "Based on 'AI-Based Phishing Detection Using Machine Learning and Root Cause Analysis'."
)
