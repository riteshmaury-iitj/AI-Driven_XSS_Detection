import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# CONFIGURATION
# ==========================
MAX_LENGTH = 100
HF_REPO_ID = "Recurrent/xsss_models"


def download_model(filename: str) -> str:
    """Download a model file from HuggingFace Hub and return the local path."""
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)


# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load(download_model("xss_rf_model.pkl"))
    tfidf = joblib.load(download_model("xss_tfidf_vectorizer.pkl"))
    tokenizer = joblib.load(download_model("xss_tokenizer.pkl"))

    lstm_model = load_model(download_model("xss_lstm_model.h5"))

    # Remove output layer to extract features
    feature_extractor = Sequential(lstm_model.layers[:-1])

    return rf_model, tfidf, tokenizer, feature_extractor


rf_model, tfidf, tokenizer, feature_extractor = load_artifacts()

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="XSS Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Driven XSS Detection System")
st.markdown(
    """
    Detect whether an input payload is a **Cross-Site Scripting (XSS) Attack**
    using a hybrid **LSTM + Random Forest** model.
    """
)

# ==========================
# USER INPUT
# ==========================
user_input = st.text_area(
    "Enter URL / Script / Payload",
    height=150,
    placeholder="<script>alert('XSS')</script>"
)

# ==========================
# PREDICTION
# ==========================
if st.button("Detect XSS"):

    if not user_input.strip():
        st.warning("Please enter a payload.")
    else:

        try:
            # --------------------------
            # TF-IDF Features
            # --------------------------
            tfidf_features = tfidf.transform([user_input]).toarray()

            # --------------------------
            # LSTM Features
            # --------------------------
            sequence = tokenizer.texts_to_sequences([user_input])

            padded_sequence = pad_sequences(
                sequence,
                maxlen=MAX_LENGTH
            )

            lstm_features = feature_extractor.predict(
                padded_sequence,
                verbose=0
            )

            # --------------------------
            # Combine Features
            # --------------------------
            combined_features = np.hstack(
                (tfidf_features, lstm_features)
            )

            # --------------------------
            # Prediction
            # --------------------------
            prediction = rf_model.predict(combined_features)[0]

            probability = rf_model.predict_proba(
                combined_features
            )[0]

            # ==========================
            # DISPLAY RESULTS
            # ==========================
            st.subheader("Result")

            if prediction == 1:
                st.error(
                    f"⚠️ XSS Attack Detected\n\nConfidence: {max(probability)*100:.2f}%"
                )
            else:
                st.success(
                    f"✅ Safe Script \n\nConfidence: {max(probability)*100:.2f}%"
                )

            st.subheader("Prediction Probabilities")

            st.write({
                "SAFE": float(probability[0]),
                "XSS": float(probability[1])
            })

            # Store results in session state for analysis
            st.session_state["probability"] = probability
            st.session_state["prediction"] = prediction
            st.session_state["has_result"] = True

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ==========================
# ANALYSIS BUTTON
# ==========================
if st.session_state.get("has_result"):
    if st.button("📊 Show Graph"):
        probability = st.session_state["probability"]
        prediction = st.session_state["prediction"]
        safe_conf = float(probability[0]) * 100
        xss_conf = float(probability[1]) * 100

        st.subheader("📊 Confidence Analysis Dashboard")

        # ---------- Row 1: Gauge + Bar ----------
        col1, col2 = st.columns(2)

        with col1:
            # Bar Chart
            fig_bar, ax_bar = plt.subplots(figsize=(5, 3.5))
            bars = ax_bar.bar(["SAFE", "XSS"], [safe_conf, xss_conf], color=["#2ecc71", "#e74c3c"], edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, [safe_conf, xss_conf]):
                ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{val:.2f}%", ha="center", fontweight="bold", fontsize=11)
            ax_bar.set_ylim(0, 105)
            ax_bar.set_ylabel("Confidence (%)")
            ax_bar.set_title("Confidence: SAFE vs XSS", fontweight="bold")
            ax_bar.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        with col2:
            # Donut / Pie Chart
            fig_pie, ax_pie = plt.subplots(figsize=(5, 3.5))
            wedges, texts, autotexts = ax_pie.pie(
                [safe_conf, xss_conf],
                labels=["SAFE", "XSS"],
                colors=["#2ecc71", "#e74c3c"],
                autopct="%1.2f%%",
                startangle=90,
                pctdistance=0.75,
                wedgeprops=dict(width=0.4, edgecolor="white"),
            )
            for t in autotexts:
                t.set_fontweight("bold")
            ax_pie.set_title("Confidence Distribution", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_pie)
            plt.close(fig_pie)

        # ---------- Summary Metrics ----------
        st.markdown("#### Summary Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("SAFE Confidence", f"{safe_conf:.2f}%")
        m2.metric("XSS Confidence", f"{xss_conf:.2f}%")
        m3.metric(
            "Verdict",
            "⚠️ XSS ATTACK" if prediction == 1 else "✅ SAFE",
        )

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("About")

st.sidebar.info(
    """
    Model Pipeline:

    1. Tokenization
    2. LSTM Feature Extraction
    3. TF-IDF Feature Extraction
    4. Feature Fusion
    5. Random Forest Classification

    Developed for XSS Detection Research.
    """
)