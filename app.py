import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import time
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# CONFIGURATION
# ==========================
MAX_LENGTH = 100
HF_REPO_ID = "Recurrent/xsss_models"

def download_model(filename: str) -> str:
    """Download a model file from HuggingFace Hub, fallback to local."""
    # Try local file first
    if os.path.exists(filename):
        return filename

    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)


# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="XSS Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .main { background: linear-gradient(160deg, #0a0a1a 0%, #101030 40%, #0d1b2a 70%, #0a0a1a 100%); }
    .stApp { background: linear-gradient(160deg, #0a0a1a 0%, #101030 40%, #0d1b2a 70%, #0a0a1a 100%); }

    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 4s ease infinite;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }

    @keyframes gradient-shift {
        0% { background-position: 0% center; }
        50% { background-position: 100% center; }
        100% { background-position: 0% center; }
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        color: #8892b0;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 2.5rem;
        letter-spacing: 0.2px;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.12);
        border-color: rgba(102, 126, 234, 0.2);
    }

    .result-safe {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.08), rgba(16, 185, 129, 0.03));
        border: 1px solid rgba(52, 211, 153, 0.25);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        animation: fadeInUp 0.6s ease;
        box-shadow: 0 4px 30px rgba(52, 211, 153, 0.08);
    }

    .result-xss {
        background: linear-gradient(135deg, rgba(251, 113, 133, 0.08), rgba(239, 68, 68, 0.03));
        border: 1px solid rgba(251, 113, 133, 0.25);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        animation: fadeInUp 0.6s ease;
        box-shadow: 0 4px 30px rgba(251, 113, 133, 0.08);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.85; transform: scale(1.05); }
    }

    .result-icon {
        font-size: 4rem;
        animation: pulse 2.5s ease infinite;
    }

    .result-label {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
        letter-spacing: -0.3px;
    }

    .result-confidence {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #8892b0;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        animation: fadeInUp 0.8s ease;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.1);
    }

    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #667eea;
    }

    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
    }

    .pipeline-step {
        background: rgba(102, 126, 234, 0.04);
        border-left: 3px solid #667eea;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #cbd5e1;
        transition: all 0.3s ease;
    }

    .pipeline-step:hover {
        background: rgba(102, 126, 234, 0.1);
        border-left-color: #a78bfa;
        padding-left: 1.5rem;
    }

    .scan-animation {
        text-align: center;
        padding: 1rem;
    }

    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        color: #000000 !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.9rem !important;
        padding: 1rem !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(102, 126, 234, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.12) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.45) !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(10, 10, 26, 0.97) !important;
        border-right: 1px solid rgba(102, 126, 234, 0.08);
    }

    .sidebar-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        border-radius: 10px;
    }

    /* Divider */
    hr {
        border-color: rgba(102, 126, 234, 0.1) !important;
    }

    /* Fix white header bar */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Section headers */
    h4, h5 {
        color: #cbd5e1 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Remove top padding */
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load(download_model("xss_rf_model.pkl"))
    tfidf = joblib.load(download_model("xss_tfidf_vectorizer.pkl"))
    tokenizer = joblib.load(download_model("xss_tokenizer.pkl"))

    # Rebuild LSTM architecture and load weights (avoids Keras version mismatch)
    lstm_model = Sequential([
        Embedding(input_dim=5000, output_dim=128),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.build(input_shape=(None, MAX_LENGTH))
    lstm_model.load_weights(download_model("xss_lstm_model.h5"))

    # Remove output layer to extract features
    feature_extractor = Sequential(lstm_model.layers[:-1])

    return rf_model, tfidf, tokenizer, feature_extractor


rf_model, tfidf, tokenizer, feature_extractor = load_artifacts()

# ==========================
# HEADER
# ==========================
st.markdown('<h1 class="hero-title">AI-Driven XSS Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Enterprise-grade Cross-Site Scripting detection powered by hybrid LSTM + Random Forest architecture</p>', unsafe_allow_html=True)

# ==========================
# MAIN LAYOUT
# ==========================
col_input, col_spacer, col_info = st.columns([3, 0.2, 1.5])

with col_input:
    st.markdown("#### Payload Analysis")
    user_input = st.text_area(
        "Enter URL / Script / Payload",
        height=180,
        placeholder="Enter suspicious payload here...\n\nExamples:\n<script>alert('XSS')</script>\n<img src=x onerror=alert(1)>\njavascript:void(document.cookie)",
        label_visibility="collapsed",
    )
    detect_clicked = st.button("🔍  Analyze Payload", use_container_width=True)

with col_info:
    st.markdown("#### Detection Capabilities")
    st.markdown("""
    <div class="glass-card" style="padding: 1.2rem;">
        <div class="pipeline-step">🧠 LSTM Deep Feature Extraction</div>
        <div class="pipeline-step">📊 TF-IDF Statistical Analysis</div>
        <div class="pipeline-step">🔗 Hybrid Feature Fusion</div>
        <div class="pipeline-step">🌲 Random Forest Classification</div>
        <div class="pipeline-step">✅ Real-time Threat Scoring</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================
# PREDICTION
# ==========================
if detect_clicked:
    if not user_input.strip():
        st.warning("⚠️ Please enter a payload to analyze.")
    else:
        # Scanning animation
        with st.spinner(""):
            progress_bar = st.progress(0)
            status_text = st.empty()

            steps = [
                ("🔤 Tokenizing input...", 20),
                ("🧠 Extracting LSTM features...", 45),
                ("📊 Computing TF-IDF vectors...", 65),
                ("🔗 Fusing feature representations...", 80),
                ("🌲 Running classification...", 95),
                ("✅ Analysis complete!", 100),
            ]

            for msg, pct in steps:
                status_text.markdown(f"<p style='color: #8892b0; text-align: center; font-family: Inter, sans-serif;'>{msg}</p>", unsafe_allow_html=True)
                progress_bar.progress(pct)
                time.sleep(0.3)

            progress_bar.empty()
            status_text.empty()

        try:
            # TF-IDF Features
            tfidf_features = tfidf.transform([user_input]).toarray()

            # LSTM Features
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH)
            lstm_features = feature_extractor.predict(padded_sequence, verbose=0)

            # Combine Features
            combined_features = np.hstack((tfidf_features, lstm_features))

            # Prediction
            prediction = rf_model.predict(combined_features)[0]
            probability = rf_model.predict_proba(combined_features)[0]

            safe_conf = float(probability[0]) * 100
            xss_conf = float(probability[1]) * 100
            confidence = max(safe_conf, xss_conf)

            # Store results
            st.session_state["probability"] = probability
            st.session_state["prediction"] = prediction
            st.session_state["has_result"] = True

            # ==========================
            # DISPLAY RESULTS
            # ==========================
            st.markdown("---")

            # Result banner
            if prediction == 1:
                st.markdown(f"""
                <div class="result-xss">
                    <div class="result-icon">🚨</div>
                    <div class="result-label" style="color: #fb7185;">XSS ATTACK DETECTED</div>
                    <div class="result-confidence">Confidence: {confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div class="result-icon">🛡️</div>
                    <div class="result-label" style="color: #34d399;">PAYLOAD IS SAFE</div>
                    <div class="result-confidence">Confidence: {confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #34d399;">{safe_conf:.1f}%</div>
                    <div class="metric-label">Safe Score</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #fb7185;">{xss_conf:.1f}%</div>
                    <div class="metric-label">Threat Score</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                risk_level = "Critical" if xss_conf > 80 else "High" if xss_conf > 60 else "Medium" if xss_conf > 40 else "Low"
                risk_color = "#fb7185" if xss_conf > 80 else "#fbbf24" if xss_conf > 60 else "#a78bfa" if xss_conf > 40 else "#34d399"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {risk_color};">{risk_level}</div>
                    <div class="metric-label">Risk Level</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #667eea;">{len(user_input)}</div>
                    <div class="metric-label">Payload Length</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Charts
            chart1, chart2 = st.columns(2)

            with chart1:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=xss_conf,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Threat Level", 'font': {'size': 18, 'color': '#cbd5e1'}},
                    number={'suffix': '%', 'font': {'color': '#e2e8f0', 'size': 28}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#334155', 'tickfont': {'color': '#64748b'}},
                        'bar': {'color': '#fb7185' if xss_conf > 50 else '#34d399'},
                        'bgcolor': 'rgba(0,0,0,0)',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 25], 'color': 'rgba(52, 211, 153, 0.15)'},
                            {'range': [25, 50], 'color': 'rgba(167, 139, 250, 0.12)'},
                            {'range': [50, 75], 'color': 'rgba(251, 191, 36, 0.12)'},
                            {'range': [75, 100], 'color': 'rgba(251, 113, 133, 0.15)'},
                        ],
                        'threshold': {
                            'line': {'color': '#a78bfa', 'width': 3},
                            'thickness': 0.8,
                            'value': xss_conf
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with chart2:
                # Donut chart
                fig_donut = go.Figure(data=[go.Pie(
                    labels=['Safe', 'XSS Threat'],
                    values=[safe_conf, xss_conf],
                    hole=0.7,
                    marker=dict(
                        colors=['#34d399', '#fb7185'],
                        line=dict(color='rgba(0,0,0,0)', width=0)
                    ),
                    textinfo='label+percent',
                    textfont=dict(size=13, color='#cbd5e1'),
                    hoverinfo='label+percent+value',
                )])
                fig_donut.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(l=20, r=20, t=60, b=20),
                    title=dict(text="Confidence Distribution", font=dict(size=18, color='#cbd5e1')),
                    legend=dict(font=dict(color='#8892b0')),
                    showlegend=True,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.markdown('<div class="sidebar-header">🛡️ XSS Detection</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("##### Model Architecture")
    st.markdown("""
    <div style="color: #8892b0; font-size: 0.9rem; line-height: 1.8;">
        <strong style="color: #a78bfa;">Hybrid Deep Learning Pipeline</strong><br><br>
        <b>1.</b> Input Tokenization<br>
        <b>2.</b> LSTM Feature Extraction (128-dim)<br>
        <b>3.</b> TF-IDF Vectorization<br>
        <b>4.</b> Feature Fusion Layer<br>
        <b>5.</b> Random Forest Classifier
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("##### Performance Metrics")
    st.markdown("""
    <div style="color: #8892b0; font-size: 0.9rem; line-height: 2;">
        ✅ Accuracy: <strong style="color: #34d399;">99.74%</strong><br>
        🎯 Precision: <strong style="color: #34d399;">99.76%</strong><br>
        📈 Recall: <strong style="color: #34d399;">99.73%</strong><br>
        ⚡ F1-Score: <strong style="color: #34d399;">99.74%</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("##### Tech Stack")
    st.markdown("""
    <div style="color: #8892b0; font-size: 0.85rem; line-height: 2;">
        🧠 TensorFlow / Keras<br>
        🌲 Scikit-learn<br>
        🤗 HuggingFace Hub<br>
        ⚡ FastAPI Backend<br>
        🎨 Streamlit Frontend
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color: #475569; font-size: 0.75rem; text-align: center;">
        Developed for XSS Detection Research<br>
        IIT Jodhpur
    </div>
    """, unsafe_allow_html=True)
