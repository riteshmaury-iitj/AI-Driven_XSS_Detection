# AI-Driven Detection and Prevention of Cross-Site Scripting (XSS) Attacks

A hybrid **LSTM + Random Forest** model for detecting Cross-Site Scripting (XSS) attacks in real-time using deep learning feature extraction and ensemble classification.

**Live Demo:**
- 🌐 **Streamlit App:** https://xss-detection.streamlit.app/
- ⚡ **FastAPI Docs:** https://ai-driven-xss-detection-1.onrender.com/docs

## Project Structure

```
AI-Driven-XSS-Detection/
├── AI_Driven_XSS_Detection.ipynb   # Main training notebook
├── XSS_testing.ipynb               # Model testing & evaluation
├── app.py                          # Streamlit web application
├── fastapi_app.py                  # FastAPI REST API
├── XSS_dataset.csv                 # Primary dataset (13,686 samples)
├── XSS_dataset_new.csv             # Additional dataset (6,042 samples)
├── xss_lstm_model.h5               # Trained LSTM model weights
├── xss_rf_model.pkl                # Trained Random Forest model
├── xss_tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
├── xss_tokenizer.pkl               # Fitted Keras tokenizer
├── requirements.txt                # Python dependencies
├── .python-version                 # Python version for deployment
└── .gitignore
```

## Dataset

The model is trained on a **combined dataset of 19,728 samples** from two sources:

| Dataset | Samples | Safe | XSS |
|---------|---------|------|-----|
| XSS_dataset.csv | 13,686 | 8,459 | 5,227 |
| XSS_dataset_new.csv | 6,042 | 2,262 | 3,780 |
| **Combined** | **19,728** | **10,721** | **9,007** |

The additional dataset improves model performance by providing more diverse XSS attack patterns and better class balance.

## Architecture

1. **Data Input** — Combined XSS dataset with labeled scripts (Safe / Malicious)
2. **Preprocessing** — Label encoding, Train/Test split (80:20)
3. **Feature Extraction**
   - TF-IDF Vectorizer (5,000 features)
   - LSTM Neural Network → Feature Extractor (64-dim)
4. **Feature Fusion (LSTF)** — Concatenate TF-IDF + LSTM features (5,064-dim)
5. **Classification** — Random Forest, Logistic Regression, Naive Bayes, Decision Tree, GRU
6. **Evaluation** — Accuracy, Confusion Matrix, ROC-AUC, K-Fold CV, SHAP
7. **Deployment** — Streamlit UI + FastAPI REST API

## Model Performance

| Model               | Accuracy |
|----------------------|----------|
| **Random Forest**    | **99.74%** |
| Logistic Regression  | 99.74%   |
| Decision Tree        | 99.74%   |
| LSTM                 | 99.77%   |
| GRU                  | 99.95%   |
| Naive Bayes          | 88.50%   |

**Random Forest Classification Report:**
| Metric | Safe (0) | XSS (1) | Weighted Avg |
|--------|----------|----------|--------------|
| Precision | 1.00 | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 | 1.00 |
| F1-Score | 1.00 | 1.00 | 1.00 |

**K-Fold Cross Validation (10-fold):** 99.94%

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Run the FastAPI backend
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"payload": "<script>alert(1)</script>"}'
```

Response:
```json
{
  "prediction": "XSS",
  "confidence": 99.87,
  "probabilities": {"SAFE": 0.0013, "XSS": 0.9987}
}
```

## Deployment (Render)

The app is configured for deployment on Render:
- **Python Version:** 3.10.0 (`.python-version`)
- **Start Command:** `uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT`
- **Models:** Downloaded from HuggingFace Hub (`Recurrent/xsss_models`)

## Tech Stack

- Python, TensorFlow/Keras, Scikit-learn
- LSTM, GRU, Random Forest, TF-IDF
- FastAPI, Streamlit, Plotly, SHAP
- HuggingFace Hub, Render
