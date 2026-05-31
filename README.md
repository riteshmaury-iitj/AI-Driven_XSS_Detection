# AI-Driven Detection and Prevention of Cross-Site Scripting (XSS) Attacks

A hybrid **LSTM + Random Forest** model for detecting Cross-Site Scripting (XSS) attacks in real-time using deep learning feature extraction and ensemble classification.

![Architecture](xss_architecture.png)

## Project Structure

```
AI-Driven-XSS-Detection/
├── AI_Driven_XSS_Detection.ipynb   # Main training notebook
├── XSS_testing.ipynb               # Model testing & evaluation
├── app.py                          # Streamlit web application
├── XSS_dataset.csv                 # Dataset (Sentence, Label)
├── xss_lstm_model.h5               # Trained LSTM model
├── xss_rf_model.pkl                # Trained Random Forest model
├── xss_tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
├── xss_tokenizer.pkl               # Fitted Keras tokenizer
├── xss_architecture.png            # Architecture diagram
├── requirements.txt                # Python dependencies
└── .gitignore
```

## Architecture

1. **Data Input** — XSS dataset with labeled scripts (Safe / Malicious)
2. **Preprocessing** — Label encoding, Train/Test split (80:20)
3. **Feature Extraction**
   - TF-IDF Vectorizer (5,000 features)
   - LSTM Neural Network → Feature Extractor (64-dim)
4. **Feature Fusion (LSTF)** — Concatenate TF-IDF + LSTM features (5,064-dim)
5. **Classification** — Random Forest, Logistic Regression, Naive Bayes, Decision Tree, GRU
6. **Evaluation** — Accuracy, Confusion Matrix, ROC-AUC, K-Fold CV, SHAP
7. **Deployment** — Streamlit web app for real-time detection

## Model Accuracy

| Model               | Accuracy |
|----------------------|----------|
| Random Forest        | 99.74%   |
| Logistic Regression  | 99.74%   |
| Decision Tree        | 99.74%   |
| LSTM                 | 99.77%   |
| GRU                  | 99.95%   |
| Naive Bayes          | 88.50%   |

**Random Forest K-Fold CV (10):** 99.94%

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Tech Stack

- Python, TensorFlow/Keras, Scikit-learn
- LSTM, GRU, Random Forest, TF-IDF
- Streamlit, Matplotlib, SHAP
