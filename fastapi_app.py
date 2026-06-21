from contextlib import asynccontextmanager
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# CONFIGURATION
# ==========================
MAX_LENGTH = 100
HF_REPO_ID = "Recurrent/xsss_models"

# ==========================
# MODEL LOADING
# ==========================
models = {}


def download_model(filename: str) -> str:
    """Download a model file from HuggingFace Hub, fallback to local."""
    if os.path.exists(filename):
        return filename

    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
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

    models["rf_model"] = rf_model
    models["tfidf"] = tfidf
    models["tokenizer"] = tokenizer
    models["feature_extractor"] = feature_extractor

    yield

    models.clear()


# ==========================
# APP INITIALIZATION
# ==========================
app = FastAPI(
    title="AI-Driven XSS Detection API",
    description="Detect Cross-Site Scripting (XSS) attacks using a hybrid LSTM + Random Forest model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================
# SCHEMAS
# ==========================
class PayloadRequest(BaseModel):
    payload: str = Field(
        ...,
        min_length=1,
        description="The URL, script, or payload to analyze for XSS.",
        json_schema_extra={"example": "<script>alert('XSS')</script>"},
    )


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


# ==========================
# ENDPOINTS
# ==========================
@app.get("/health")
async def health_check():
    """Check if the API and models are ready."""
    return {"status": "healthy", "models_loaded": bool(models)}


@app.post("/predict", response_model=PredictionResponse)
async def predict_xss(request: PayloadRequest):
    """Predict whether the given payload is an XSS attack."""
    user_input = request.payload.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Payload cannot be empty.")

    try:
        rf_model = models["rf_model"]
        tfidf = models["tfidf"]
        tokenizer = models["tokenizer"]
        feature_extractor = models["feature_extractor"]

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

        label = "XSS" if prediction == 1 else "SAFE"
        confidence = float(max(probability)) * 100

        return PredictionResponse(
            prediction=label,
            confidence=round(confidence, 2),
            probabilities={
                "SAFE": round(float(probability[0]), 4),
                "XSS": round(float(probability[1]), 4),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/predict/batch")
async def predict_batch(payloads: list[str]):
    """Predict XSS for multiple payloads at once."""
    if not payloads:
        raise HTTPException(status_code=400, detail="Payload list cannot be empty.")

    if len(payloads) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 payloads per batch.")

    results = []
    for payload in payloads:
        request = PayloadRequest(payload=payload)
        result = await predict_xss(request)
        results.append(result)

    return results


# ==========================
# RUN SERVER
# ==========================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
