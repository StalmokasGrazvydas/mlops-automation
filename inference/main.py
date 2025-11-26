import os
import io
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image

# Try to import TF only once app starts (prevents slow import when running tests)
import tensorflow as tf
from tensorflow import keras

app = FastAPI(title="Animals Class Counter API", version="1.0.0")

# --- Where we expect the model to be (adjust if your paths differ) ---
CANDIDATE_PATHS: List[str] = [
    os.path.join("job_artifacts", "named-outputs", "model", "model.keras"),
    os.path.join("model_artifacts", "animals-class-counter", "model.keras"),
    # SavedModel directory (if present):
    os.path.join("job_artifacts", "named-outputs", "model"),
    os.path.join("model_artifacts", "animals-class-counter"),
]

class_names_default = [f"class_{i}" for i in range(10)]  # fallback labels

def _is_saved_model_dir(p: str) -> bool:
    return os.path.isdir(p) and os.path.exists(os.path.join(p, "saved_model.pb"))


def _load_model():
    # 1) Prefer .keras file
    for p in CANDIDATE_PATHS:
        if p.endswith(".keras") and os.path.exists(p):
            try:
                m = keras.models.load_model(p)
                print(f"[INFO] Loaded Keras model: {p}")
                return m
            except Exception as e:
                print(f"[WARN] Failed to load keras model at {p}: {e}")

    # 2) Check for SavedModel directories
    for p in CANDIDATE_PATHS:
        if _is_saved_model_dir(p):
            try:
                # Option A: load as a Keras model
                m = keras.models.load_model(p)
                print(f"[INFO] Loaded SavedModel via keras.load_model: {p}")
                return m
            except Exception as e:
                print(f"[WARN] keras.load_model failed at {p}: {e}")
                try:
                    # Option B: TFSMLayer fallback (if model is a serving graph)
                    layer = keras.layers.TFSMLayer(p, call_endpoint="serving_default")
                    # Wrap layer in a simple Keras model with an Input spec.
                    # You may need to adapt input shape to your training input.
                    dummy_input = keras.Input(shape=(224, 224, 3), dtype="float32")
                    out = layer(dummy_input)
                    m = keras.Model(dummy_input, out)
                    print(f"[INFO] Wrapped SavedModel via TFSMLayer: {p}")
                    return m
                except Exception as e2:
                    print(f"[WARN] TFSMLayer fallback failed at {p}: {e2}")

    # 3) Dummy model fallback so the API still runs
    class DummyModel:
        def __call__(self, x):
            # Return zeros with a reasonable shape
            batch = x.shape[0]
            return np.zeros((batch, len(class_names_default)), dtype=np.float32)
    print("[INFO] Using DummyModel (no real model file found).")
    return DummyModel()

model = _load_model()

def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    # Convert to RGB, resize, scale to [0,1], add batch dim
    image = image.convert("RGB").resize(target_size)
    arr = np.asarray(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

class PredictResponse(BaseModel):
    top_classes: List[str]
    top_scores: List[float]
    raw_scores: Optional[List[float]] = None

@app.get("/")
def root():
    return {"status": "ok", "message": "Animals Class Counter API running."}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), top_k: int = 3, return_raw: bool = False):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    x = preprocess_image(img, target_size=(224, 224))

    # Run inference (supports both Keras models and DummyModel)
    try:
        preds = model(x)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = np.array(preds)
        if preds.ndim == 2:
            preds = preds[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Softmax if it doesn’t look like probabilities
    if preds.min() < 0 or preds.max() > 1.0:
        preds = tf.nn.softmax(preds).numpy()

    # top-k
    k = min(top_k, preds.shape[0] if preds.ndim==1 else preds.shape[-1])
    indices = np.argsort(preds)[-k:][::-1]
    class_names = class_names_default  # swap with your real labels if available
    top_classes = [class_names[i] if i < len(class_names) else f"class_{i}" for i in indices]
    top_scores = [float(preds[i]) for i in indices]

    resp = PredictResponse(
        top_classes=top_classes,
        top_scores=top_scores,
        raw_scores=preds.tolist() if return_raw else None
    )
    return resp
