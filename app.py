q"""
app.py  —  Face Recognition Flask Web Application
Supports:
  • /              → main UI (webcam + upload)
  • /predict/sklearn   → POST image → sklearn model result
  • /predict/teachable → POST image → Teachable Machine result (TM.js runs client-side)
"""

import os
import io
import base64
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB max upload

# ── Load sklearn model ────────────────────────────────────────────────────────
SKLEARN_MODEL_PATH = os.path.join("models", "sklearn_model.pkl")
sklearn_payload = None

def load_sklearn_model():
    global sklearn_payload
    if os.path.exists(SKLEARN_MODEL_PATH):
        sklearn_payload = joblib.load(SKLEARN_MODEL_PATH)
        print(f"[OK] Sklearn model loaded: {sklearn_payload['model_name']} "
              f"(accuracy: {sklearn_payload['test_accuracy']:.4f})")
    else:
        print(f"[WARN] sklearn model not found at {SKLEARN_MODEL_PATH}. "
              "Run train_model.py first.")

load_sklearn_model()

# ── Haar cascade ──────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_image(data_source) -> np.ndarray | None:
    """Accept a Flask file upload or a base64 data URL string → BGR numpy array."""
    try:
        if isinstance(data_source, str):
            # base64 data URL: "data:image/jpeg;base64,/9j/..."
            header, encoded = data_source.split(",", 1)
            img_bytes = base64.b64decode(encoded)
        else:
            img_bytes = data_source.read()

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[ERROR] decode_image: {e}")
        return None


def extract_face_features(bgr_img: np.ndarray,
                           img_size=(64, 64)) -> np.ndarray | None:
    """Detect face in BGR image → resize → flatten feature vector."""
    gray  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

    if len(faces) == 0:
        # No face detected — use full image resized
        roi = cv2.resize(gray, img_size)
        face_found = False
    else:
        x, y, w, h = faces[0]
        roi = cv2.resize(gray[y:y+h, x:x+w], img_size)
        face_found = True

    feat = roi.flatten().astype(np.float32) / 255.0
    return feat, face_found


def get_sklearn_prediction(bgr_img: np.ndarray) -> dict:
    """Run sklearn model on a BGR image and return result dict."""
    if sklearn_payload is None:
        return {"error": "Model not loaded. Run train_model.py first."}

    model   = sklearn_payload["model"]
    scaler  = sklearn_payload["scaler"]
    labels  = sklearn_payload["labels"]           # {0:"NOT_ME", 1:"ME"}
    size    = sklearn_payload["img_size"]

    feat, face_found = extract_face_features(bgr_img, img_size=size)
    feat_scaled = scaler.transform([feat])

    pred    = int(model.predict(feat_scaled)[0])
    proba   = model.predict_proba(feat_scaled)[0] if hasattr(model, "predict_proba") else None
    label   = labels[pred]

    confidence = None
    if proba is not None:
        confidence = round(float(max(proba)) * 100, 1)

    return {
        "label":      label,
        "is_me":      label == "ME",
        "confidence": confidence,
        "face_found": face_found,
        "model_name": sklearn_payload["model_name"],
        "model_accuracy": round(sklearn_payload["test_accuracy"] * 100, 1),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    model_loaded = sklearn_payload is not None
    model_name   = sklearn_payload["model_name"] if model_loaded else "Not loaded"
    model_acc    = (round(sklearn_payload["test_accuracy"] * 100, 1)
                    if model_loaded else None)
    return render_template("index.html",
                           model_loaded=model_loaded,
                           model_name=model_name,
                           model_acc=model_acc)


@app.route("/predict/sklearn", methods=["POST"])
def predict_sklearn():
    """Accepts: JSON {image: 'data:image/...;base64,...'} or multipart file."""
    try:
        if request.is_json:
            data     = request.get_json()
            bgr_img  = decode_image(data.get("image", ""))
        else:
            file     = request.files.get("image")
            bgr_img  = decode_image(file) if file else None

        if bgr_img is None:
            return jsonify({"error": "Could not decode image."}), 400

        result = get_sklearn_prediction(bgr_img)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "sklearn_model": sklearn_payload["model_name"] if sklearn_payload else None,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
