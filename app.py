"""
app.py
=====================================================
AI Smart Agro System — External AI API server.

This is the "external AI server" referenced in the PHP project's
includes/config.php (AI_DISEASE_API_URL / AI_CROP_API_URL). It runs
separately from the InfinityFree PHP site because InfinityFree cannot
run Python/TensorFlow.

Two REAL (non-demo) endpoints:

1. POST /api/predict-disease
   - Accepts a multipart/form-data image upload (field name "image")
   - Forwards it to a free, publicly available, pre-trained plant-disease
     image classification model hosted on Hugging Face's Inference API
     (trained on the real PlantVillage dataset — genuine AI, not a rule).
   - Maps the raw class label to farmer-friendly text via disease_info.py
   - Returns JSON matching what the PHP site expects.

2. POST /api/recommend-crop
   - Accepts JSON: {soil_type, nitrogen, phosphorus, potassium,
     temperature, humidity, rainfall, ph}
   - Runs them through a REAL trained scikit-learn RandomForestClassifier
     (see train_crop_model.py) and returns the top 3 predicted crops
     with genuine model-derived probabilities.

Security:
- Requires the same X-API-KEY header value configured in the PHP site's
  includes/config.php (AI_API_KEY), so random people can't hit your
  Hugging Face free quota or your model directly.

Deploy this on any free Python host (Render, Railway, PythonAnywhere).
See README.md in this folder for full deployment steps.
"""
import os
import json
import time
import joblib
import requests
from flask import Flask, request, jsonify
from disease_info import get_disease_info

app = Flask(__name__)

# ---------------- CONFIG (set these as environment variables on your host) ----------------
API_SHARED_KEY = os.environ.get("AI_API_KEY", "replace-with-a-shared-secret-key")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")  # Get a free token at huggingface.co/settings/tokens
# Any public PlantVillage-trained model on the Hugging Face Hub that supports
# image-classification via the free Inference API. Verify the model still
# exists/works before going live — swap this if the model repo changes.
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

BASE_DIR = os.path.dirname(__file__)
CROP_MODEL_PATH = os.path.join(BASE_DIR, "crop_model.joblib")
CROP_RANGES_PATH = os.path.join(BASE_DIR, "crop_ranges.json")

_crop_bundle = None
_crop_ranges = None


def load_crop_model():
    global _crop_bundle, _crop_ranges
    if _crop_bundle is None:
        if not os.path.exists(CROP_MODEL_PATH):
            raise RuntimeError("crop_model.joblib not found. Run train_crop_model.py first.")
        _crop_bundle = joblib.load(CROP_MODEL_PATH)
        with open(CROP_RANGES_PATH) as f:
            _crop_ranges = json.load(f)
    return _crop_bundle, _crop_ranges


def check_auth():
    key = request.headers.get("X-API-KEY", "")
    return key == API_SHARED_KEY


@app.route("/api/predict-disease", methods=["POST"])
def predict_disease():
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401

    if "image" not in request.files:
        return jsonify({"error": "No image file provided (expected field name 'image')."}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    if not HF_API_TOKEN:
        return jsonify({"error": "Server misconfigured: HF_API_TOKEN not set."}), 500

    try:
        hf_response = requests.post(
            HF_INFERENCE_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            data=image_bytes,
            timeout=50,
        )
        # If the model is "cold" (not loaded yet), Hugging Face returns 503
        # with an estimated_time. Wait briefly and retry once automatically
        # instead of immediately giving up and falling back to demo mode.
        if hf_response.status_code == 503:
            try:
                wait_hint = hf_response.json().get("estimated_time", 8)
            except Exception:
                wait_hint = 8
            time.sleep(min(float(wait_hint), 20))
            hf_response = requests.post(
                HF_INFERENCE_URL,
                headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
                data=image_bytes,
                timeout=50,
            )
    except requests.RequestException as e:
        return jsonify({"error": f"Could not reach AI model service: {e}"}), 502

    if hf_response.status_code == 503:
        # Model is "cold" and loading on Hugging Face's free tier — ask caller to retry shortly.
        return jsonify({"error": "AI model is warming up, please try again in ~20 seconds."}), 503

    if hf_response.status_code != 200:
        return jsonify({"error": f"AI model service error ({hf_response.status_code}): {hf_response.text[:300]}"}), 502

    predictions = hf_response.json()
    if not isinstance(predictions, list) or not predictions:
        return jsonify({"error": "Unexpected response format from AI model service."}), 502

    # Hugging Face image-classification pipeline returns a list like:
    # [{"label": "Tomato___Early_blight", "score": 0.93}, ...] sorted by score desc
    top = predictions[0]
    raw_label = top.get("label", "unknown")
    confidence = round(float(top.get("score", 0)) * 100, 2)

    info = get_disease_info(raw_label)

    return jsonify({
        "crop_name": info["crop"],
        "disease_name": info["disease"],
        "confidence": confidence,
        "description": info["description"],
        "symptoms": info["symptoms"],
        "treatment": info["treatment"],
        "prevention": info["prevention"],
        "raw_label": raw_label,
    })


@app.route("/api/recommend-crop", methods=["POST"])
def recommend_crop():
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or request.form
    required = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "rainfall"]
    for field in required:
        if field not in data or data[field] in (None, ""):
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        features = [
            float(data["nitrogen"]),
            float(data["phosphorus"]),
            float(data["potassium"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data.get("ph", 6.5)),
            float(data["rainfall"]),
        ]
    except (TypeError, ValueError):
        return jsonify({"error": "All numeric fields must be valid numbers."}), 400

    bundle, ranges = load_crop_model()
    model = bundle["model"]
    classes = bundle["classes"]

    probs = model.predict_proba([features])[0]
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]

    recommendations = []
    for crop, prob in ranked:
        r = ranges.get(crop, {})
        temp_range = r.get("temp", ["-", "-"])
        rain_range = r.get("rainfall", ["-", "-"])
        recommendations.append({
            "crop_name": crop,
            "score": round(float(prob) * 100, 1),
            "suitable_conditions": f"Typically grown at {temp_range[0]}-{temp_range[1]}°C with {rain_range[0]}-{rain_range[1]}mm rainfall.",
            "info": "Prediction from a trained Random Forest model based on documented agronomic requirement ranges. Consult a local agricultural officer to confirm suitability for your exact field conditions.",
        })

    return jsonify({"recommendations": recommendations})


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "AI Smart Agro AI server is running.", "demo_mode": False})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
