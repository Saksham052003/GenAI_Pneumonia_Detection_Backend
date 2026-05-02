import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Dense
import google.generativeai as genai
import datetime
import gdown
from dotenv import load_dotenv

load_dotenv()

# ─── Setup ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

IMG_SIZE = (224, 224)

# ─── Gemini API Setup ─────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

genai.configure(api_key=GEMINI_API_KEY)

# ─── FIX Dense globally (VERY IMPORTANT) ───────────
_original_dense_init = Dense.__init__

def patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _original_dense_init(self, *args, **kwargs)

Dense.__init__ = patched_dense_init

# ─── Load Model ───────────────────────────────────────
MODEL_URL = os.getenv("MODEL_URL")
MODEL_PATH = "/tmp/model.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")

        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

        print("Model downloaded!")
        print("Model size:", os.path.getsize(MODEL_PATH))
    else:
        print("Model already exists. Skipping download.")

download_model()

model = load_model(MODEL_PATH, compile=False)

print("✅ Model Loaded Successfully")

# ─── Preprocess Image ─────────────────────────────────
def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(image) / 255.0
    return np.expand_dims(arr, axis=0)

# ─── Generate Report using Gemini ─────────────────────
def generate_report(prediction, confidence):

    prompt = f"""
You are an expert radiologist AI assistant.

Analysis Result:
- Prediction: {prediction}
- Confidence: {confidence:.1f}%

Generate a structured medical report with:

1. Clinical Findings
2. Impression
3. Recommendations
4. Disclaimer

Keep it professional and concise.
"""

    try:
        gemini = genai.GenerativeModel("models/gemini-3-flash-preview")
        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating report: {str(e)}"

# ─── Routes ───────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files["image"]
        image = Image.open(file.stream)

        # ── Prediction ──
        arr = preprocess_image(image)
        preds = model.predict(arr)[0]

        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        prediction = "PNEUMONIA" if class_idx == 1 else "NORMAL"

        # ── Generate Report ──
        report = generate_report(prediction, confidence)

        # ── Convert image to base64 ──
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "report": report,
            "image": img_base64,
            "timestamp": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Run Server ───────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)