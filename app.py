import os
import json
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load artefacts once at startup ───────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model    = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))

with open(os.path.join(MODEL_DIR, "unique_values.json")) as f:
    unique_values = json.load(f)

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    metrics = json.load(f)

with open(os.path.join(MODEL_DIR, "year_range.json")) as f:
    year_range = json.load(f)


# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_encode(encoder, value):
    """Encode a label; return -1 if unseen (fallback)."""
    classes = list(encoder.classes_)
    if value in classes:
        return encoder.transform([value])[0]
    # Nearest match (lowercased) fallback
    value_lower = value.lower()
    for cls in classes:
        if cls.lower() == value_lower:
            return encoder.transform([cls])[0]
    return 0  # default to first class if truly unseen


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/options")
def options():
    return jsonify({
        "states":    unique_values.get("State", []),
        "districts": unique_values.get("District", []),
        "crops":     unique_values.get("Crop", []),
        "seasons":   unique_values.get("Season", []),
        "year_min":  year_range["min"],
        "year_max":  year_range["max"],
    })


@app.route("/api/metrics")
def get_metrics():
    return jsonify(metrics)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        state      = str(data["state"]).strip()
        district   = str(data["district"]).strip()
        crop       = str(data["crop"]).strip()
        season     = str(data["season"]).strip()
        crop_year  = float(data["crop_year"])
        area       = float(data["area"])
        production = float(data["production"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    if area <= 0 or production <= 0:
        return jsonify({"error": "Area and Production must be positive numbers."}), 400

    # Encode
    state_enc    = safe_encode(encoders["State"],    state)
    district_enc = safe_encode(encoders["District"], district)
    crop_enc     = safe_encode(encoders["Crop"],     crop)
    season_enc   = safe_encode(encoders["Season"],   season)

    features = np.array([[state_enc, district_enc, crop_enc, season_enc,
                           crop_year, area, production]])

    prediction = model.predict(features)[0]

    return jsonify({
        "yield":  round(float(prediction), 4),
        "unit":   "tonnes / hectare",
        "inputs": {
            "state":      state,
            "district":   district,
            "crop":       crop,
            "season":     season,
            "crop_year":  int(crop_year),
            "area":       area,
            "production": production,
        }
    })


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
