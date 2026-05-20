from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import math
import os
import json

app = Flask(__name__)
CORS(app)

# =========================
# FILE PATHS
# =========================

BASE_DIR = os.path.dirname(__file__)

csv_path = os.path.join(BASE_DIR, "trend_weights.csv")
json_path = os.path.join(BASE_DIR, "nit_data.json")

# =========================
# LOAD CSV DATA
# =========================

try:
    trends = pd.read_csv(csv_path)

    if 'Gender' in trends.columns:
        trends['Gender'] = trends['Gender'].fillna('').astype(str)

    for col in ['mean', 'std', 'NIRF_Rank', 'Priority']:
        if col in trends.columns:
            trends[col] = pd.to_numeric(trends[col], errors='coerce')

    print("✅ Trend weights loaded")

except Exception as e:
    print(f"❌ CSV Load Error: {e}")
    trends = pd.DataFrame()

# =========================
# LOAD JSON DATA
# =========================

try:
    with open(json_path, "r") as f:
        nit_data = json.load(f)

    print("✅ JSON loaded")

except Exception as e:
    print(f"❌ JSON Load Error: {e}")
    nit_data = {}

# =========================
# HELPER
# =========================

def get_prediction_meta(prob, std):

    if prob > 85:
        status = "Safe"
    elif prob > 60:
        status = "Moderate"
    elif prob > 35:
        status = "Risky"
    else:
        status = "Dream"

    if std < 450:
        confidence = "High"
    elif std < 950:
        confidence = "Medium"
    else:
        confidence = "Low"

    return status, confidence

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return jsonify({
        "message": "NIT Predictor API Running"
    })

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        user_rank = int(data["rank"])
        category = data["category"]
        gender = data["gender"]

        if trends.empty:
            return jsonify([])

        filtered = trends[
            (trends["Category"] == category) &
            (trends["Gender"].str.contains(gender, case=False, na=False))
        ].copy()

        results = []

        for _, row in filtered.iterrows():

            mean = row["mean"]

            std = row["std"]

            if pd.isna(std) or std <= 0:
                std = 500

            z = (mean - user_rank) / std

            probability = round(
                (1 / (1 + math.exp(-1.4 * z))) * 100,
                1
            )

            status, confidence = get_prediction_meta(
                probability,
                std
            )

            results.append({
                "institute": row.get("Institute", ""),
                "branch": row.get("Branch", ""),
                "nirf": int(row.get("NIRF_Rank", 999)),
                "probability": probability,
                "status": status,
                "confidence": confidence,
                "priority": int(row.get("Priority", 999)),
                "quota": row.get("Quota", "")
            })

        results.sort(
            key=lambda x: (
                x["nirf"],
                x["priority"]
            )
        )

        return jsonify(results)

    except Exception as e:

        print(f"❌ Prediction Error: {e}")

        return jsonify({
            "error": str(e)
        }), 500

# =========================
# LOCAL RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True)