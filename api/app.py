from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import math
import os

app = Flask(__name__)
CORS(app)

# Load the processed weights (path relative to this file)
csv_path = os.path.join(os.path.dirname(__file__), "trend_weights.csv")
try:
    trends = pd.read_csv(csv_path)

    # Coerce expected columns to safe dtypes and fill na for string ops
    if 'Gender' in trends.columns:
        trends['Gender'] = trends['Gender'].fillna('').astype(str)
    else:
        trends['Gender'] = ''

    for col in ['mean', 'std', 'NIRF_Rank', 'Priority']:
        if col in trends.columns:
            trends[col] = pd.to_numeric(trends[col], errors='coerce')
        else:
            trends[col] = pd.NA

    print("✅ 2026 Admissions Guide Engine Online (Port 5001)")
except FileNotFoundError:
    print(f"❌ {csv_path} missing! Please run engine.py first.")
    trends = pd.DataFrame()
except Exception as exc:
    print(f"❌ Error loading trend_weights.csv: {exc}")
    trends = pd.DataFrame()

def get_prediction_meta(prob, std):
    """Categorizes the results based on probability and data quality."""
    if prob > 85: status = "Safe"
    elif prob > 60: status = "Moderate"
    elif prob > 35: status = "Risky"
    else: status = "Dream"

    # Confidence check based on volatility (std)
    if std < 450: confidence = "High"
    elif std < 950: confidence = "Medium"
    else: confidence = "Low (Highly Volatile)"

    return status, confidence

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        u_rank = int(data['rank'])
        
        # Filter based on Category and Gender (safe string contains)
        if trends.empty:
            return jsonify([])

        mask = (trends.get('Category') == data.get('category')) & \
               (trends['Gender'].str.contains(str(data.get('gender', '')), case=False, na=False))
        eligible = trends[mask].copy()

        # Drop rows without a numeric mean (can't predict) and ensure numeric columns
        eligible = eligible[eligible['mean'].notna()]

        results = []
        for _, row in eligible.iterrows():
            # SAFETY: Final check to prevent ZeroDivisionError and handle NaNs
            current_std = row['std'] if (pd.notna(row['std']) and row['std'] > 0) else 500

            # Probability Math (Logistic/Sigmoid Function)
            z = (row['mean'] - u_rank) / current_std
            prob = round((1 / (1 + math.exp(-1.4 * z))) * 100, 1)

            status, confidence = get_prediction_meta(prob, current_std)

            # Safe casts with fallbacks
            try:
                nirf_val = int(row['NIRF_Rank']) if pd.notna(row['NIRF_Rank']) else 99999
            except Exception:
                nirf_val = 99999

            try:
                priority_val = int(row['Priority']) if pd.notna(row['Priority']) else 99999
            except Exception:
                priority_val = 99999

            results.append({
                "institute": row.get('Institute', ''),
                "branch": row.get('Branch', ''),
                "nirf": nirf_val,
                "probability": prob,
                "status": status,
                "confidence": confidence,
                "priority": priority_val,
                "quota": row.get('Quota', '')
            })

        # Sort: NIRF Rank first, then Branch Priority
        results.sort(key=lambda x: (x['nirf'], x['priority']))
        return jsonify(results)

    except Exception as e:
        print(f"❌ Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Running on Port 5001 to avoid Mac system conflicts
    app.run(debug=True, port=5001)