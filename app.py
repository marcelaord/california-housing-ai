import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"No se encontro '{model_path}'. Ejecuta 'python train_model.py' primero."
    )

model = joblib.load(model_path)

FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se recibieron datos."}), 400

        values = []
        for f in FEATURES:
            if f not in data:
                return jsonify({"error": f"Falta el campo: {f}"}), 400
            val = float(data[f])
            values.append(val)

        X = np.array([values])
        prediction = model.predict(X)[0]

        if prediction < 0:
            prediction = 0

        return jsonify({"prediction": round(prediction, 2)})
    except ValueError:
        return jsonify({"error": "Los valores deben ser numericos."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
