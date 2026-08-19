import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)

MODELS = {}
SCALERS = {}

def load_models():
    for name in ["california-housing", "wine-quality", "diabetes"]:
        model_path = os.path.join(BASE_DIR, name, "model.pkl")
        scaler_path = os.path.join(BASE_DIR, name, "scaler.pkl")
        if os.path.exists(model_path):
            MODELS[name] = joblib.load(model_path)
        if os.path.exists(scaler_path):
            SCALERS[name] = joblib.load(scaler_path)

FEATURES = {
    "california-housing": {
        "name": "California Housing",
        "desc": "Prediccion de precios de viviendas en California",
        "fields": [
            {"id": "longitude", "label": "Longitud", "tag": "°W", "value": "-119.5", "min": "-124.35", "max": "-114.31", "step": "any"},
            {"id": "latitude", "label": "Latitud", "tag": "°N", "value": "35.6", "min": "32.54", "max": "41.95", "step": "any"},
            {"id": "housing_median_age", "label": "Edad Media", "tag": "años", "value": "28", "min": "1", "max": "52", "step": "any"},
            {"id": "total_rooms", "label": "Habitaciones", "tag": "uds", "value": "1500", "min": "2", "max": "39320", "step": "1"},
            {"id": "total_bedrooms", "label": "Dormitorios", "tag": "uds", "value": "300", "min": "1", "max": "6445", "step": "1"},
            {"id": "population", "label": "Poblacion", "tag": "hab", "value": "1200", "min": "3", "max": "35682", "step": "1", "hidden": True},
            {"id": "households", "label": "Hogares", "tag": "uds", "value": "400", "min": "1", "max": "6082", "step": "1", "hidden": True},
            {"id": "median_income", "label": "Ingreso Medio", "tag": "x10k", "value": "4.5", "min": "0.5", "max": "15.0", "step": "any"},
        ],
        "feature_names": ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income"],
        "metrics": {"r2": "0.6139", "mae": "$51,810", "rmse": "$71,133", "r2_pct": "61.4", "mae_pct": "52", "rmse_pct": "71"},
        "algo": "LinearRegression", "records": "20,640", "features_count": "8", "target": "median_house_value",
        "ranges": {
            "longitude": {"min": -124.35, "max": -114.31, "label": "Longitud"},
            "latitude": {"min": 32.54, "max": 41.95, "label": "Latitud"},
            "housing_median_age": {"min": 1, "max": 52, "label": "Edad Media"},
            "total_rooms": {"min": 2, "max": 39320, "label": "Habitaciones"},
            "total_bedrooms": {"min": 1, "max": 6445, "label": "Dormitorios"},
            "median_income": {"min": 0.5, "max": 15.0, "label": "Ingreso Medio"},
            "population": {"min": 3, "max": 35682, "label": "Poblacion"},
            "households": {"min": 1, "max": 6082, "label": "Hogares"},
        }
    },
    "wine-quality": {
        "name": "Wine Quality",
        "desc": "Prediccion de calidad de vinos",
        "fields": [],
        "feature_names": [],
        "metrics": {"r2": "-", "mae": "-", "rmse": "-", "r2_pct": "0", "mae_pct": "0", "rmse_pct": "0"},
        "algo": "Pendiente", "records": "-", "features_count": "-", "target": "-",
        "ranges": {}
    },
    "diabetes": {
        "name": "Diabetes",
        "desc": "Prediccion de progresion de diabetes",
        "fields": [],
        "feature_names": [],
        "metrics": {"r2": "-", "mae": "-", "rmse": "-", "r2_pct": "0", "mae_pct": "0", "rmse_pct": "0"},
        "algo": "Pendiente", "records": "-", "features_count": "-", "target": "-",
        "ranges": {}
    }
}

PROJECTS = ["california-housing", "wine-quality", "diabetes"]

@app.route("/")
def index():
    return render_template("california.html", project="california-housing", projects=PROJECTS, config=FEATURES["california-housing"])

@app.route("/<project>")
def project_page(project):
    if project not in FEATURES:
        return "Proyecto no encontrado", 404
    return render_template(f"{project}.html", project=project, projects=PROJECTS, config=FEATURES[project])

@app.route("/api/predict/<project>", methods=["POST"])
def predict(project):
    if project not in MODELS:
        return jsonify({"error": f"Modelo de '{project}' no disponible aun."}), 400

    config = FEATURES[project]
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se recibieron datos."}), 400

        values = []
        for f in config["feature_names"]:
            if f not in data:
                return jsonify({"error": f"Falta el campo: {f}"}), 400
            val = float(data[f])
            r = config["ranges"].get(f)
            if r and (val < r["min"] or val > r["max"]):
                return jsonify({"error": f"{config['ranges'][f]['label']}: fuera de rango ({r['min']} a {r['max']})"}), 400
            values.append(val)

        X = np.array([values])
        if project in SCALERS:
            X = SCALERS[project].transform(X)
        prediction = MODELS[project].predict(X)[0]
        if prediction < 0:
            prediction = 0

        return jsonify({"prediction": round(prediction, 2)})
    except ValueError:
        return jsonify({"error": "Los valores deben ser numericos."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
