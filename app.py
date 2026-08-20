import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)

MODELS = {}
SCALERS = {}
MODEL_TYPES = {}

def load_models():
    for name in ["california-housing", "wine-quality", "diabetes"]:
        model_path = os.path.join(BASE_DIR, name, "model.pkl")
        poly_path = os.path.join(BASE_DIR, name, "model_poly.pkl")
        scaler_path = os.path.join(BASE_DIR, name, "scaler.pkl")
        try:
            if os.path.exists(poly_path):
                MODELS[name] = joblib.load(poly_path)
                MODEL_TYPES[name] = "poly"
            elif os.path.exists(model_path):
                MODELS[name] = joblib.load(model_path)
                MODEL_TYPES[name] = "linear"
            if os.path.exists(scaler_path):
                SCALERS[name] = joblib.load(scaler_path)
        except Exception as e:
            print(f"Error cargando modelo '{name}': {e}")

load_models()

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
        "desc": "Prediccion de calidad de vinos tintos",
        "fields": [],
        "feature_names": ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],
        "metrics": {"r2": "0.3360", "mae": "0.4700", "rmse": "0.6079", "r2_pct": "33.6", "mae_pct": "47", "rmse_pct": "61"},
        "algo": "Polinomial G2 + Ridge", "records": "1,143", "features_count": "11", "target": "quality",
        "ranges": {
            "fixed acidity": {"min": 4.6, "max": 15.9, "label": "Acidez fija"},
            "volatile acidity": {"min": 0.12, "max": 1.58, "label": "Acidez volatil"},
            "citric acid": {"min": 0.0, "max": 1.0, "label": "Acido citrico"},
            "residual sugar": {"min": 0.9, "max": 15.5, "label": "Azucar residual"},
            "chlorides": {"min": 0.012, "max": 0.611, "label": "Cloruros"},
            "free sulfur dioxide": {"min": 1, "max": 68, "label": "SO2 libre"},
            "total sulfur dioxide": {"min": 6, "max": 289, "label": "SO2 total"},
            "density": {"min": 0.9901, "max": 1.0037, "label": "Densidad"},
            "pH": {"min": 2.74, "max": 4.01, "label": "pH"},
            "sulphates": {"min": 0.33, "max": 2.0, "label": "Sulfatos"},
            "alcohol": {"min": 8.4, "max": 14.9, "label": "Alcohol"},
        }
    },
    "diabetes": {
        "name": "Diabetes",
        "desc": "Prediccion de progresion de diabetes",
        "fields": [],
        "feature_names": ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"],
        "metrics": {"r2": "0.5008", "mae": "41.3405", "rmse": "51.4267", "r2_pct": "50.1", "mae_pct": "41", "rmse_pct": "51"},
        "algo": "Polinomial G2 + Ridge", "records": "442", "features_count": "10", "target": "progression",
        "ranges": {
            "age": {"min": -0.1072, "max": 0.1107, "label": "Edad"},
            "sex": {"min": -0.0446, "max": 0.0507, "label": "Sexo"},
            "bmi": {"min": -0.0903, "max": 0.1706, "label": "IMC"},
            "bp": {"min": -0.1124, "max": 0.1320, "label": "Presion Arterial"},
            "s1": {"min": -0.1268, "max": 0.1539, "label": "Colesterol Total"},
            "s2": {"min": -0.1156, "max": 0.1988, "label": "LDL"},
            "s3": {"min": -0.1023, "max": 0.1812, "label": "HDL"},
            "s4": {"min": -0.0764, "max": 0.1852, "label": "Relacion Col/HDL"},
            "s5": {"min": -0.1261, "max": 0.1336, "label": "Trigliceridos"},
            "s6": {"min": -0.1378, "max": 0.1356, "label": "Glucosa"}
        }
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
        if MODEL_TYPES.get(project) == "linear" and project in SCALERS:
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
