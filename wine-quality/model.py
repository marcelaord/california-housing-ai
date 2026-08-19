import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "WineQT.csv")
TARGET = "quality"
RANDOM_STATE = 42
LINEAR_DEGREE = 1
POLYNOMIAL_DEGREES = [2, 3, 4]

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

FEATURE_LABELS = {
    "fixed acidity": "Acidez fija (g/dm³)",
    "volatile acidity": "Acidez volátil (g/dm³)",
    "citric acid": "Ácido cítrico (g/dm³)",
    "residual sugar": "Azúcar residual (g/dm³)",
    "chlorides": "Cloruros (g/dm³)",
    "free sulfur dioxide": "Dióxido de azufre libre (mg/dm³)",
    "total sulfur dioxide": "Dióxido de azufre total (mg/dm³)",
    "density": "Densidad (g/cm³)",
    "pH": "pH",
    "sulphates": "Sulfatos (g/dm³)",
    "alcohol": "Alcohol (% vol.)",
}


def _metrics(y_true, y_pred):
    return {
        "R²": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def _build_dict(metrics):
    return {
        "R²": round(float(metrics["R²"]), 4),
        "RMSE": round(float(metrics["RMSE"]), 4),
        "MAE": round(float(metrics["MAE"]), 4),
    }


class WineSimulator:
    """Entrena Regresión Lineal Múltiple y Polinomial usando únicamente datos reales de WineQT.csv."""

    def __init__(self, data_path=DATA_PATH, random_state=RANDOM_STATE):
        self.df = pd.read_csv(data_path)
        self.X = self.df[FEATURES]
        self.y = self.df[TARGET]
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )
        self._fitted = {}
        self._train_all()

    def _train_all(self):
        for degree in [LINEAR_DEGREE] + POLYNOMIAL_DEGREES:
            self._fitted[degree] = self._train_degree(degree)

    def _train_degree(self, degree):
        if degree == LINEAR_DEGREE:
            model = LinearRegression()
            model.fit(self.X_train, self.y_train)
            return {
                "degree": degree,
                "type": "lineal",
                "model": model,
                "scaler": None,
                "poly": None,
                "train_metrics": _build_dict(_metrics(self.y_train, model.predict(self.X_train))),
                "test_metrics": _build_dict(_metrics(self.y_test, model.predict(self.X_test))),
            }
        scaler = StandardScaler().fit(self.X_train)
        Xs_train = scaler.transform(self.X_train)
        Xs_test = scaler.transform(self.X_test)
        poly = PolynomialFeatures(degree=degree, include_bias=False).fit(Xs_train)
        Xp_train = poly.transform(Xs_train)
        Xp_test = poly.transform(Xs_test)
        model = RidgeCV(alphas=np.logspace(-2, 5, 40), cv=5)
        model.fit(Xp_train, self.y_train)
        return {
            "degree": degree,
            "type": "polinomial",
            "model": model,
            "scaler": scaler,
            "poly": poly,
            "train_metrics": _build_dict(_metrics(self.y_train, model.predict(Xp_train))),
            "test_metrics": _build_dict(_metrics(self.y_test, model.predict(Xp_test))),
        }

    def predict_bulk(self, X, degree=LINEAR_DEGREE):
        fitted = self._fitted[degree]
        XX = X
        if fitted["scaler"] is not None:
            XX = fitted["scaler"].transform(XX)
        if fitted["poly"] is not None:
            XX = fitted["poly"].transform(XX)
        return fitted["model"].predict(XX)

    def predict(self, features, degree=LINEAR_DEGREE):
        row = pd.DataFrame([features])[FEATURES]
        return float(self.predict_bulk(row, degree)[0])

    def response_curves(self, features, vary_feature, degrees, n_points=60):
        """Predicción de los modelos dados a lo largo del rango real de una variable
        (los demás features quedan fijos en `features`)."""
        lo, hi = self.feature_ranges()[vary_feature]
        x = np.linspace(lo, hi, n_points)
        grid = pd.DataFrame({vary_feature: x})
        for f, value in features.items():
            grid[f] = value
        out = {vary_feature: x}
        for deg in degrees:
            out[f"pred_{deg}"] = self.predict_bulk(grid[FEATURES], deg)
        return pd.DataFrame(out)

    def available_degrees(self):
        return sorted(self._fitted.keys())

    def fitted(self, degree):
        return self._fitted[degree]

    def feature_ranges(self):
        return {f: (float(self.df[f].min()), float(self.df[f].max())) for f in FEATURES}

    def n_polynomial_terms(self):
        """Número de términos (coeficientes) generados para cada grado polinomial sobre features estandarizadas."""
        out = {}
        for degree in POLYNOMIAL_DEGREES:
            fitted = self._fitted[degree]
            coef = np.asarray(fitted["model"].coef_)
            out[degree] = int(coef.shape[0]) + 1
        return out

    def equation(self, degree=LINEAR_DEGREE):
        fitted = self._fitted[degree]
        model = fitted["model"]
        b0 = float(model.intercept_)
        if degree == LINEAR_DEGREE:
            terms = [f"{float(c):.4f}·{FEATURE_LABELS[f]}" for f, c in zip(FEATURES, model.coef_)]
            return f"calidad = {b0:.4f} + " + " + ".join(terms)
        coefs = np.asarray(model.coef_)
        names = fitted["poly"].get_feature_names_out(FEATURES)
        ordered = np.argsort(-np.abs(coefs))[:8]
        parts = [f"{b0:.4f}"]
        for i in ordered:
            parts.append(f"({coefs[i]:+.4f})·{names[i].replace(' ', '·')}")
        n_total = int(coefs.shape[0]) + 1
        alpha = float(model.alpha_)
        return (
            f"calidad = {b0:.4f} + Σ de {n_total - 1} términos polinomiales de grado {degree} "
            f"(sobre features estandarizadas).\n"
            f"Regularización Ridge con λ = {alpha:.2f} (seleccionada por cross-validation).\n"
            f"Top {len(ordered)} términos por |coeficiente|:\n  "
            + "\n  ".join(parts)
        )

    def comparison_table(self):
        rows = []
        for degree in self.available_degrees():
            fitted = self._fitted[degree]
            n_terms = fitted["model"].coef_.shape[0] + 1
            if degree == LINEAR_DEGREE:
                nombre = "Regresión Lineal Múltiple"
                reg = "Sin regularizar"
            else:
                nombre = f"Polinomial (grado {degree})"
                reg = f"Ridge λ={float(fitted['model'].alpha_):.2f}"
            rows.append(
                {
                    "Modelo": nombre,
                    "Regularización": reg,
                    "N° de términos": int(n_terms),
                    "R² (train)": fitted["train_metrics"]["R²"],
                    "R² (test)": fitted["test_metrics"]["R²"],
                    "RMSE (test)": fitted["test_metrics"]["RMSE"],
                    "MAE (test)": fitted["test_metrics"]["MAE"],
                }
            )
        return pd.DataFrame(rows)