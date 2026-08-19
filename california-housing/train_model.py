import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

print("=" * 60)
print("  ENTRENAMIENTO DEL MODELO - California Housing")
print("=" * 60)

csv_path = os.path.join(os.path.dirname(__file__), "housing.csv")

if not os.path.exists(csv_path):
    print(f"\n[ERROR] No se encontro el archivo '{csv_path}'.")
    print("Descarga el dataset de Kaggle y coloca 'housing.csv' en la carpeta del proyecto.")
    print("URL: https://www.kaggle.com/datasets/camnugent/california-housing-prices")
    exit(1)

print(f"\nCargando dataset desde: {csv_path}")
df = pd.read_csv(csv_path)
print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")

print("\nValores nulos por columna:")
nulls = df.isnull().sum()
print(nulls[nulls > 0] if nulls.sum() > 0 else "No hay valores nulos.")

if df["total_bedrooms"].isnull().sum() > 0:
    mediana = df["total_bedrooms"].median()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(mediana)
    print(f"\nValores nulos en 'total_bedrooms' reemplazados por la mediana: {mediana}")

features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
target = "median_house_value"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDatos de entrenamiento: {X_train.shape[0]} muestras")
print(f"Datos de prueba: {X_test.shape[0]} muestras")

print("\nEntrenando modelo de Regresion Lineal Multiple...")
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n" + "=" * 60)
print("  METRICAS DEL MODELO")
print("=" * 60)
print(f"  R2:  {r2:.4f}")
print(f"  MAE: ${mae:,.2f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: ${rmse:,.2f}")
print("=" * 60)

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
joblib.dump(model, model_path)
print(f"\nModelo guardado en: {model_path}")
print("Entrenamiento completado exitosamente.")
