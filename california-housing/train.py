import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("=" * 60)
print("  FASE A: ANALISIS EXPLORATORIO Y PREPROCESAMIENTO")
print("  California Housing Dataset")
print("=" * 60)

# ── 1. CARGA DE DATOS ──
csv_path = os.path.join(os.path.dirname(__file__), "housing.csv")

if not os.path.exists(csv_path):
    print(f"\n[ERROR] No se encontro '{csv_path}'.")
    exit(1)

df = pd.read_csv(csv_path)
print(f"\nDataset: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Columnas: {list(df.columns)}")

# ── 2. VALORES NULOS ──
print("\n--- 2. VALORES NULOS ---")
nulls = df.isnull().sum()
print(nulls[nulls > 0] if nulls.sum() > 0 else "No hay valores nulos.")

if df["total_bedrooms"].isnull().sum() > 0:
    mediana = df["total_bedrooms"].median()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(mediana)
    print(f"total_bedrooms: {nulls['total_bedrooms']} nulos reemplazados por mediana ({mediana})")

# ── 3. ESTADISTICAS DESCRIPTIVAS ──
print("\n--- 3. ESTADISTICAS DESCRIPTIVAS ---")
features = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income",
]
target = "median_house_value"

desc = df[features + [target]].describe().round(2)
print(desc.to_string())

# ── 4. DETECCION Y TRATAMIENTO DE OUTLIERS (IQR) ──
print("\n--- 4. OUTLIERS (metodo IQR) ---")
before = len(df)
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    if outliers > 0:
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"  {col}: {outliers} outliers recortados a [{lower:.2f}, {upper:.2f}]")

after = len(df)
print(f"\nRegistros antes: {before} | despues: {after} (sin eliminaciones, solo clipping)")

# ── 5. MATRIZ DE CORRELACION ──
print("\n--- 5. MATRIZ DE CORRELACION ---")
corr_cols = features + [target]
corr_matrix = df[corr_cols].corr()
print(corr_matrix.round(3).to_string())

print("\nCorrelaciones con la variable objetivo:")
target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
for feat, val in target_corr.items():
    print(f"  {feat:>25s}: {val:+.3f}")

# ── 6. ANALISIS VIF (Factor de Inflacion de Varianza) ──
print("\n--- 6. VIF (Factor de Inflacion de Varianza) ---")
print("VIF > 10 = multicolinealidad alta | VIF > 5 = preocupante")

from numpy.linalg import inv

X_vif = df[features].values
X_centered = X_vif - X_vif.mean(axis=0)
cov = np.cov(X_centered, rowvar=False)
inv_cov = inv(cov)

vif_data = []
for i, feat in enumerate(features):
    vif = inv_cov[i, i]
    vif_data.append((feat, round(vif, 2)))
    status = "ALTA" if vif > 10 else ("MEDIA" if vif > 5 else "OK")
    print(f"  {feat:>25s}: {vif:>8.2f}  [{status}]")

# ── 7. SEPARACION TRAIN/TEST ──
print("\n--- 7. SEPARACION DE DATOS ---")
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba:        {X_test.shape[0]} muestras")

# ── 8. ESTANDARIZACION (StandardScaler) ──
print("\n--- 8. ESTANDARIZACION (StandardScaler) ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaler ajustado con datos de entrenamiento.")
print("Medias originales vs estandarizadas:")
for i, feat in enumerate(features):
    print(f"  {feat:>25s}: media={X_train[feat].mean():>10.2f}  -> escalada~0")

# ── 9. ENTRENAMIENTO ──
print("\n--- 9. ENTRENAMIENTO DEL MODELO ---")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Modelo LinearRegression entrenado.")

# ── 10. EVALUACION ──
print("\n--- 10. EVALUACION ---")
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"  R2 Score: {r2:.4f}")
print(f"  MAE:      ${mae:,.2f}")
print(f"  MSE:      {mse:,.2f}")
print(f"  RMSE:     ${rmse:,.2f}")

# ── 11. COEFICIENTES DEL MODELO ──
print("\n--- 11. COEFICIENTES ---")
coef_df = pd.DataFrame({
    "Variable": features,
    "Coeficiente": model.coef_.round(2)
}).sort_values("Coeficiente", key=abs, ascending=False)

print(coef_df.to_string(index=False))
print(f"\nIntercepto: {model.intercept_:,.2f}")

# ── GUARDADO ──
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"\nModelo guardado: {model_path}")
print(f"Scaler guardado: {scaler_path}")
print("=" * 60)
print("  FASE A COMPLETADA")
print("=" * 60)
