import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
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

# ══════════════════════════════════════════════
#  FASE B: MODELANOMIENTO Y COMPARATIVA
# ══════════════════════════════════════════════

# ── 9. MODELO 1: REGRESION LINEAL MULTIPLE ──
print("\n--- 9. MODELO 1: REGRESION LINEAL MULTIPLE ---")
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
print("LinearRegression entrenado.")

y_pred_lr = model_lr.predict(X_test_scaled)

r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

print(f"\n  R2 Score: {r2_lr:.4f}")
print(f"  MAE:      ${mae_lr:,.2f}")
print(f"  MSE:      {mse_lr:,.2f}")
print(f"  RMSE:     ${rmse_lr:,.2f}")

# ── 10. MODELO 2: REGRESION POLINOMIAL (grado 2) ──
print("\n--- 10. MODELO 2: REGRESION POLINOMIAL (grado 2) ---")
poly_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])
poly_pipeline.fit(X_train, y_train)
print("PolynomialFeatures(degree=2) + LinearRegression entrenado.")

y_pred_poly = poly_pipeline.predict(X_test)

r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)

print(f"\n  R2 Score: {r2_poly:.4f}")
print(f"  MAE:      ${mae_poly:,.2f}")
print(f"  MSE:      {mse_poly:,.2f}")
print(f"  RMSE:     ${rmse_poly:,.2f}")

# ── 11. COMPARATIVA DE MODELOS ──
print("\n" + "=" * 60)
print("  COMPARATIVA DE MODELOS")
print("=" * 60)
print(f"{'Metrica':<15} {'Lineal':>18} {'Polinomial':>18} {'Mejor':>12}")
print("-" * 63)

winner_count = {"Lineal": 0, "Polinomial": 0}

metrics = [
    ("R2 Score", r2_lr, r2_poly, True),
    ("MAE", mae_lr, mae_poly, False),
    ("MSE", mse_lr, mse_poly, False),
    ("RMSE", rmse_lr, rmse_poly, False),
]

for name, val_lr, val_poly, higher_better in metrics:
    if higher_better:
        best = "Lineal" if val_lr >= val_poly else "Polinomial"
    else:
        best = "Lineal" if val_lr <= val_poly else "Polinomial"
    winner_count[best] += 1

    if name == "R2 Score":
        lr_str = f"{val_lr:.4f}"
        poly_str = f"{val_poly:.4f}"
    else:
        lr_str = f"${val_lr:,.2f}"
        poly_str = f"${val_poly:,.2f}"

    print(f"{name:<15} {lr_str:>18} {poly_str:>18} {best:>12}")

print("-" * 63)
print(f"\nMarcador: Lineal {winner_count['Lineal']} - {winner_count['Polinomial']} Polinomial")

mejor = max(winner_count, key=winner_count.get)
print(f"Mejor modelo global: {mejor}")

# ── 12. COEFICIENTES DEL MEJOR MODELO ──
print(f"\n--- 12. COEFICIENTES ({mejor}) ---")
if mejor == "Lineal":
    coef_df = pd.DataFrame({
        "Variable": features,
        "Coeficiente": model_lr.coef_.round(2)
    }).sort_values("Coeficiente", key=abs, ascending=False)
    print(coef_df.to_string(index=False))
    print(f"\nIntercepto: {model_lr.intercept_:,.2f}")
else:
    poly_feature_names = poly_pipeline.named_steps["poly"].get_feature_names_out(features)
    coefs = poly_pipeline.named_steps["lr"].coef_
    coef_df = pd.DataFrame({
        "Variable": poly_feature_names,
        "Coeficiente": coefs.round(2)
    })
    top = coef_df.reindex(coef_df["Coeficiente"].abs().sort_values(ascending=False).index).head(15)
    print(f"(Mostrando top 15 de {len(coef_df)} coeficientes)")
    print(top.to_string(index=False))
    print(f"\nIntercepto: {poly_pipeline.named_steps['lr'].intercept_:,.2f}")

# ── GUARDADO ──
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
poly_path = os.path.join(os.path.dirname(__file__), "model_poly.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

joblib.dump(model_lr, model_path)
joblib.dump(poly_pipeline, poly_path)
joblib.dump(scaler, scaler_path)

print(f"\nModelo Lineal guardado:  {model_path}")
print(f"Modelo Polinomial guardado: {poly_path}")
print(f"Scaler guardado: {scaler_path}")
print("=" * 60)
print("  FASE A + B COMPLETADAS")
print("=" * 60)
