import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  FASE A: ANALISIS EXPLORATORIO Y PREPROCESAMIENTO")
print("  Diabetes Dataset (sklearn)")
print("=" * 60)

# ── 1. CARGA ──
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["progression"] = diabetes.target

print(f"\nDataset: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Features ({len(diabetes.feature_names)}): {diabetes.feature_names}")
print(f"Target: progression (progresion de diabetes, rango {diabetes.target.min():.0f}-{diabetes.target.max():.0f})")

# ── 2. LIMPIEZA ──
print("\n--- 2. LIMPIEZA DE DATOS ---")
nulls = df.isnull().sum().sum()
dups = df.duplicated().sum()
print(f"Valores nulos: {nulls}")
print(f"Registros duplicados: {dups}")
print("Dataset pre-procesado por sklearn (sin nulos ni duplicados).")
print("Las variables ya vienen estandarizadas (mean-centered, scaled by std).")

features = list(diabetes.feature_names)
target = "progression"

# ── 3. ESTADISTICAS ──
print("\n--- 3. ESTADISTICAS DESCRIPTIVAS ---")
print(df.describe().round(4).to_string())

# ── 4. OUTLIERS (IQR) ──
print("\n--- 4. OUTLIERS (metodo IQR) ---")
total_outliers = 0
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    if n_out > 0:
        print(f"  {col}: {n_out} outliers")
        total_outliers += n_out
print(f"\nTotal outliers: {total_outliers}")
print("Decision: Se conservan (valores clinicos legitimos).")

# ── 5. CORRELACION ──
print("\n--- 5. MATRIZ DE CORRELACION ---")
corr = df[features + [target]].corr()
print(f"\nCorrelaciones con '{target}':")
corr_target = corr[target].drop(target).sort_values(key=abs, ascending=False)
for feat, val in corr_target.items():
    sign = "+" if val > 0 else "-"
    bar = "#" * int(abs(val) * 30)
    print(f"  {feat:<8s} {sign}{abs(val):.3f}  {bar}")

print("\nPares con |r| > 0.5:")
seen = set()
for i, f1 in enumerate(features):
    for j, f2 in enumerate(features):
        if i < j:
            r = corr.loc[f1, f2]
            if abs(r) > 0.5:
                pair = tuple(sorted([f1, f2]))
                if pair not in seen:
                    seen.add(pair)
                    print(f"  {f1} <-> {f2}: {r:+.3f}")

# ── 6. VIF ──
print("\n--- 6. VIF (Factor de Inflacion de Varianza) ---")
from sklearn.linear_model import LinearRegression as _LR
vif_vals = []
for feat in features:
    other = [f for f in features if f != feat]
    r2_vif = _LR().fit(df[other], df[feat]).score(df[other], df[feat])
    vif_vals.append(1.0 / (1.0 - r2_vif) if r2_vif < 1.0 else float("inf"))
print("VIF > 10 = multicolinealidad alta | VIF > 5 = preocupante")
for feat, vif in zip(features, vif_vals):
    status = "[!]" if vif > 10 else "[warn]" if vif > 5 else "[OK]"
    print(f"  {feat:<8s} {vif:>8.2f}  {status}")

# ── 7. SEPARACION ──
print("\n--- 7. SEPARACION DE DATOS ---")
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba:        {X_test.shape[0]} muestras")

# ── 8. ESCALADO ──
print("\n--- 8. ESTANDARIZACION (StandardScaler) ---")
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaler ajustado con datos de entrenamiento.")

# ══════════════════════════════════════════════
#  FASE B: MODELAMIENTO Y COMPARATIVA
# ══════════════════════════════════════════════

# ── 9. MODELO 1: REGRESION LINEAL MULTIPLE ──
print("\n--- 9. MODELO 1: REGRESION LINEAL MULTIPLE ---")
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
print("LinearRegression entrenado.")

y_pred_lr = model_lr.predict(X_test_scaled)
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print(f"\n  R2 Score: {r2_lr:.4f}")
print(f"  MAE:      {mae_lr:.4f}")
print(f"  RMSE:     {rmse_lr:.4f}")

# ── 10. MODELO 2: REGRESION POLINOMIAL + RIDGE ──
print("\n--- 10. MODELO 2: REGRESION POLINOMIAL (grado 2) + Ridge ---")
poly_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-2, 5, 40), cv=5))
])
poly_pipeline.fit(X_train, y_train)
print("PolynomialFeatures(degree=2) + StandardScaler + RidgeCV entrenado.")
print(f"Ridge alpha optimo: {poly_pipeline.named_steps['ridge'].alpha_:.2f}")

y_pred_poly = poly_pipeline.predict(X_test)
r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))

print(f"\n  R2 Score: {r2_poly:.4f}")
print(f"  MAE:      {mae_poly:.4f}")
print(f"  RMSE:     {rmse_poly:.4f}")

# ── 11. POLINOMIOS GRADO 3 ──
print("\n--- 11. POLINOMIO GRADO 3 (demostracion de sobreajuste) ---")
pipe3 = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-2, 5, 40), cv=5))
])
pipe3.fit(X_train, y_train)
y_pred_3 = pipe3.predict(X_test)
r2_3 = r2_score(y_test, y_pred_3)
rmse_3 = np.sqrt(mean_squared_error(y_test, y_pred_3))
n_terms_3 = pipe3.named_steps["poly"].transform(scaler.transform(X_train)).shape[1]
print(f"  Grado 3: {n_terms_3} terminos, R2={r2_3:.4f}, RMSE={rmse_3:.4f}, alpha={pipe3.named_steps['ridge'].alpha_:.2f}")

# ── 12. COMPARATIVA ──
print("\n" + "=" * 60)
print("  COMPARATIVA DE MODELOS")
print("=" * 60)
print(f"{'Metrica':<12} {'Lineal':>12} {'Polinomial G2':>14} {'Mejor':>14}")
print("-" * 52)

winner_count = {"Lineal": 0, "Polinomial": 0}
metrics = [
    ("R2 Score", r2_lr, r2_poly, True),
    ("MAE", mae_lr, mae_poly, False),
    ("RMSE", rmse_lr, rmse_poly, False),
]
for name, val_lr, val_poly, higher_better in metrics:
    if higher_better:
        best = "Lineal" if val_lr >= val_poly else "Polinomial"
    else:
        best = "Lineal" if val_lr <= val_poly else "Polinomial"
    winner_count[best] += 1
    print(f"{name:<12} {val_lr:>12.4f} {val_poly:>14.4f} {best:>14}")

print("-" * 52)
print(f"\nMarcador: Lineal {winner_count['Lineal']} - {winner_count['Polinomial']} Polinomial")
mejor = max(winner_count, key=winner_count.get)
print(f"Mejor modelo global: {mejor}")

# ── 13. COEFICIENTES ──
print(f"\n--- 13. COEFICIENTES (Lineal) ---")
coef_df = pd.DataFrame({
    "Variable": features,
    "Coeficiente": model_lr.coef_.round(4)
}).sort_values("Coeficiente", key=abs, ascending=False)
print(coef_df.to_string(index=False))
print(f"\nIntercepto: {model_lr.intercept_:.4f}")

print(f"\n--- 14. COEFICIENTES (Polinomial G2, Top 10) ---")
poly_feat_names = poly_pipeline.named_steps["poly"].get_feature_names_out(features)
poly_coefs = poly_pipeline.named_steps["ridge"].coef_
top_idx = np.argsort(-np.abs(poly_coefs))[:10]
for i in top_idx:
    print(f"  {poly_feat_names[i]:<25s} {poly_coefs[i]:>10.4f}")
print(f"\nIntercepto: {poly_pipeline.named_steps['ridge'].intercept_:.4f}")

# ── GUARDADO ──
model_lr_path = os.path.join(BASE, "model.pkl")
model_poly_path = os.path.join(BASE, "model_poly.pkl")
scaler_path = os.path.join(BASE, "scaler.pkl")

joblib.dump(model_lr, model_lr_path)
joblib.dump(poly_pipeline, model_poly_path)
joblib.dump(scaler, scaler_path)

print(f"\nModelo Lineal guardado:    {model_lr_path}")
print(f"Modelo Polinomial guardado: {model_poly_path}")
print(f"Scaler guardado:           {scaler_path}")
print("=" * 60)
print("  FASE A + B COMPLETADAS - DIABETES")
print("=" * 60)
