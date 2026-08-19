import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Predicción Diabetes", page_icon="🏥", layout="wide")

# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; text-align: center; color: #1a1a2e; margin-bottom: 0; }
    .sub-header  { font-size: 1rem; color: #6c757d; text-align: center; margin-top: 0; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1.2rem; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .metric-card p  { margin: 0; font-size: 1.6rem; font-weight: 700; }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 16px; padding: 2rem; text-align: center; color: white;
        box-shadow: 0 6px 20px rgba(17,153,142,0.4);
    }
    .prediction-box h2 { margin: 0; font-size: 1rem; opacity: 0.9; }
    .prediction-box p  { margin: 0.3rem 0 0 0; font-size: 2.4rem; font-weight: 800; }
    .section-divider { border-top: 2px solid #f0f2f6; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Cargar dataset ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target
    return df

df_raw = load_data()

# ── Feature engineering y modelos ────────────────────────────────────────────
df = df_raw.dropna().copy()
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

# ── Mapeo de valores reales al espacio escalado ──────────────────────────────
# Rangos clínicos reales (valores que ve el usuario)
REAL_RANGES = {
    "age": {"min": 20,   "max": 80,   "default": 50,   "step": 1,    "unit": "años",      "label": "Edad"},
    "sex": {"min": 0,    "max": 1,    "default": 0,     "step": 1,    "unit": "",           "label": "Sexo"},
    "bmi": {"min": 15.0, "max": 50.0, "default": 25.0,  "step": 0.1,  "unit": "kg/m²",     "label": "IMC"},
    "bp":  {"min": 60,   "max": 200,  "default": 100,   "step": 1,    "unit": "mmHg",      "label": "Presión Arterial"},
    "s1":  {"min": 100,  "max": 400,  "default": 200,   "step": 1,    "unit": "mg/dL",     "label": "Colesterol Total"},
    "s2":  {"min": 50,   "max": 300,  "default": 130,   "step": 1,    "unit": "mg/dL",     "label": "LDL (Colesterol malo)"},
    "s3":  {"min": 20,   "max": 100,  "default": 50,    "step": 1,    "unit": "mg/dL",     "label": "HDL (Colesterol bueno)"},
    "s4":  {"min": 1.0,  "max": 10.0, "default": 4.0,   "step": 0.1,  "unit": "",           "label": "Relación Colesterol/HDL"},
    "s5":  {"min": 3.0,  "max": 7.0,  "default": 4.5,   "step": 0.05, "unit": "log(mg/dL)","label": "Triglicéridos (log)"},
    "s6":  {"min": 50,   "max": 300,  "default": 100,   "step": 1,    "unit": "mg/dL",     "label": "Glucosa en Sangre"},
}

# Rangos reales del dataset escalado (calculados del dataset)
SCALED_RANGES = {
    "age": {"min": -0.1072, "max": 0.1107},
    "sex": {"min": -0.0446, "max": 0.0507},
    "bmi": {"min": -0.0903, "max": 0.1706},
    "bp":  {"min": -0.1124, "max": 0.1320},
    "s1":  {"min": -0.1268, "max": 0.1539},
    "s2":  {"min": -0.1156, "max": 0.1988},
    "s3":  {"min": -0.1023, "max": 0.1812},
    "s4":  {"min": -0.0764, "max": 0.1852},
    "s5":  {"min": -0.1261, "max": 0.1336},
    "s6":  {"min": -0.1378, "max": 0.1356},
}

def real_to_scaled(feature, real_value):
    """Convierte un valor real clínico al espacio escalado del modelo."""
    r = REAL_RANGES[feature]
    s = SCALED_RANGES[feature]
    # Mapeo lineal del rango real al rango escalado
    proportion = (real_value - r["min"]) / (r["max"] - r["min"])
    return s["min"] + proportion * (s["max"] - s["min"])

# Descripciones de variables
VAR_DESCRIPTIONS = {
    "age": "Edad del paciente en años (rango: 20-80)",
    "sex": "Sexo del paciente (Masculino / Femenino)",
    "bmi": "Índice de Masa Corporal: peso(kg) / altura(m)² (rango: 15-50)",
    "bp": "Presión arterial diastólica promedio en mmHg (rango: 60-200)",
    "s1": "Colesterol sérico total en mg/dL (rango: 100-400)",
    "s2": "Lipoproteínas de baja densidad (LDL) en mg/dL (rango: 50-300)",
    "s3": "Lipoproteínas de alta densidad (HDL) en mg/dL (rango: 20-100)",
    "s4": "Relación colesterol total / HDL (rango: 1-10)",
    "s5": "Logaritmo de triglicéridos séricos (rango: 3-7)",
    "s6": "Nivel de glucosa en sangre en mg/dL (rango: 50-300)",
}

FEATURE_NAMES = {
    "age": "Edad", "sex": "Sexo", "bmi": "IMC", "bp": "Presión Arterial",
    "s1": "Colesterol Total", "s2": "LDL", "s3": "HDL",
    "s4": "Colesterol/HDL", "s5": "Triglicéridos", "s6": "Glucosa",
}

@st.cache_data
def build_models():
    X = df[FEATURES].copy()
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo Lineal
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    y_pred_linear = linear_model.predict(X_test_scaled)

    linear_metrics = {
        "R2": r2_score(y_test, y_pred_linear),
        "MAE": mean_absolute_error(y_test, y_pred_linear),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_linear)),
    }

    importances = pd.Series(np.abs(linear_model.coef_), index=FEATURES)
    importances = (importances / importances.max()).sort_values(ascending=False)

    # Modelos Polinomiales
    poly_results = {}
    for degree in [2, 3]:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)
        y_pred_poly = poly_model.predict(X_test_poly)

        poly_results[degree] = {
            "model": poly_model,
            "poly": poly,
            "metrics": {
                "R2": r2_score(y_test, y_pred_poly),
                "MAE": mean_absolute_error(y_test, y_pred_poly),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_poly)),
            },
            "y_test": y_test.values,
            "y_pred": y_pred_poly,
        }

    return {
        "linear": {
            "model": linear_model,
            "scaler": scaler,
            "metrics": linear_metrics,
            "y_test": y_test.values,
            "y_pred": y_pred_linear,
            "importances": importances,
        },
        "poly": poly_results,
        "X_train": X_train,
        "X_test": X_test,
    }

models = build_models()

# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🏥 Predicción de Progresión de Diabetes</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predicción cuantitativa de la progresión de la enfermedad un año después del inicio, basada en 10 variables basales | Regresión Lineal y Polinomial</p>', unsafe_allow_html=True)

tab_prediccion, tab_datos, tab_lineal, tab_polinomial = st.tabs([
    "🔮 Predicción", "📊 Dataset", "📈 Modelo Lineal", "🌀 Modelo Polinomial"
])

# ────────────────────────── PREDICCIÓN ──────────────────────────────────────
with tab_prediccion:
    st.subheader("Predicción de progresión de diabetes (1 año)")
    st.info("Ingrese las 10 variables basales del paciente para predecir la progresión cuantitativa de la enfermedad diabetes un año después del inicio.")

    with st.sidebar:
        st.header("🩺 Variables Basales del Paciente")

        st.markdown("**Demográficas**")
        age_real = st.slider(
            "Edad (años)",
            min_value=REAL_RANGES["age"]["min"],
            max_value=REAL_RANGES["age"]["max"],
            value=REAL_RANGES["age"]["default"],
            step=REAL_RANGES["age"]["step"],
            help=VAR_DESCRIPTIONS["age"]
        )
        sex_option = st.selectbox(
            "Sexo",
            options=["Masculino", "Femenino"],
            help=VAR_DESCRIPTIONS["sex"]
        )

        st.markdown("**Antropométricas**")
        bmi_real = st.slider(
            "IMC (kg/m²)",
            min_value=REAL_RANGES["bmi"]["min"],
            max_value=REAL_RANGES["bmi"]["max"],
            value=REAL_RANGES["bmi"]["default"],
            step=REAL_RANGES["bmi"]["step"],
            help=VAR_DESCRIPTIONS["bmi"]
        )
        bp_real = st.slider(
            "Presión Arterial (mmHg)",
            min_value=REAL_RANGES["bp"]["min"],
            max_value=REAL_RANGES["bp"]["max"],
            value=REAL_RANGES["bp"]["default"],
            step=REAL_RANGES["bp"]["step"],
            help=VAR_DESCRIPTIONS["bp"]
        )

        st.markdown("**Bioquímicas (Sangre)**")
        s1_real = st.slider(
            "Colesterol Total (mg/dL)",
            min_value=REAL_RANGES["s1"]["min"],
            max_value=REAL_RANGES["s1"]["max"],
            value=REAL_RANGES["s1"]["default"],
            step=REAL_RANGES["s1"]["step"],
            help=VAR_DESCRIPTIONS["s1"]
        )
        s2_real = st.slider(
            "LDL - Colesterol malo (mg/dL)",
            min_value=REAL_RANGES["s2"]["min"],
            max_value=REAL_RANGES["s2"]["max"],
            value=REAL_RANGES["s2"]["default"],
            step=REAL_RANGES["s2"]["step"],
            help=VAR_DESCRIPTIONS["s2"]
        )
        s3_real = st.slider(
            "HDL - Colesterol bueno (mg/dL)",
            min_value=REAL_RANGES["s3"]["min"],
            max_value=REAL_RANGES["s3"]["max"],
            value=REAL_RANGES["s3"]["default"],
            step=REAL_RANGES["s3"]["step"],
            help=VAR_DESCRIPTIONS["s3"]
        )
        s4_real = st.slider(
            "Relación Colesterol / HDL",
            min_value=REAL_RANGES["s4"]["min"],
            max_value=REAL_RANGES["s4"]["max"],
            value=REAL_RANGES["s4"]["default"],
            step=REAL_RANGES["s4"]["step"],
            help=VAR_DESCRIPTIONS["s4"]
        )
        s5_real = st.slider(
            "Triglicéridos (log mg/dL)",
            min_value=REAL_RANGES["s5"]["min"],
            max_value=REAL_RANGES["s5"]["max"],
            value=REAL_RANGES["s5"]["default"],
            step=REAL_RANGES["s5"]["step"],
            help=VAR_DESCRIPTIONS["s5"]
        )
        s6_real = st.slider(
            "Glucosa en Sangre (mg/dL)",
            min_value=REAL_RANGES["s6"]["min"],
            max_value=REAL_RANGES["s6"]["max"],
            value=REAL_RANGES["s6"]["default"],
            step=REAL_RANGES["s6"]["step"],
            help=VAR_DESCRIPTIONS["s6"]
        )

        predict_btn = st.button("🔍 Predecir", type="primary", use_container_width=True)

    if predict_btn:
        # Convertir valores reales a espacio escalado
        sex_real = 0.04 if sex_option == "Masculino" else -0.04
        input_data = pd.DataFrame([{
            "age": real_to_scaled("age", age_real),
            "sex": sex_real,
            "bmi": real_to_scaled("bmi", bmi_real),
            "bp":  real_to_scaled("bp", bp_real),
            "s1":  real_to_scaled("s1", s1_real),
            "s2":  real_to_scaled("s2", s2_real),
            "s3":  real_to_scaled("s3", s3_real),
            "s4":  real_to_scaled("s4", s4_real),
            "s5":  real_to_scaled("s5", s5_real),
            "s6":  real_to_scaled("s6", s6_real),
        }])

        input_scaled = models["linear"]["scaler"].transform(input_data)

        # Predicción Lineal
        pred_linear = models["linear"]["model"].predict(input_scaled)[0]

        # Predicciones Polinomiales
        pred_poly2 = models["poly"][2]["model"].predict(
            models["poly"][2]["poly"].transform(input_scaled)
        )[0]
        pred_poly3 = models["poly"][3]["model"].predict(
            models["poly"][3]["poly"].transform(input_scaled)
        )[0]

        st.markdown(
            f'<div class="prediction-box">'
            f'  <h2>Progresión estimada de la enfermedad (1 año)</h2>'
            f'  <p>Lineal: {pred_linear:.1f} | Polinomial (grado 2): {pred_poly2:.1f} | Polinomial (grado 3): {pred_poly3:.1f}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Comparación de modelos
        st.subheader("Comparación de Modelos")
        comp_df = pd.DataFrame({
            "Modelo": ["Regresión Lineal", "Polinomial (Grado 2)", "Polinomial (Grado 3)"],
            "Predicción": [pred_linear, pred_poly2, pred_poly3],
            "R² Score": [
                models["linear"]["metrics"]["R2"],
                models["poly"][2]["metrics"]["R2"],
                models["poly"][3]["metrics"]["R2"],
            ],
            "MAE": [
                models["linear"]["metrics"]["MAE"],
                models["poly"][2]["metrics"]["MAE"],
                models["poly"][3]["metrics"]["MAE"],
            ],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Resumen de entrada
        with st.expander("Ver resumen de las variables ingresadas", expanded=False):
            input_display = pd.DataFrame({
                "Variable": ["Edad", "Sexo", "IMC", "Presión Arterial",
                             "Colesterol Total", "LDL", "HDL",
                             "Relación Colesterol/HDL", "Triglicéridos", "Glucosa"],
                "Valor": [f"{age_real} años", sex_option, f"{bmi_real} kg/m²", f"{bp_real} mmHg",
                          f"{s1_real} mg/dL", f"{s2_real} mg/dL", f"{s3_real} mg/dL",
                          f"{s4_real}", f"{s5_real}", f"{s6_real} mg/dL"],
            })
            st.dataframe(input_display, use_container_width=True, hide_index=True)

# ────────────────────────── DATOS ────────────────────────────────────────────
with tab_datos:
    st.subheader("Diabetes Dataset (sklearn)")
    st.markdown("**Objetivo:** Predecir la progresión cuantitativa de la enfermedad diabetes un año después del inicio, basada en 10 variables basales.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df_raw):,}")
    c2.metric("Variables", len(FEATURES))
    c3.metric("Target promedio", f"{df_raw['target'].mean():.1f}")
    c4.metric("Sin nulos", f"{len(df):,}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Primeras 20 filas**")
        st.dataframe(df_raw.head(20), use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Estadísticas descriptivas**")
        st.dataframe(df_raw.describe().round(3), use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Distribuciones de Variables")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, feat in enumerate(FEATURES):
        ax = axes[i // 5, i % 5]
        ax.hist(df[feat], bins=30, edgecolor="black", alpha=0.7)
        ax.set_title(FEATURE_NAMES.get(feat, feat))
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Mapa de Correlaciones")

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    corr_df = df[FEATURES + ["target"]].copy()
    corr_df.columns = [FEATURE_NAMES.get(c, c) for c in corr_df.columns]
    corr_matrix = corr_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    plt.tight_layout()
    st.pyplot(fig_corr)

# ────────────────────────── MODELO LINEAL ────────────────────────────────────
with tab_lineal:
    st.subheader("Regresión Lineal Múltiple - Métricas")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-card"><h3>R² Score</h3><p>{models["linear"]["metrics"]["R2"]:.4f}</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><h3>MAE</h3><p>{models["linear"]["metrics"]["MAE"]:.2f}</p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><h3>RMSE</h3><p>{models["linear"]["metrics"]["RMSE"]:.2f}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([3, 2])
    with col_f1:
        st.markdown("**Importancia de variables (normalizada)**")
        imp_df = models["linear"]["importances"].reset_index()
        imp_df.columns = ["Feature", "Importance"]
        imp_df["Feature"] = imp_df["Feature"].map(FEATURE_NAMES).fillna(imp_df["Feature"])
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax_imp)
        plt.tight_layout()
        st.pyplot(fig_imp)

    with col_f2:
        st.markdown("**Coeficientes del modelo**")
        coef_df = pd.DataFrame({
            "Variable": [FEATURE_NAMES.get(f, f) for f in FEATURES],
            "Coeficiente": np.round(models["linear"]["model"].coef_, 2),
        }).sort_values("Coeficiente", key=abs, ascending=False)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        st.caption(f"**Intercepto:** {models['linear']['model'].intercept_:.2f}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.subheader("Real vs Predicho")
    fig_vs, ax_vs = plt.subplots(figsize=(10, 6))
    ax_vs.scatter(models["linear"]["y_test"], models["linear"]["y_pred"], alpha=0.6, edgecolors="k")
    min_val = min(models["linear"]["y_test"].min(), models["linear"]["y_pred"].min())
    max_val = max(models["linear"]["y_test"].max(), models["linear"]["y_pred"].max())
    ax_vs.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfecta predicción")
    ax_vs.set_xlabel("Valor Real")
    ax_vs.set_ylabel("Predicción")
    ax_vs.set_title("Regresión Lineal: Real vs Predicho")
    ax_vs.legend()
    plt.tight_layout()
    st.pyplot(fig_vs)

# ────────────────────────── MODELO POLINOMIAL ───────────────────────────────
with tab_polinomial:
    st.subheader("Regresión Polinomial - Comparación por Grado")

    st.markdown("**Métricas por Grado Polinomial**")
    poly_comp = pd.DataFrame({
        "Grado": ["Lineal (1)", "Polinomial (2)", "Polinomial (3)"],
        "R² Score": [
            models["linear"]["metrics"]["R2"],
            models["poly"][2]["metrics"]["R2"],
            models["poly"][3]["metrics"]["R2"],
        ],
        "MAE": [
            models["linear"]["metrics"]["MAE"],
            models["poly"][2]["metrics"]["MAE"],
            models["poly"][3]["metrics"]["MAE"],
        ],
        "RMSE": [
            models["linear"]["metrics"]["RMSE"],
            models["poly"][2]["metrics"]["RMSE"],
            models["poly"][3]["metrics"]["RMSE"],
        ],
    })
    st.dataframe(poly_comp, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    for degree in [2, 3]:
        st.subheader(f"Polinomial Grado {degree}")

        pm1, pm2, pm3 = st.columns(3)
        with pm1:
            st.markdown(
                f'<div class="metric-card"><h3>R² Score</h3><p>{models["poly"][degree]["metrics"]["R2"]:.4f}</p></div>',
                unsafe_allow_html=True,
            )
        with pm2:
            st.markdown(
                f'<div class="metric-card"><h3>MAE</h3><p>{models["poly"][degree]["metrics"]["MAE"]:.2f}</p></div>',
                unsafe_allow_html=True,
            )
        with pm3:
            st.markdown(
                f'<div class="metric-card"><h3>RMSE</h3><p>{models["poly"][degree]["metrics"]["RMSE"]:.2f}</p></div>',
                unsafe_allow_html=True,
            )

        fig_poly, ax_poly = plt.subplots(figsize=(10, 6))
        ax_poly.scatter(models["poly"][degree]["y_test"], models["poly"][degree]["y_pred"], alpha=0.6, edgecolors="k")
        min_val = min(models["poly"][degree]["y_test"].min(), models["poly"][degree]["y_pred"].min())
        max_val = max(models["poly"][degree]["y_test"].max(), models["poly"][degree]["y_pred"].max())
        ax_poly.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfecta predicción")
        ax_poly.set_xlabel("Valor Real")
        ax_poly.set_ylabel("Predicción")
        ax_poly.set_title(f"Polinomial Grado {degree}: Real vs Predicho")
        ax_poly.legend()
        plt.tight_layout()
        st.pyplot(fig_poly)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Resumen final
    st.subheader("📊 Resumen Comparativo")
    fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(3)
    models_names = ["Lineal", "Pol. Grado 2", "Pol. Grado 3"]
    r2_scores = [poly_comp["R² Score"].values[0], poly_comp["R² Score"].values[1], poly_comp["R² Score"].values[2]]
    bars = ax_summary.bar(x_pos, r2_scores, color=["#667eea", "#11998e", "#e74c3c"])
    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels(models_names)
    ax_summary.set_ylabel("R² Score")
    ax_summary.set_title("Comparación de R² Score entre Modelos")
    for bar, score in zip(bars, r2_scores):
        ax_summary.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{score:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    st.pyplot(fig_summary)
