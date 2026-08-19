# 🍷 Simulador de Calidad de Vino (Wine Quality)

Simulador interactivo para predecir la **calidad del vino (escala 0–10)** a partir de sus
propiedades físico-químicas, usando **dos enfoques de regresión comparables**:

1. **Regresión Lineal Múltiple** (grado 1)
2. **Regresión Polinomial** (grados 2, 3 y 4)

Desarrollado para el **Taller 1.1 · Caso 2 (Wine Quality)** de la asignatura
*Inteligencia Artificial / Aprendizaje Automático*.

---

## 📊 Dataset

`data/WineQT.csv` — vinos tintos con **11 features** físico-químicas:

| Feature | Descripción |
|---|---|
| fixed acidity | Acidez fija (g/dm³) |
| volatile acidity | Acidez volátil (g/dm³) |
| citric acid | Ácido cítrico (g/dm³) |
| residual sugar | Azúcar residual (g/dm³) |
| chlorides | Cloruros (g/dm³) |
| free sulfur dioxide | SO₂ libre (mg/dm³) |
| total sulfur dioxide | SO₂ total (mg/dm³) |
| density | Densidad (g/cm³) |
| pH | pH |
| sulphates | Sulfatos (g/dm³) |
| alcohol | Alcohol (% vol.) |

**Target:** `quality` (3–8 en el dataset). No se genera **ningún dato sintético**: los modelos
se entrenan y evalúan **solo** con los datos reales del CSV (split 80/20).

---

## 🚀 Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Despliegue en la nube

- **Streamlit Community Cloud:** conectar el repositorio de GitHub → *New app* → archivo `app.py`
- **Hugging Face Spaces:** crear Space con SDK `streamlit` y subir estos archivos
- **Render:** Web Service → comando `streamlit run app.py`

> El tema visual (paleta de vino "bodega clara") está definido en `.streamlit/config.toml`
> y los estilos personalizados en `assets/style.css`.

---

## 🧭 Uso del simulador

1. En la barra lateral elige el **tipo de regresión** (Lineal Múltiple o Polinomial con su grado).
2. Ajusta los **valores físico-químicos** de tu vino con los deslizadores agrupados por tema
   (🍇 acidez · 🍬 azúcar y cuerpo · 🧂 minerales · 🌬️ frescura · 🍷 alcohol), con rangos reales del dataset.
3. Obtén la **nota de calidad** en vivo: número /10 con etiqueta, estrellas y la **mejor opción**
   entre ambas regresiones.
4. 🎢 **¿Cómo cambia la nota según una variable?** esconde la clave de la diferencia: la línea
   recta de la regresión lineal vs. la curva de la polinomial (gráfico interactivo Plotly).
5. 🔬 En *Así se entrena y se mide el modelo* quedan la ecuación, coeficientes, real vs. predicho
   y la comparativa de los 4 modelos (métricas R²/RMSE/MAE).

## 📁 Estructura

```
wine-quality/
├─ app.py                # Simulador Streamlit
├─ model.py              # Entrenamiento (MLR y Polinomial), métricas y predicción
├─ data/WineQT.csv       # Dataset (fila "Id" excluida por ser solo identificador)
├─ assets/style.css      # Estilo personalizado (fuentes, tarjetas, sliders)
├─ .streamlit/config.toml# Tema global (paleta vino "bodega clara")
├─ requirements.txt
└─ README.md
```

## 🧪 Pipeline (Fase A y B de la guía)

- Preprocesamiento: datos limpios sin nulos ni duplicados, features estandarizadas con `StandardScaler`.
- Modelado: `LinearRegression` (múltiple, sin regularizar) y `PolynomialFeatures + RidgeCV` (polinomial con regularización Ridge, λ elegida por cross-validation).
- Métricas: R², RMSE y MAE calculadas sobre los conjuntos de train y test.