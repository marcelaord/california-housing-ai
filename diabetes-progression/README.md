# 🏥 Predicción de Progresión de Diabetes

Aplicación web para predecir la progresión cuantitativa de la enfermedad diabetes un año después del inicio, basada en 10 variables basales.

## Modelos Implementados

- **Regresión Lineal Múltiple**: Modelo base con las 10 features originales
- **Regresión Polinomial**: Grados 2 y 3 para capturar relaciones no lineales

## Variables de Entrada

| Variable | Descripción |
|----------|-------------|
| age | Edad del paciente |
| sex | Sexo |
| bmi | Índice de Masa Corporal |
| bp | Presión Arterial Media |
| s1 | Colesterol Total |
| s2 | LDL (lipoproteínas de baja densidad) |
| s3 | HDL (lipoproteínas de alta densidad) |
| s4 | Cociente Colesterol/HDL |
| s5 | Logaritmo de Triglicéridos |
| s6 | Nivel de Azúcar en Sangre |

## Métricas del Modelo

- **R² Score**: Coeficiente de determinación
- **MAE**: Error Absoluto Medio
- **RMSE**: Raíz del Error Cuadrático Medio

## Despliegue

La aplicación está desplegada en [Render](https://render.com/) y utiliza Streamlit como framework web.

## Ejecución Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

Utiliza el dataset de diabetes de scikit-learn, que contiene 442 registros y 10 features numéricas.

## Tecnologías

- Python 3.12
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn