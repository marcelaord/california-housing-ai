# 📋 Documento de Proyecto: Predicción de Progresión de Diabetes

## 🎯 Objetivo
Predecir la progresión cuantitativa de la enfermedad diabetes un año después del inicio, basada en 10 variables basales, utilizando Regresión Lineal Múltiple y Regresión Polinomial.

---

## 📊 Datasets Utilizados

### Dataset 1: Diabetes (sklearn)
- **Fuente:** `sklearn.datasets.load_diabetes()`
- **Registros:** 442
- **Variables:** 10 features + 1 target
- **Descripción:** Datos médicos de pacientes con diabetes
- **Variables:**
  - `age` - Edad del paciente
  - `sex` - Sexo del paciente
  - `bmi` - Índice de Masa Corporal
  - `bp` - Presión arterial promedio
  - `s1` - Colesterol total
  - `s2` - LDL (lipoproteínas de baja densidad)
  - `s3` - HDL (lipoproteínas de alta densidad)
  - `s4` - Relación colesterol/HDL
  - `s5` - Logaritmo de triglicéridos
  - `s6` - Glucosa en sangre
- **Target:** Progresión de la enfermedad (25-346)

### Dataset 2: California Housing (Kaggle)
- **Fuente:** Kaggle (manosarhan/diabetes-dataset-from-sklearn-datasets)
- **Registros:** 20,640
- **Variables:** 8 features + 1 target
- **Descripción:** Valores de viviendas en California
- **Variables:**
  - `longitude` - Longitud geográfica
  - `latitude` - Latitud geográfica
  - `housing_median_age` - Edad media de la vivienda
  - `total_rooms` - Total de habitaciones
  - `total_bedrooms` - Total de dormitorios
  - `population` - Población del bloque
  - `households` - Número de hogares
  - `median_income` - Ingreso medio del hogar
  - `ocean_proximity` - Proximidad al océano
- **Target:** Valor mediano de la vivienda

### Dataset 3: (Por definir)
- **Fuente:** Pendiente
- **Descripción:** Pendiente

---

## 🔬 Fase A: Análisis Exploratorio y Preprocesamiento

### Dataset 1: Diabetes

#### 1. Limpieza de Datos
- [x] **Valores nulos:** Verificado - No hay valores nulos en el dataset
- [x] **Valores faltantes:** Completos (442/442 registros)
- [x] **Tipos de datos:** Numéricos continuos (excepto sex que es categórico codificado)
- [x] **Duplicados:** No hay registros duplicados

#### 2. Tratamiento de Outliers
- [ ] **Método a utilizar:** IQR (Interquartile Range) o Z-score
- [ ] **Variables a analizar:** Todas las numéricas
- [ ] **Acción:** Identificar y decidir si eliminar o imputar

#### 3. Análisis de Multicolinealidad
- [ ] **Matriz de correlación:** Pendiente de implementar
- [ ] **VIF (Variance Inflation Factor):** 
  - VIF > 5: Multicolinealidad moderada
  - VIF > 10: Multicolinealidad severa
- [ ] **Variables sospechosas:** s1-s6 pueden tener correlación (mediciones de sangre)

#### 4. Normalización/Estandarización
- [x] **Método aplicado:** StandardScaler (media=0, std=1)
- [x] **Justificación:** Las variables tienen diferentes escalas médicas
- [x] **Implementado en:** `app.py` línea 58-60

#### 5. Feature Engineering
- [x] **Variables creadas:** Ninguna adicional
- [x] **Selección de features:** 10 variables originales
- [x] **Justificación:** Dataset ya viene preprocesado por sklearn

---

### Dataset 2: California Housing

#### 1. Limpieza de Datos
- [ ] **Valores nulos:** Verificar `df.isnull().sum()`
- [ ] **Valores faltantes:** Tratar `total_bedrooms` (puede tener nulos)
- [ ] **Tipos de datos:** Numéricos + categórico (ocean_proximity)
- [ ] **Duplicados:** Verificar

#### 2. Tratamiento de Outliers
- [ ] **Método a utilizar:** IQR o percentiles
- [ ] **Variables a analizar:** `median_house_value`, `median_income`, `population`
- [ ] **Acción:** Decidir estrategia

#### 3. Análisis de Multicolinealidad
- [ ] **Matriz de correlación:** Implementar
- [ ] **VIF:** Calcular para todas las variables numéricas
- [ ] **Variables sospechosas:**
  - `total_rooms` y `total_bedrooms` (alta correlación esperada)
  - `population` y `households` (correlación esperada)
  - `total_rooms` y `households` (correlación esperada)

#### 4. Normalización/Estandarización
- [ ] **Método a utilizar:** StandardScaler o MinMaxScaler
- [ ] **Variables a escalar:** Todas las numéricas
- [ ] **Justificación:** Variables en diferentes unidades (USD, años, personas)

#### 5. Feature Engineering
- [x] **Variables creadas:**
  - `rooms_per_household` = total_rooms / households
  - `population_per_household` = population / households
  - `bedrooms_per_room` = total_bedrooms / total_rooms
- [x] **Selección de features:** 12 variables (10 originales + 3 nuevas)
- [x] **Implementado en:** `app.py` (California Housing) líneas 44-53

---

## 📈 Fase B: Modelamiento Estadístico y Comparativa

### Dataset 1: Diabetes

#### 1. Regresión Lineal Múltiple
- [x] **Implementación:** `sklearn.linear_model.LinearRegression`
- [x] **División de datos:** 80% train, 20% test (`random_state=42`)
- [x] **Métricas calculadas:**
  - R² Score
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
- [x] **Análisis de coeficientes:** Implementado
- [x] **Importancia de variables:** Implementado (valores absolutos normalizados)
- [x] **Gráfico Real vs Predicho:** Implementado

**Resultados Esperados:**
- R² Score: ~0.45-0.55 (dataset difícil)
- MAE: ~40-50
- RMSE: ~50-60

#### 2. Regresión Polinomial
- [x] **Grado 2:** Implementado
- [x] **Grado 3:** Implementado
- [x] **Métricas por grado:** Implementado
- [x] **Comparación de modelos:** Implementado
- [x] **Gráficos Real vs Predicho:** Implementado

**Análisis de Overfitting:**
- Grado 1 (Lineal): Bias alto, varianza baja
- Grado 2: Balance ideal
- Grado 3: Posible overfitting

#### 3. Comparativa de Modelos
- [x] **Tabla comparativa:** Implementado
- [x] **Gráfico de barras R²:** Implementado
- [x] **Conclusión:** Pendiente de análisis

---

### Dataset 2: California Housing

#### 1. Regresión Lineal Múltiple
- [ ] **Implementación:** Pendiente
- [ ] **División de datos:** 80/20
- [ ] **Métricas:** R², MAE, RMSE
- [ ] **Análisis de coeficientes:** Pendiente
- [ ] **Importancia de variables:** Pendiente
- [ ] **Gráfico Real vs Predicho:** Pendiente

**Resultados Esperados:**
- R² Score: ~0.60-0.70
- MAE: ~30,000-50,000
- RMSE: ~40,000-60,000

#### 2. Regresión Polinomial
- [ ] **Grado 2:** Pendiente
- [ ] **Grado 3:** Pendiente (cuidado con 12 features)
- [ ] **Métricas por grado:** Pendiente
- [ ] **Comparación:** Pendiente

**Nota:** Con 12 features, el grado 2 genera ~90 features y grado 3 ~360. Riesgo de overfitting.

#### 3. Comparativa de Modelos
- [ ] **Tabla comparativa:** Pendiente
- [ ] **Gráfico de barras R²:** Pendiente
- [ ] **Conclusión:** Pendiente

---

## 🛠️ Herramientas Utilizadas

### Librerías Python
```python
# Análisis de datos
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Datasets
from sklearn.datasets import load_diabetes

# Web App
import streamlit as st
```

### Métricas de Evaluación
| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **R² Score** | 1 - (SS_res / SS_tot) | 1.0 = perfecto, 0.0 = media |
| **MAE** | mean(\|y - ŷ\|) | Menor = mejor |
| **RMSE** | √(mean((y - ŷ)²)) | Penaliza errores grandes |

---

## 📁 Estructura del Proyecto

```
diabetes-progression/
├── app.py                 # Aplicación Streamlit principal
├── requirements.txt       # Dependencias Python
├── setup.sh              # Configuración para Render
├── Procfile              # Comando de inicio
├── .gitignore           # Archivos ignorados
├── README.md            # Documentación
└── PROYECTO.md          # Este documento
```

---

## 🚀 Despliegue

### GitHub
- **Repositorio:** https://github.com/DJosh-gif/diabetes-progression
- **Último commit:** Sliders con valores médicos reales

### Render
- **URL:** https://diabetes-progression.onrender.com (pendiente)
- **Runtime:** Python 3.12
- **Build:** `pip install -r requirements.txt`
- **Start:** `sh setup.sh && streamlit run app.py --server.port $PORT`

---

## 📝 Notas Importantes

### Sobre el Dataset de Diabetes
1. Los valores están **pre-escalados** (mean-centered y std-scaled)
2. Rango típico: -0.2 a 0.2
3. La app convierte automáticamente valores reales al espacio escalado
4. El target (progresión) rango: 25-346

### Sobre la Conversión de Valores
```python
# Fórmula de conversión
proporcion = (valor_real - real_min) / (real_max - real_min)
valor_escalado = scaled_min + proporcion * (scaled_max - scaled_min)
```

### Sobre Multicolinealidad
- **VIF = 1:** No hay correlación
- **VIF 1-5:** Correlación moderada (aceptable)
- **VIF 5-10:** Correlación alta (problemas)
- **VIF > 10:** Correlación severa (eliminar variable)

---

## ✅ Checklist de Entrega

### Fase A
- [ ] Limpieza de datos completada
- [ ] Outliers tratados
- [ ] Matriz de correlación generada
- [ ] VIF calculado
- [ ] Variables normalizadas/estandarizadas

### Fase B
- [ ] Regresión Lineal Múltiple entrenada
- [ ] Regresión Polinomial implementada
- [ ] Métricas calculadas
- [ ] Comparativa realizada
- [ ] Conclusiones documentadas

### Entrega Final
- [ ] Código funcionando localmente
- [ ] Código desplegado en Render
- [ ] Documentación completa
- [ ] Presentación preparada

---

## 📅 Cronograma

| Fase | Tarea | Estado | Fecha |
|------|-------|--------|-------|
| A | Limpieza de datos | ✅ Completado | - |
| A | Análisis exploratorio | ✅ Completado | - |
| A | Multicolinealidad | ⏳ Pendiente | - |
| A | Normalización | ✅ Completado | - |
| B | Regresión Lineal | ✅ Completado | - |
| B | Regresión Polinomial | ✅ Completado | - |
| B | Comparativa | ✅ Completado | - |
| - | Despliegue | ✅ Completado | - |

---

**Documento generado automáticamente - Última actualización: 2026-08-19**
