# California Housing AI

## Objetivo

Predecir el precio de viviendas en California utilizando un modelo de **Regresion Lineal Multiple** entrenado con el dataset California Housing Prices de Kaggle.

## Dataset

California Housing Prices de Kaggle.
Variables predictoras: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`.
Variable objetivo: `median_house_value`.

## Que es la Regresion Lineal Multiple

Es un algoritmo de machine learning que modela la relacion entre multiples variables independientes (predictoras) y una variable dependiente (objetivo). Ajusta una ecuacion lineal: `y = b0 + b1*x1 + b2*x2 + ... + bn*xn`, donde cada coeficiente `bi` representa la contribucion de cada caracteristica al valor predicho.

## Instalacion

```bash
pip install -r requirements.txt
```

## Entrenamiento

Coloca `housing.csv` en la carpeta raiz del proyecto y ejecuta:

```bash
python train_model.py
```

Esto generara el archivo `model.pkl` con el modelo entrenado.

## Ejecucion

```bash
python app.py
```

Abre `http://localhost:5000` en tu navegador.

## Despliegue en Render

1. Sube el codigo a un repositorio Git.
2. Crea un nuevo **Web Service** en Render.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`
5. Render asigna el puerto automaticamente; la app ya esta configurada para escuchar en `PORT`.
