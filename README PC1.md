# README PC1 - Documentación del Proyecto de Ciencia de Datos

Este documento proporciona una visión general de la metodología científica, la estructura del proyecto y la documentación de las funciones y clases utilizadas en este proyecto de ciencia de datos.

## Metodología Científica del Proyecto

El proyecto sigue una metodología de ciencia de datos estructurada, similar al marco de referencia **CRISP-DM (Cross-Industry Standard Process for Data Mining)**, que se divide en las siguientes fases:

### 1. Comprensión del Negocio (Business Understanding)
El objetivo principal del proyecto es desarrollar un modelo de pronóstico de la demanda. Específicamente, se busca predecir la cantidad de un producto específico que se venderá en fechas futuras. Esto permite optimizar el inventario, la logística y las estrategias de marketing.

### 2. Comprensión de los Datos (Data Understanding)
Los datos utilizados en este proyecto provienen de **Google Analytics**. La consulta SQL utilizada para extraer los datos se encuentra en el archivo `ga_transactions_query.sql`. Los datos contienen información detallada sobre transacciones, productos, sesiones de usuario, fuentes de tráfico y comportamiento en el sitio.

### 3. Preparación de los Datos (Data Preparation)
Esta fase es crucial para transformar los datos brutos en un formato adecuado para el modelado. El código para esta fase se encuentra en el directorio `src/preprocess`.

- **Limpieza (`cleaning.py`):** La función `clean_data` se encarga de manejar valores nulos o placeholders (como `(not set)`), convertir columnas a los tipos de datos correctos (fechas, números) e imputar valores faltantes.
- **Agregación (`aggregation.py`):** La función `aggregate_data_by_product` transforma los datos a nivel de transacción en una serie de tiempo diaria por producto (SKU). Agrega métricas como la cantidad total de productos vendidos, los ingresos totales y el número de transacciones.
- **Pipeline de Preprocesamiento (`preprocessing.py`):** El script `preprocess_for_group2` orquesta los pasos de limpieza y agregación, generando un archivo `group2_data.parquet` que sirve como entrada para la fase de modelado.

### 4. Modelado (Modeling)
El enfoque de modelado adoptado es un **modelo de dos etapas** que utiliza **XGBoost**, una potente biblioteca de gradient boosting. Este enfoque se implementa en `src/pipeline/training.py`.

1.  **Modelo de Clasificación:** Se entrena un `XGBClassifier` para predecir si un producto tendrá ventas (`cantidad > 0`) en un día determinado. Esto es útil porque muchos días pueden tener cero ventas, y tratarlo como un problema de clasificación ayuda al modelo a enfocarse.
2.  **Modelo de Regresión:** Se entrena un `XGBRegressor` para predecir la cantidad de ventas, pero **solo se entrena con los datos de los días en que hubo ventas**.

La predicción final es el producto de las dos predicciones: `predicción_cantidad = predicción_clasificador * predicción_regresor`.

La **ingeniería de características** es una parte fundamental de esta fase. Las funciones `create_features_enhanced` y `create_features_for_prediction` crean características basadas en el tiempo (día de la semana, mes, año) y características de retardo (lags) y de ventana móvil (rolling windows) para capturar tendencias y estacionalidades.

### 5. Evaluación (Evaluation)
Aunque el código de evaluación no está completamente implementado en los scripts de `src`, los notebooks (`04_model_comparison.ipynb` y `05_modeling_final.ipynb`) se realizaron la fase de evaluación y comparación de modelos. El archivo `src/utils/metrics.py` está destinado a contener funciones para calcular métricas de evaluación como RMSE, MAE o métricas de clasificación.

### 6. Despliegue (Deployment)
El proyecto está estructurado para ser desplegable. Los scripts en `src/pipeline` permiten la operacionalización del modelo:
- `training.py`: Permite re-entrenar los modelos para un SKU de producto específico y guardar los artefactos del modelo en el directorio `models`.
- `prediction.py`: Carga un modelo entrenado y genera predicciones para un número determinado de días en el futuro.

## Estructura del Proyecto

```
.
├── data/                # Almacena los datos brutos, intermedios y procesados.
│   ├── raw/
│   └── processed/
├── models/              # Almacena los modelos entrenados.
├── notebooks/           # Jupyter notebooks para exploración y experimentación.
├── src/                 # Código fuente del proyecto.
│   ├── models/          # Scripts relacionados con diferentes tipos de modelos.
│   ├── pipeline/        # Pipelines de entrenamiento y predicción.
│   ├── preprocess/      # Scripts para la preparación de datos.
│   └── utils/           # Funciones de utilidad (métricas, visualización).
├── tests/               # Pruebas unitarias para el código fuente.
├── ga_transactions_query.sql # Query para extraer datos de Google Analytics.
└── README.md            # Documentación del proyecto.
```

## Documentación de Clases y Funciones

A continuación se documentan las principales funciones del directorio `src`.

### `src/preprocess`

- **`clean_data(df)`** en `cleaning.py`:
    - **Propósito:** Limpiar el DataFrame de transacciones crudas.
    - **Acciones:** Reemplaza placeholders, convierte la columna de fecha, convierte columnas numéricas y rellena valores nulos.
    - **Retorna:** Un DataFrame de pandas limpio.

- **`aggregate_data_by_product(df)`** en `aggregation.py`:
    - **Propósito:** Agregar los datos limpios por fecha y SKU de producto.
    - **Acciones:** Agrupa los datos y calcula la suma de `product_quantity` y `product_revenue_usd`, la media de `product_price_usd` y el número de transacciones únicas.
    - **Retorna:** Un DataFrame agregado.

- **`preprocess_for_group2(input_path, output_path)`** en `preprocessing.py`:
    - **Propósito:** Orquestar el pipeline de preprocesamiento.
    - **Acciones:** Lee los datos crudos, llama a `clean_data` y `aggregate_data_by_product`, y guarda el resultado en un archivo parquet.
    - **Retorna:** El DataFrame preprocesado y agregado.

### `src/pipeline`

- **`run_training(product_sku)`** en `training.py`:
    - **Propósito:** Ejecutar el pipeline completo de entrenamiento para un producto.
    - **Acciones:** Carga los datos procesados, crea una serie de tiempo para el SKU, realiza la ingeniería de características, y entrena y guarda los modelos de clasificación y regresión de XGBoost.

- **`run_prediction(product_sku, prediction_days)`** en `prediction.py`:
    - **Propósito:** Generar predicciones de demanda para un producto.
    - **Acciones:** Carga los modelos entrenados y los datos históricos, crea características para las fechas futuras, y utiliza el modelo de dos etapas para generar las predicciones.
    - **Retorna:** Un DataFrame con las predicciones de cantidad para cada fecha futura.

### `src/models`

- **`create_features(df, target_col)`** en `ml.py`:
    - **Propósito:** Crear características de series de tiempo a partir de un DataFrame.
    - **Acciones:** Añade características como día de la semana, mes, año, y valores de lag y medias móviles.

- **`create_sequences(dataset, look_back)`** en `deep_learning.py`:
    - **Propósito:** Preparar secuencias de datos para modelos de deep learning como LSTMs.
    - **Acciones:** Convierte un array de valores en secuencias de entrada (X) y valores de salida (Y).

