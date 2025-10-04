
# ANALISIS DE DATOS POR SERIE DE TIEMPO


 Objetivo del Proyecto
El propósito principal de este notebook es realizar la Predicción Diaria por Marca  de una métrica denominada events. El proyecto se centra en comparar la eficacia de tres familias de modelos de pronóstico de series de tiempo en un contexto de predicción univariada.
## 1.	Configuración inicial e importación de datos
   python

* IMPORTS & CONFIG

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 140)

* Ruta del archivo
PARQUET_PATH = Path('../data/raw/data_daily_group04.parquet')

print(' Archivo esperado:', PARQUET_PATH)
* Carga y exploración inicial de datos
python
* Leer Parquet
   data = pd.read_parquet('../data/raw/data_daily_group04.parquet')

*  Mostrar resultados
   display(data.head())  # primeros 5 registros
   print(f'Filas: {data.shape[0]}, Columnas: {data.shape[1]}')

### Análisis de los datos mostrados:

   El dataset contiene 74,457 filas y 77 columnas con información de transacciones de Google Analytics, incluyendo:
•	Datos de transacciones: fecha, ID, ingresos, impuestos, envío
•	Información de productos: SKU, nombre, categoría, marca, precio
•	Datos de sesión: ID de visitante, ID de sesión, tiempo en el sitio
•	Información de dispositivo: navegador, SO, resolución de pantalla
•	Datos geográficos: continente, país, región, ciudad
•	Información de marketing: fuente de tráfico, campaña, palabras clave

### Problemas identificados:

•	Valores faltantes: Muchas columnas tienen valores NaN o None
•	Columnas vacías: is_impression, is_click, promo_id, etc., están completamente vacías
•	Datos demo: Algunos campos contienen "not available in demo dataset"
•	Procesos realizados:
   o	Limpieza de datos: Eliminar columnas completamente vacías
   o	Análisis de valores faltantes: Decidir estrategia para manejar NaNs
   o	Conversión de tipos de datos: Fechas y columnas categóricas
   o	Análisis exploratorio: Distribuciones, correlaciones, outliers

## 2. Metodología y Modelos Comparados

El notebook establece una comparativa de rendimiento entre tres enfoques principales para el pronóstico:
1.	Serie de Tiempo Clásica: Modelo SARIMA (Seasonal AutoRegressive Integrated Moving Average).
2.	Machine Learning: Modelo XGBoost (Extreme Gradient Boosting), utilizando ingeniería de características (lags, medias móviles y variables de calendario) para transformar la serie de tiempo en un problema de regresión supervisada.
3.	Deep Learning: Redes LSTM (Long Short-Term Memory) con una estructura de ventanas deslizantes (lookback), comunes en el modelado de secuencias.
6.	Instalación de librerias

Para ejecutar el código, se requiere la instalación de las siguientes bibliotecas (normalmente ejecutadas con pip si se corre en un entorno como Colab o un entorno local):

•	pandas (específicamente la versión 2.2.2 mencionada en el código), numpy, matplotlib, scikit-learn
•	statsmodels (para SARIMA)
•	xgboost (para XGBoost)
•	tensorflow (para LSTM)
•	pmdarima (sugerido en el setup para auto-ARIMA, aunque no se usa en los imports visibles)

## 3 .Estructura del Código (Pipeline)

El flujo del notebook sigue una estructura estándar para proyectos de Data Science en series de tiempo:
Sección	Descripción	Funciones Clave / Componentes

1) Setup  
2) Imports	
   Configuración inicial del entorno e importación de todas las bibliotecas necesarias (incluyendo utilidades de sklearn.metrics, XGBRegressor, SARIMAX, LSTM, Dense).	
   warnings.filterwarnings('ignore'), plt.rcParams['figure.figsize']
3) Utilidades	Definición de las funciones de soporte para la carga de datos y la ingeniería de características.	load_single_brand_series, make_features
4) Cargar datos	Lectura de los datos desde un archivo Parquet y transformación en una serie de tiempo univariada diaria para una marca específica.
load_single_brand_series
5) Modelado (SARIMA / XGBoost / LSTM)	Implementación de la lógica de entrenamiento, validación y pronóstico para cada uno de los tres modelos.
SARIMAX.fit(), XGBRegressor.fit(), MinMaxScaler, Sequential(LSTM, Dense).fit()
6) Evaluación y Pronóstico	Cálculo de métricas de error y generación del pronóstico de 30 días con los modelos entrenados (solo se muestra la gráfica de XGBoost en el snippet).	Cálculo de MAE, RMSE, MAPE
7) Resumen del Pipeline	Confirmación del guardado de artefactos.	
Guardado de métricas y predicciones en ./notebooks/, y el modelo XGBoost en ./src/models/.



###	Componentes Clave (Funciones de Utilidad)
load_single_brand_series
Esta función es crucial para la preparación inicial de los datos:
•	Entrada: Ruta del archivo Parquet (parquet_path), nombres de columnas (fecha, marca, métrica) y valor de la marca a filtrar (brand_value, por defecto '(not set)'), y frecuencia de remuestreo (freq, por defecto 'D' - diario).

•	Proceso:

1.	Carga el DataFrame desde el archivo Parquet.
2.	Asegura el parseo de la columna de fecha (transaction_date), manejando diferentes formatos.
3.	Filtra la serie por el valor de marca especificado (product_brand).
4.	Convierte la métrica objetivo (events) a tipo numérico.
5.	Remuestrea los datos a la frecuencia diaria, suma los eventos por día y rellena los días sin datos con cero (0).
•	Salida: Un objeto pd.Series con la serie de tiempo diaria de eventos para la marca elegida.
make_features (Ingeniería de Características para ML)
Esta función crea las variables predictoras (explicativas) para el modelo XGBoost:
•	Entrada: La serie de tiempo univariada (series).
•	Características Creadas:
o	Lags: Valores históricos de la métrica (target) desplazados para diferentes períodos (por defecto: 1, 2, 3, 5, 7, 14, 21, 28 días).
o	Medias Móviles (MA): Promedios de la métrica en ventanas pasadas (por defecto: 7, 14, 28 días).
o	Características de Calendario: Día de la semana (dow), día del mes (dom), semana (week), mes (month), año (year).
o	Indicador de Fin de Semana: Variable binaria (is_weekend) para días 5 (Sábado) y 6 (Domingo).
•	Salida: Un objeto pd.DataFrame con la columna target (el valor a predecir) y todas las variables rezagadas y de calendario.

##	Evaluación
Aunque la función evaluate está comentada y probablemente implementada directamente en el notebook, el enfoque de evaluación se basa en métricas de error comunes para regresión y pronóstico de series de tiempo:
•	MAE (Mean Absolute Error)
•	RMSE (Root Mean Squared Error)
•	MAPE% (Mean Absolute Percentage Error)
El código guarda las métricas y las predicciones en archivos para un análisis posterior.

## 3	Calidad de Código y Testing

Para garantizar la robustez, mantenibilidad y cumplimiento de buenas prácticas, se exige que el código cumpla con los siguientes estándares de calidad y estrategia de testing:

### 1. Tests Unitarios con pytest
•	Objetivo: Implementar pruebas unitarias para cada función clave del pipeline (utilidades y lógica de modelado) utilizando el framework pytest.

•	Ubicación: Todos los archivos de prueba deben residir en la carpeta test/ o tests/ del proyecto.

•	Requerimientos: Se deben testear al menos las utilidades de transformación de datos, como load_single_brand_series y make_features, utilizando datos simulados (mocks) para verificar:

   o	La correcta estructura de las series de tiempo y DataFrames.
   o	La generación correcta de lags y medias móviles.
   o	La gestión adecuada de valores nulos (NaN) o ausentes (cero).
   •	Comando de Ejecución: pytest

### 2. Estándares de Código y Type Checking
El proyecto debe mantener un código limpio, legible y tipado consistentemente, utilizando las siguientes herramientas en el flujo de integración continua (CI) o antes de la subida a control de versiones:

Herramienta	Función	Detalle de Aplicación
### ruff
	Linting y Formatting	Se recomienda como la herramienta principal para formato automático y detección de errores de estilo (reemplazando a isort y cubriendo la mayoría de las reglas de flake8) para asegurar el cumplimiento del estándar PEP 8.
   `Instalacion`: pip install ruff
   `Prueba`: 
   ruff check .
   ruff format .
   ruff format --check . --fix
   

### flake8	
   Style Guide Enforcement	Asegurar la adhesión estricta a las guías de estilo, incluyendo la detección de complejidad excesiva (cyclomatic complexity) y la verificación de líneas largas o importaciones incorrectas que ruff pudiera omitir en una configuración básica.

   `Instalacion`: pip install flake8
   `Prueba`: 
   flake8 .
   Flake8 src/ tests/ 
   
# mypy
	Type Checking	Ejecutar una comprobación estática de tipos sobre todos los archivos .py de las utilidades. Esto requiere que todas las funciones clave, como load_single_brand_series y make_features, estén debidamente anotadas con tipos (type hints) en sus firmas.

   `Instalacion`: pip install mypy
   `Prueba`: 
   mypy 
   mypy src/ tests/
   

