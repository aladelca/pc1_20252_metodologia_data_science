# Práctica Calificada 1 - Metodología para Data Science

## Descripción General

Este repositorio forma parte de la **Práctica Calificada 1** del curso de **Metodología para Data Science** . El objetivo principal es desarrollar un pipeline completo de machine learning para entrenamiento y predicción de series de tiempo, aplicando las mejores prácticas de desarrollo colaborativo y metodologías de ciencia de datos.

## Estructura del Proyecto

Los estudiantes trabajarán en **4 grupos**, cada uno enfocado en un nivel diferente de agregación de datos para desarrollar modelos de predicción de series de tiempo.

### Dataset Inicial

Todos los grupos trabajarán con el mismo dataset base:
- **Archivo**: `data/raw/data_sample.parquet`
- **Descripción**: Datos de transacciones de Google Analytics con información de productos, sesiones, dispositivos, geografía y comportamiento de usuarios

### Niveles de Agregación por Grupo

Cada grupo trabajará con un nivel de agregación específico:

1. **Grupo 1 - Día/Producto/Transacción**:
   - Granularidad más detallada
   - Predicción a nivel de transacción individual por día y producto

2. **Grupo 2 - Día/Producto**:
   - Agregación intermedia
   - Predicción de métricas diarias por producto

3. **Grupo 3 - Día/Categoría**:
   - Agregación por categoría de producto
   - Predicción de tendencias diarias por categoría

4. **Grupo 4 - Día/Marca**:
   - Agregación por marca de producto
   - Predicción de performance diaria por marca

## Objetivos y Entregables

### 1. Procesamiento de Datos
- Crear el pipeline de agregación correspondiente a su grupo. Revisar la carpeta preprocess y el archivo `preprocessing.py`
- Implementar limpieza y transformación de datos. Revisar la carpeta preprocess y el archivo `cleaning.py`
- Implementar funciones asociadas a la agregación para cada grupo. Revisar la carpeta preprocess y el archivo `aggregation.py`
- Validar la calidad de los datos procesados y de las funciones implementadas

### 2. Modelado de Series de Tiempo
- **Experimentar** con al menos 3 modelos diferentes de series de tiempo:
  - Modelos estadísticos (ARIMA, SARIMA, Exponential Smoothing)
  - Modelos de machine learning (XGBoost, Random Forest, LightGBM)
  - Modelos de deep learning (LSTM, GRU, Transformer)
- Evaluar y comparar el rendimiento de los modelos
- Seleccionar el mejor modelo basado en métricas de evaluación
- En muchos casos, debido a la agregación, deberán entrenar varios modelos
- Los experimentos deberá quedar registrados en la carpeta `notebooks`
### 3. Pipeline de Entrenamiento y Predicción
- Implementar pipeline automatizado de entrenamiento
- Desarrollar sistema de predicción para nuevos datos

### 4. Calidad de Código y Testing
- **Tests con pytest**: Implementar tests unitarios para cada función desarrollada. Los tests deben estar en la carpeta `test`
- **Estándares de código**: Cumplir con los estándares de:
  - `ruff` (linting y formatting)
  - `flake8` (style guide enforcement)
  - `mypy` (type checking)
- **Documentación**: Documentar apropiadamente todas las funciones y clases, además de la metodología científica.

## Estructura de Carpetas Recomendada

```
├── data/
│   └── raw/
│       └── data_sample.parquet          # Dataset inicial (NO MODIFICAR)
├── src/
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── preprocessing.py             # Funciones de procesamiento
│   │   ├── cleaning.py                  # Funciones de limpieza
│   │   └── aggregation.py               # Funciones de agregación
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                      # Clase base para modelos
│   │   ├── statistical.py              # Modelos estadísticos
│   │   ├── ml.py                        # Modelos de ML
│   │   └── deep_learning.py             # Modelos de DL
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training.py                  # Pipeline de entrenamiento
│   │   └── prediction.py               # Pipeline de predicción
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                   # Métricas de evaluación
│       └── visualization.py            # Funciones de visualización
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_pipeline.py
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_experimentation.ipynb
│   └── 03_results_analysis.ipynb
├── configs/
│   ├── model_config.yaml
│   └── pipeline_config.yaml
├── requirements.txt
├── pyproject.toml                       # Configuración de ruff, mypy
├── .pre-commit-hooks.yaml              # Hooks de pre-commit
└── README.md
```

## Criterios de Evaluación

### 1. Contribución Individual (30%)
- **Commits**: Frecuencia y calidad de commits
- **Pull Requests**: Revisiones de código y colaboración
- **Participación**: Contribución activa en el desarrollo del proyecto

### 2. Calidad Técnica (40%)
- **Completitud de tests**: Cobertura de tests unitarios con pytest
- **Estándares de código**: Cumplimiento con ruff, flake8 y mypy
- **Arquitectura**: Diseño modular y mantenible del código
- **Rendimiento del modelo**: Calidad de las predicciones

### 3. Documentación y Organización (20%)
- **Documentación de código**: Docstrings y comentarios apropiados
- **README y documentación**: Claridad en la documentación del proyecto
- **Organización del repositorio**: Estructura clara y mantenible

### 4. Metodología de Trabajo (10%)
- **Workflow colaborativo**: Uso apropiado de Git y GitHub
- **Gestión de proyecto**: Organización y planificación del trabajo
- **Buenas prácticas**: Seguimiento de metodologías de data science

## Configuración del Entorno

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### Configuración de Herramientas de Calidad
```bash
# Instalar pre-commit hooks
pre-commit install

# Ejecutar linting
ruff check src/ tests/
ruff format src/ tests/

# Ejecutar type checking
mypy src/

# Ejecutar tests
pytest tests/ -v --cov=src
```

## Reglas de Trabajo Colaborativo

1. **Branching Strategy**:
   - Usar `main` como rama principal
   - Crear feature branches para nuevas funcionalidades
   - Realizar merge solo mediante Pull Requests

2. **Code Review**:
   - Todos los PRs deben ser revisados por al menos un compañero
   - Resolver todos los comentarios antes del merge

3. **Commits**:
   - Usar mensajes de commit descriptivos
   - Hacer commits atómicos y frecuentes

4. **Testing**:
   - Escribir tests antes o junto con el código
   - Mantener cobertura de tests > 80%


## Recursos Adicionales

- [Documentación de pytest](https://docs.pytest.org/)
- [Guía de ruff](https://docs.astral.sh/ruff/)
- [Documentación de mypy](https://mypy.readthedocs.io/)
- [Mejores prácticas de Git](https://www.atlassian.com/git/tutorials/comparing-workflows)

## Contacto

Para consultas sobre la práctica calificada, contactar al equipo docente del curso.

---

**Nota Importante**: El dataset `data/raw/data_sample.parquet` es el único archivo de datos proporcionado. Los grupos deben generar sus propios datasets agregados a partir de este archivo base.