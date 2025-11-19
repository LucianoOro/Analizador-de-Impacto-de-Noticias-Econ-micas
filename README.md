# 📊 Macro Impact Analyzer: Predicción de Mercados Financieros ante Shocks Económicos

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Modeling-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-ETL-017CEE?logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)

> **Proyecto Final de Ciencia de Datos** - UTN FRM
> **Equipo:** Emanuel Gomez, Luciano Oro, Julieta Vicente.

## 🔗 [Ver Aplicación en Vivo (Demo)](PEGAR_AQUI_TU_LINK_DE_STREAMLIT)

---

## 📝 Descripción del Proyecto

**Macro Impact Analyzer** es un sistema de inteligencia artificial diseñado para cuantificar y predecir la reacción inmediata de activos financieros (Índices, Criptomonedas, Forex y Commodities) ante la publicación de indicadores macroeconómicos de EE.UU. (CPI, NFP, FOMC, etc.).

A diferencia de los modelos tradicionales de series temporales, este proyecto no utiliza precios pasados para predecir el futuro. En su lugar, modela la **psicología del mercado** basándose exclusivamente en la **"Sorpresa Matemática"** (Z-Score) de la noticia en el instante de su publicación.

### 🎯 Objetivo Principal
Resolver el problema de la incertidumbre durante eventos de alta volatilidad, proporcionando una clasificación binaria (**Sube/Baja**) de la vela de 1 hora (H1) y estimando la probabilidad de éxito.

---

## ⚙️ Arquitectura y Tecnologías

El proyecto sigue un ciclo de vida completo de Ciencia de Datos (End-to-End):

| Etapa | Tecnología | Descripción |
| :--- | :--- | :--- |
| **1. Ingeniería de Datos** | `Apache Airflow`, `Pandas` | Pipeline ETL automatizado para ingesta, limpieza y alineación temporal de fuentes heterogéneas. |
| **2. EDA & Insights** | `Matplotlib`, `Seaborn` | Análisis estadístico, detección de no-linealidad y correlaciones inter-mercado. |
| **3. Machine Learning** | `Scikit-learn` | Modelado predictivo, validación cruzada, detección de *Data Leakage* y ajuste de hiperparámetros. |
| **4. Despliegue** | `Streamlit`, `Altair` | Aplicación web interactiva con simulador en tiempo real y backtesting. |

---

## 🚀 Metodología y Hallazgos Clave

### 1. Ingeniería de Características (Feature Engineering)
El mayor desafío fue la **heterogeneidad de datos** (comparar Inflación en `%` vs Empleo en `miles`).
* **Solución:** Estandarización universal mediante **Z-Score**.
* $$Sorpresa = \frac{(Valor Real - Pronóstico)}{Desviación Estándar Histórica}$$

### 2. Análisis Exploratorio (EDA) - El "Cuarteto de Anscombe"
Descubrimos que la relación entre la noticia y el precio **no es lineal**.
* **Hallazgo:** El mercado ignora las sorpresas pequeñas (ruido). La señal direccional solo emerge en los **eventos extremos** (> 2 Desvíos Estándar).
* **Acción:** Filtramos el dataset para entrenar los modelos únicamente con los eventos de "Cola" (Extremos), eliminando el 50% de ruido central.

### 3. Modelado: El Desafío del Data Leakage
Durante la fase de desarrollo, detectamos una **fuga de datos (data leakage)** crítica: el uso de características de la vela (mechas) que no están disponibles al momento de la predicción.
* **Corrección:** Se re-entrenaron los modelos estrictamente con información *ex-ante* (`Noticia` y `Sorpresa`).
* **Pivote Estratégico:** Los resultados honestos mostraron que los modelos complejos (`GradientBoosting`) se sobreajustaban al ruido. Los modelos simples (`LogisticRegression` y `RandomForest`) demostraron ser más robustos y generalizables.

---

## 🏆 Resultados del Modelo

Logramos obtener una **ventaja estadística real (Alpha)** superior al azar (0.50) en los cuatro mercados analizados en el set de prueba (Test Set).

| Grupo de Activo | Modelo Ganador | F1-Score | Diagnóstico |
| :--- | :--- | :--- | :--- |
| **₿ Cripto (BTC)** | `Logistic Regression (Optimized)` | **0.56** | Alta volatilidad, curva de reacción compleja (W). El tuning fue clave. |
| **🥇 Commodities (Oro)** | `Gradient Boosting (Regularized)` | **0.54** | Señal clara y no-lineal (Curva en V). Único activo donde la complejidad sumó valor. |
| **🏢 Índices (S&P500)** | `Logistic Regression (Baseline)` | **0.52** | Mercado altamente eficiente. El modelo simple fue el más robusto. |
| **💱 Forex (EURUSD)** | `Random Forest (Baseline)` | **0.51** | Señal ruidosa. El ensamble de árboles ayudó a estabilizar la predicción. |

---

## 💻 Instalación y Ejecución Local

Si deseas correr este proyecto en tu máquina local:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/macro-impact-analyzer.git](https://github.com/TU_USUARIO/macro-impact-analyzer.git)
   cd macro-impact-analyzer
2. **Crear un entorno virtual (Recomendado: Python 3.11):**
   Es crucial usar Python 3.11 para garantizar la compatibilidad con las librerías utilizadas (especialmente scikit-learn 1.2.2).
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # En Windows
   # source venv/bin/activate  # En Mac/Linux
3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
4. **Ejecutar la aplicación:**
   ```bash
   streamlit run app.py

## 📂 Estructura del Repositorio
├── app.py                        # Aplicación principal (Streamlit Dashboard)
├── requirements.txt              # Dependencias exactas (Python 3.11 + Sklearn 1.2.2)
├── dataset_completo_procesado.parquet  # Dataset final con Z-Scores y Retornos (Output del ETL)
├── feature_info.pkl              # Metadatos de las columnas de entrenamiento
├── preprocessor.joblib           # Pipeline de preprocesamiento (Escalado + Encoding)
├── model_indices.joblib          # Modelo entrenado para S&P 500
├── model_cripto.joblib           # Modelo entrenado para Bitcoin
├── model_forex.joblib            # Modelo entrenado para EURUSD
├── model_commodities.joblib      # Modelo entrenado para Oro
└── README.md                     # Documentación del proyecto

## 📞 Contacto
Este proyecto fue realizado como trabajo final integrador para la carrera de Ciencia de Datos. Si tienes dudas sobre la metodología, el código o los hallazgos sobre la no-linealidad de los mercados, no dudes en contactarme.
