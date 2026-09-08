# Notebooks del proyecto

Este directorio contiene, en orden cronológico, los tres notebooks originales
del proceso de Ciencia de Datos que dio origen a la app desplegada (`app.py`).
Ninguno fue modificado en su contenido: solo se renombraron y movieron desde
la carpeta de recuperación (`Ciencia de Datos/PROYECTO/`).

## Secuencia

1. **`01_eda.ipynb`** (antes `SegundaEntregaProyectoFINAL.ipynb`, Entrega 2)
   Análisis exploratorio de datos (EDA). Estudia la relación entre las
   sorpresas macroeconómicas y el movimiento de los activos, y es donde se
   originan las funciones de ingeniería de características (luego extraídas
   a `src/features.py`).

   > **Nota:** este archivo pesa ~11.9 MB (por la cantidad de gráficos
   > embebidos). Es posible que **no renderice en el visor web de GitHub**;
   > para revisarlo, descargarlo y abrirlo localmente con Jupyter, o usar
   > [nbviewer](https://nbviewer.org/).

2. **`02_modelado_con_fuga.ipynb`** (antes `TerceraEntregaProyecto.ipynb`)
   Primer intento de modelado. **Se conserva deliberadamente**, no es
   descarte: este notebook contiene la versión del pipeline que usaba
   features de la vela (ratios de "mecha" superior/inferior), las cuales
   filtran información del futuro (*data leakage*). Es la evidencia del
   proceso de detección de la fuga de datos, un hallazgo central del
   proyecto, y por eso se mantiene en el repositorio en vez de eliminarse.

3. **`03_modelado_final.ipynb`** (antes `Copy of TerceraEntregaProyecto.ipynb`)
   **Notebook autoritativo.** A pesar de su nombre original ("Copy of..."),
   esta es la versión que corrigió la fuga de datos (eliminando las features
   de mecha) y produjo los modelos y el preprocesador realmente desplegados
   en la aplicación (`model_*.joblib`, `preprocessor.joblib`). Las funciones
   de ingeniería de características (`parse_numeric_value`,
   `agregar_variables_estandarizadas`, `segmentar_por_magnitud`) fueron
   extraídas verbatim de este notebook hacia `src/features.py`.
