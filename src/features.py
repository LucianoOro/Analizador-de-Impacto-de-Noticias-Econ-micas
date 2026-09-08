"""
Funciones de ingeniería de características extraídas de forma verbatim
del notebook `notebooks/03_modelado_final.ipynb` (el notebook autoritativo
de modelado, sin fuga de datos).

Estas funciones son el vínculo entre el pipeline de entrenamiento
(`src/train.py`) y la aplicación en vivo (`app.py`): garantizan que la
app clasifique la magnitud de una sorpresa con exactamente los mismos
criterios (cuartiles) que se usaron para entrenar los modelos.
"""

import pandas as pd


def parse_numeric_value(value):
    """
    Función auxiliar para limpiar y convertir valores monetarios/porcentaje
    (ej. '3.5%', '110K', '2.1M') a números.
    """
    if isinstance(value, str):
        value = value.strip().replace('%', '').replace('K', 'e3').replace('M', 'e6').replace('B', 'e9')
    return pd.to_numeric(value, errors='coerce')


def agregar_variables_estandarizadas(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    (De EDA Sección 4.1)
    Calcula 'sorpresa' (Actual vs Forecast) y la estandariza (Z-Score)
    para cada tipo de noticia, creando 'sorpresa_std'.
    """
    df = df_input.copy()
    print("Calculando 'sorpresa_std'...")

    if 'actual' in df.columns and 'forecast' in df.columns:
        df['actual_numeric'] = df['actual'].apply(parse_numeric_value)
        df['forecast_numeric'] = df['forecast'].apply(parse_numeric_value)
        df['sorpresa'] = df['actual_numeric'] - df['forecast_numeric']

        # Estandarizar 'sorpresa' (Z-Score) agrupando por noticia
        df['sorpresa_std'] = df.groupby('noticia')['sorpresa'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
    else:
        print("Advertencia: Faltan columnas 'actual' o 'forecast'.")
        return df_input

    print("'sorpresa_std' calculada.")
    return df


def segmentar_por_magnitud(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    (De EDA Sección 5.1)
    Segmenta 'sorpresa_std' en 4 cuartiles para crear la variable
    categórica 'magnitud_sorpresa'.
    """
    df = df_input.copy()
    print("Segmentando 'magnitud_sorpresa'...")

    if 'sorpresa_std' not in df.columns:
        print("Error: Se requiere la columna 'sorpresa_std' para segmentar.")
        return df_input

    try:
        df['magnitud_sorpresa'] = pd.qcut(
            df['sorpresa_std'],
            q=4,
            labels=['Negativa Extrema', 'Negativa Moderada', 'Positiva Moderada', 'Positiva Extrema']
        )
    except ValueError as e:
        print(f"Advertencia al segmentar en cuartiles: {e}. Puede haber datos insuficientes.")
        df['magnitud_sorpresa'] = pd.NA

    print("'magnitud_sorpresa' segmentada.")
    return df


def obtener_umbrales_cuartiles(df) -> tuple[float, float]:
    """
    Devuelve los cortes Q1 y Q3 de sorpresa_std usados para definir magnitud extrema.
    """
    q1 = df['sorpresa_std'].quantile(0.25)
    q3 = df['sorpresa_std'].quantile(0.75)
    return q1, q3


def clasificar_magnitud(z_score, q1, q3) -> str:
    """
    Clasifica un z-score puntual usando los mismos cortes por cuartiles del entrenamiento.
    """
    if z_score <= q1:
        return "Negativa Extrema"
    elif z_score >= q3:
        return "Positiva Extrema"
    elif z_score < 0:
        return "Negativa Moderada"
    else:
        return "Positiva Moderada"
