"""
Script de entrenamiento que reproduce el pipeline de modelado documentado
en `notebooks/03_modelado_final.ipynb` (el notebook autoritativo, sin fuga
de datos).

Carga el CSV fuente, construye las features, filtra a eventos extremos,
hace un split cronológico 80/20, ajusta el ColumnTransformer solo sobre
train, corre las búsquedas de hiperparámetros documentadas y evalúa sobre
test con `classification_report`.

Por defecto escribe los artefactos en `artifacts/` (no en la raíz del
proyecto), para que correr este script nunca sobreescriba los modelos
desplegados (`model_*.joblib`, `preprocessor.joblib`, `feature_info.pkl`).

Uso:
    ./venv/Scripts/python.exe -m src.train
    ./venv/Scripts/python.exe -m src.train --output-dir artifacts --input data/correlacion_final_2025-09-24.csv
"""

import argparse
import os
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import agregar_variables_estandarizadas, segmentar_por_magnitud

warnings.filterwarnings('ignore')

DEFAULT_INPUT = os.path.join('data', 'correlacion_final_2025-09-24.csv')
DEFAULT_OUTPUT_DIR = 'artifacts'

TARGET_COLS = [
    'target_Indices',
    'target_Cripto',
    'target_Forex',
    'target_Commodities',
]

ACTIVOS_TARGET = {
    'target_Indices': 'USA500IDXUSD_retorno_cuerpo_pct',
    'target_Cripto': 'BTCUSD_retorno_cuerpo_pct',
    'target_Forex': 'EURUSD_retorno_cuerpo_pct',
    'target_Commodities': 'XAUUSD_retorno_cuerpo_pct',
}


def cargar_y_preparar(path_csv: str) -> pd.DataFrame:
    """Carga el CSV fuente, ordena por fecha y construye las features del modelo."""
    df = pd.read_csv(path_csv)
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.dropna(subset=['fecha'])
    df = df.sort_values(by='fecha')

    # Columna 'noticia' requerida por agregar_variables_estandarizadas
    df['noticia'] = df['source_file'].str.replace('.csv', '', regex=False).str.replace('_', ' ').str.title()

    df_procesado = agregar_variables_estandarizadas(df)
    df_procesado = segmentar_por_magnitud(df_procesado)

    for target_name, col_retorno in ACTIVOS_TARGET.items():
        if col_retorno in df_procesado.columns:
            df_procesado[target_name] = (df_procesado[col_retorno] > 0).astype(int)

    return df_procesado


def filtrar_extremos(df_procesado: pd.DataFrame) -> pd.DataFrame:
    """Filtra el dataset a las sorpresas extremas (cuartiles superior/inferior)."""
    df_final = df_procesado[
        df_procesado['magnitud_sorpresa'].isin(['Negativa Extrema', 'Positiva Extrema'])
    ].copy()

    if pd.api.types.is_categorical_dtype(df_final['magnitud_sorpresa']):
        df_final['magnitud_sorpresa'] = df_final['magnitud_sorpresa'].cat.remove_unused_categories()

    return df_final


def construir_columnas(df_final: pd.DataFrame):
    """Define las columnas numéricas/categóricas usadas por el ColumnTransformer.

    Nota: no incluye ratios de mecha (data leakage). Solo features ex-ante.
    """
    numeric_features = ['sorpresa_std']
    categorical_features = ['magnitud_sorpresa', 'noticia']
    feature_cols = numeric_features + categorical_features
    return numeric_features, categorical_features, feature_cols


def split_cronologico(X: pd.DataFrame, Y: pd.DataFrame, test_size: float = 0.2):
    """Split 80/20 cronológico (no aleatorio): asume X/Y ya ordenados por fecha."""
    split_index = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    Y_train = Y.iloc[:split_index]
    Y_test = Y.iloc[split_index:]
    return X_train, X_test, Y_train, Y_test


def construir_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features),
        ],
        remainder='drop',
    )


def entrenar_y_evaluar(preprocessor, X_train, X_test, Y_train, Y_test):
    """Reproduce las 3 etapas de búsqueda de hiperparámetros documentadas en el notebook:

    1. GridSearchCV V1 sobre LogisticRegression (Indices, Cripto, Commodities)
       y RandomForestClassifier (Forex).
    2. GridSearchCV V2 (regularizado) sobre RandomForestClassifier, solo para Forex,
       para mitigar el overfitting detectado en V1.
    3. Commodities: el notebook exporta `best_estimators_v2['target_Commodities']`
       (un GradientBoosting regularizado) pero la celda que CONSTRUYE ese grid
       search fue borrada del notebook guardado (solo existió en el kernel de
       Colab). El modelo desplegado (`model_commodities.joblib`) SÍ es un
       GradientBoostingClassifier limpio (verificado: n_features_in_=3), así que
       ese grid corrió en algún momento, pero no es recuperable del notebook.
       La grilla de abajo es una RECONSTRUCCIÓN razonable, no el grid original.
       El F1 de test reportado en el notebook para Commodities (0.5358) no pudo
       volver a derivarse con esta reconstrucción.

    Devuelve un dict {target_name: (estimator, classification_report_dict, f1_macro)}.
    """
    resultados = {}

    # --- V1: grid search por target, según Celda 21 del notebook ---
    target_configs = {
        'target_Indices': {
            'model': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            'grid': {
                'model__C': [0.01, 0.1, 1.0, 10.0],
                'model__solver': ['liblinear', 'lbfgs'],
            },
        },
        'target_Cripto': {
            'model': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            'grid': {
                'model__C': [0.01, 0.1, 1.0, 10.0],
                'model__solver': ['liblinear', 'lbfgs'],
            },
        },
        'target_Forex': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'grid': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 5, 10],
            },
        },
        'target_Commodities': {
            'model': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            'grid': {
                'model__C': [0.01, 0.1, 1.0, 10.0],
                'model__solver': ['liblinear', 'lbfgs'],
            },
        },
    }

    best_estimators = {}
    for target_name, config in target_configs.items():
        print(f"GridSearchCV V1 para {target_name} ({config['model'].__class__.__name__})...")
        pipeline_base = Pipeline(steps=[('preprocessor', preprocessor), ('model', config['model'])])
        grid_search = GridSearchCV(
            estimator=pipeline_base,
            param_grid=config['grid'],
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
        )
        grid_search.fit(X_train, Y_train[target_name])
        best_estimators[target_name] = grid_search.best_estimator_

    # --- V2: grid search regularizado, solo para Forex (Celda 27 del notebook) ---
    print("GridSearchCV V2 (regularizado) para target_Forex...")
    param_grid_rf_v2 = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [2, 3, 5],
        'model__min_samples_leaf': [5, 10],
    }
    pipeline_base_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ])
    grid_search_rf_v2 = GridSearchCV(
        estimator=pipeline_base_rf,
        param_grid=param_grid_rf_v2,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
    )
    grid_search_rf_v2.fit(X_train, Y_train['target_Forex'])

    # Diagnóstico documentado en el notebook: el baseline (V1) de Forex fue el
    # ganador final porque el tuning (V1 y V2) sufrió overfitting. Mantenemos
    # el estimador baseline (RandomForest sin grid) como campeón de Forex.
    print("Re-entrenando baseline (campeón real) para target_Forex...")
    modelo_forex_baseline = RandomForestClassifier(class_weight='balanced', random_state=42)
    pipeline_forex = Pipeline(steps=[('preprocessor', clone(preprocessor)), ('model', modelo_forex_baseline)])
    pipeline_forex.fit(X_train, Y_train['target_Forex'])
    best_estimators['target_Forex'] = pipeline_forex

    # Índices: el notebook también concluye que el baseline (sin tuning) fue
    # el modelo ganador para Indices.
    print("Re-entrenando baseline (campeón real) para target_Indices...")
    modelo_indices_baseline = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    pipeline_indices = Pipeline(steps=[('preprocessor', clone(preprocessor)), ('model', modelo_indices_baseline)])
    pipeline_indices.fit(X_train, Y_train['target_Indices'])
    best_estimators['target_Indices'] = pipeline_indices

    # --- RECONSTRUCCIÓN: Commodities (GradientBoosting regularizado) ---
    # ADVERTENCIA: esta grilla NO es la original. La celda que construía
    # `best_estimators_v2['target_Commodities']` no está en el notebook
    # guardado (solo existió en el kernel de Colab). Este es un intento
    # razonable de reconstrucción con GradientBoostingClassifier, pero su
    # resultado de F1 no coincidirá necesariamente con el 0.5358 documentado.
    print("RECONSTRUCCIÓN (no original): GridSearchCV GradientBoosting para target_Commodities...")
    param_grid_gb_commodities = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [2, 3],
        'model__learning_rate': [0.01, 0.1],
        'model__min_samples_leaf': [5, 10],
    }
    pipeline_base_gb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(random_state=42)),
    ])
    grid_search_gb_commodities = GridSearchCV(
        estimator=pipeline_base_gb,
        param_grid=param_grid_gb_commodities,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
    )
    grid_search_gb_commodities.fit(X_train, Y_train['target_Commodities'])
    best_estimators['target_Commodities'] = grid_search_gb_commodities.best_estimator_

    # --- Evaluación final sobre test ---
    for target_name, estimator in best_estimators.items():
        y_pred = estimator.predict(X_test)
        reporte = classification_report(Y_test[target_name], y_pred, output_dict=True)
        f1_macro = f1_score(Y_test[target_name], y_pred, average='macro')
        print(f"\n--- {target_name} ---")
        print(classification_report(Y_test[target_name], y_pred))
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        resultados[target_name] = (estimator, reporte, f1_macro)

    return resultados


def guardar_artefactos(output_dir, resultados, numeric_features, categorical_features, feature_cols):
    os.makedirs(output_dir, exist_ok=True)

    # El objeto ColumnTransformer original nunca se fitea directamente: cada
    # Pipeline entrenado en entrenar_y_evaluar() clona y fitea su propia copia
    # sobre X_train. Guardamos el preprocesador YA FITEADO, extraído de uno de
    # los pipelines campeones (todos comparten los mismos datos/columnas de fit).
    algun_estimador = next(iter(resultados.values()))[0]
    preprocessor_fiteado = algun_estimador.named_steps['preprocessor']
    joblib.dump(preprocessor_fiteado, os.path.join(output_dir, 'preprocessor.joblib'))

    nombre_archivo = {
        'target_Indices': 'model_indices.joblib',
        'target_Cripto': 'model_cripto.joblib',
        'target_Forex': 'model_forex.joblib',
        'target_Commodities': 'model_commodities.joblib',
    }
    for target_name, (estimator, _reporte, _f1) in resultados.items():
        joblib.dump(estimator, os.path.join(output_dir, nombre_archivo[target_name]))

    feature_info = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_features": feature_cols,
    }
    with open(os.path.join(output_dir, 'feature_info.pkl'), 'wb') as f:
        pickle.dump(feature_info, f)

    print(f"\nArtefactos guardados en '{output_dir}/' (no se sobreescribieron los modelos desplegados en la raíz).")


def main():
    parser = argparse.ArgumentParser(description="Reproduce el pipeline de entrenamiento del notebook 03_modelado_final.")
    parser.add_argument('--input', default=DEFAULT_INPUT, help="Ruta al CSV fuente (correlacion_final_2025-09-24.csv)")
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help="Directorio donde escribir los artefactos entrenados")
    args = parser.parse_args()

    print(f"Cargando y preparando datos desde '{args.input}'...")
    df_procesado = cargar_y_preparar(args.input)
    df_final = filtrar_extremos(df_procesado)
    print(f"Dataset filtrado a eventos extremos: {len(df_final)} filas (de {len(df_procesado)} totales)")

    numeric_features, categorical_features, feature_cols = construir_columnas(df_final)
    X = df_final[feature_cols]
    Y = df_final[TARGET_COLS]

    X_train, X_test, Y_train, Y_test = split_cronologico(X, Y)
    print(f"Train: {len(X_train)} filas | Test: {len(X_test)} filas (split cronológico 80/20)")

    # El ColumnTransformer se ajusta exclusivamente sobre X_train: cada Pipeline
    # de sklearn.compose que lo incluye lo fitea solo con los datos de entrenamiento
    # que se le pasan (aquí, X_train), nunca con X_test.
    preprocessor = construir_preprocessor(numeric_features, categorical_features)

    resultados = entrenar_y_evaluar(preprocessor, X_train, X_test, Y_train, Y_test)

    guardar_artefactos(args.output_dir, resultados, numeric_features, categorical_features, feature_cols)


if __name__ == '__main__':
    main()
