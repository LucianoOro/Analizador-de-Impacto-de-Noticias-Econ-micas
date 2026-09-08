import streamlit as st
import pandas as pd
import altair as alt
import joblib
import pickle
import numpy as np
import os

from src.features import parse_numeric_value, obtener_umbrales_cuartiles, clasificar_magnitud
from src.viz import (
    get_palette,
    aplicar_estilo,
    escala_divergente,
    escala_secuencial_azul,
    condicion_texto_divergente,
    condicion_texto_secuencial,
)

# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN INICIAL
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción del Movimiento de Monedas y Activos de Riesgo ante Noticias Económicas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# 2. FUNCIONES DE CARGA
# ----------------------------------------------------------------------
@st.cache_resource
def cargar_modelo(path):
    try: return joblib.load(path)
    except: return None

@st.cache_data
def cargar_datos(path):
    try:
        df = pd.read_parquet(path)
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # --- AGREGADO: LIMPIEZA VISUAL DE NOMBRES ---
        if 'noticia' in df.columns:
            # Crea columna limpia: quita extensión y aplica formato Título (Ej: "Nfp.xlsx" -> "Nfp")
            df['noticia_clean'] = df['noticia'].astype(str).str.replace(r'\.(xlsx|xls|csv)$', '', case=False, regex=True).str.title()
        
        cols_num = ['sorpresa_std', 'USA500IDXUSD_retorno_cuerpo_pct', 
                   'BTCUSD_retorno_cuerpo_pct', 'EURUSD_retorno_cuerpo_pct', 
                   'XAUUSD_retorno_cuerpo_pct']
        for col in cols_num:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except: return pd.DataFrame()

@st.cache_data
def get_stats_dict(df):
    df_stats = df.copy()
    df_stats['actual_numeric'] = df_stats['actual'].apply(parse_numeric_value)
    df_stats['forecast_numeric'] = df_stats['forecast'].apply(parse_numeric_value)
    df_stats['sorpresa_raw'] = df_stats['actual_numeric'] - df_stats['forecast_numeric']
    
    # --- MODIFICADO: Agrupar por nombre limpio ---
    stats = df_stats.groupby('noticia_clean')['sorpresa_raw'].agg(['mean', 'std']).reset_index()
    return stats.set_index('noticia_clean').T.to_dict()

# --- Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_PREPROCESSOR = os.path.join(BASE_DIR, 'preprocessor.joblib')
PATH_DATA = os.path.join(BASE_DIR, 'dataset_completo_procesado.parquet') 
PATH_FEATURES = os.path.join(BASE_DIR, 'feature_info.pkl')

# Carga
preprocessor = cargar_modelo(PATH_PREPROCESSOR)
models = {
    "Índices (S&P500)": cargar_modelo(os.path.join(BASE_DIR, 'model_indices.joblib')),
    "Cripto (BTC)": cargar_modelo(os.path.join(BASE_DIR, 'model_cripto.joblib')),
    "Forex (EURUSD)": cargar_modelo(os.path.join(BASE_DIR, 'model_forex.joblib')),
    "Commodities (Oro)": cargar_modelo(os.path.join(BASE_DIR, 'model_commodities.joblib'))
}
df_completo = cargar_datos(PATH_DATA)
stats_dict = get_stats_dict(df_completo)

# --- Umbrales de cuartiles (mismos criterios usados en el entrenamiento) ---
if not df_completo.empty:
    Q1_SORPRESA, Q3_SORPRESA = obtener_umbrales_cuartiles(df_completo)
else:
    # Respaldo con los cortes empíricos medidos sobre el dataset completo.
    # No se usan sigmas fijos: 'sorpresa_std' tiene curtosis ~62, muy lejos de una normal.
    Q1_SORPRESA, Q3_SORPRESA = -0.262, 0.272

# --- AGREGADO: MAPEO DE NOTICIAS (Limpio -> Sucio) ---
news_map = {}
if not df_completo.empty:
    news_map = dict(zip(df_completo['noticia_clean'], df_completo['noticia']))

try:
    with open(PATH_FEATURES, 'rb') as f:
        feature_info = pickle.load(f)
    numeric_features = feature_info.get("numeric_features", [])
    categorical_features = feature_info.get("categorical_features", [])
except:
    numeric_features, categorical_features = [], []

# Preparamos df_long AQUÍ para que esté disponible en TODAS las pestañas
if not df_completo.empty:
    # --- MODIFICADO: Incluimos 'noticia_clean' en id_vars ---
    df_long = df_completo.melt(
        id_vars=['fecha', 'noticia', 'noticia_clean', 'sorpresa_std', 'magnitud_sorpresa'],
        value_vars=['USA500IDXUSD_retorno_cuerpo_pct', 'BTCUSD_retorno_cuerpo_pct', 'EURUSD_retorno_cuerpo_pct', 'XAUUSD_retorno_cuerpo_pct'],
        var_name='Activo', value_name='Retorno'
    )
    df_long['Activo'] = df_long['Activo'].str.replace('_retorno_cuerpo_pct', '').str.replace('USA500IDXUSD', 'Índices').replace('BTCUSD', 'Cripto').replace('EURUSD', 'Forex').replace('XAUUSD', 'Commodities')
    
    # Limpieza crítica
    df_long['Retorno'] = pd.to_numeric(df_long['Retorno'], errors='coerce')
    df_long['sorpresa_std'] = pd.to_numeric(df_long['sorpresa_std'], errors='coerce')
    df_long = df_long.dropna(subset=['Retorno', 'sorpresa_std'])
else:
    df_long = pd.DataFrame()

# ----------------------------------------------------------------------
# 3. SIDEBAR: SIMULADOR
# ----------------------------------------------------------------------
st.sidebar.header("🎮 Simulador en Vivo")

if not df_completo.empty:
    # --- MODIFICADO: Usamos la lista limpia ---
    noticias_list_clean = sorted(list(news_map.keys()))
    sel_noticia_clean = st.sidebar.selectbox("1. Noticia", noticias_list_clean)
    # Recuperamos el nombre original para el modelo
    sel_noticia_raw = news_map.get(sel_noticia_clean, "")
else:
    sel_noticia_clean = "Cargando..."
    sel_noticia_raw = ""

col_s1, col_s2 = st.sidebar.columns(2)
val_actual = col_s1.number_input("Actual", value=0.0, format="%.2f")
val_forecast = col_s2.number_input("Forecast", value=0.0, format="%.2f")

z_score = 0.0
# --- MODIFICADO: Buscamos en stats_dict usando el nombre limpio ---
if sel_noticia_clean in stats_dict and stats_dict[sel_noticia_clean]['std'] != 0:
    media = stats_dict[sel_noticia_clean]['mean']
    desvio = stats_dict[sel_noticia_clean]['std']
    z_score = (val_actual - val_forecast - media) / desvio

if z_score > 0:
    st.sidebar.success(f"Sorpresa: +{z_score:.2f} Std Dev")
elif z_score < 0:
    st.sidebar.error(f"Sorpresa: {z_score:.2f} Std Dev")
else:
    st.sidebar.info(f"Sorpresa: {z_score:.2f} Std Dev")

if st.sidebar.button("🔮 Predecir Dirección", type="primary"):
    val_magnitud = clasificar_magnitud(z_score, Q1_SORPRESA, Q3_SORPRESA)

    if val_magnitud not in ("Positiva Extrema", "Negativa Extrema"):
        st.sidebar.warning("⚠ Sorpresa moderada. Modelos menos fiables.")

    if numeric_features:
        input_df = pd.DataFrame([{
            'sorpresa_std': z_score,
            'magnitud_sorpresa': val_magnitud,
            # --- MODIFICADO: Pasamos el nombre original al modelo ---
            'noticia': sel_noticia_raw 
        }])[numeric_features + categorical_features]

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎯 Pronóstico:")
        
        for name, model in models.items():
            if model:
                try:
                    pred = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0][pred]
                    icon = "🟢 ALCISTA" if pred == 1 else "🔴 BAJA"
                    if pred == 1:
                        st.sidebar.success(f"{name}\n\n{icon} ({prob:.1%})")
                    else:
                        st.sidebar.error(f"{name}\n\n{icon} ({prob:.1%})")
                except: st.sidebar.write(f"{name}:** Error")



# ----------------------------------------------------------------------
# 4. PÁGINA PRINCIPAL
# ----------------------------------------------------------------------

st.title("📊 Análisis de Impacto Macro en Mercados Financieros")
st.markdown("##### Grupo 16: Emanuel Gomez, Luciano Oro, Julieta Vicente")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Inicio y Conceptos",
    "🏆 Resultados y Justificación",
    "🧩 Matriz de Cuadrantes",
    "🔬 Diagnóstico del Modelo",
    "🧪 Backtesting"
])

# --- PESTAÑA 1: INICIO (Intacta) ---
with tab1:
    st.header("Bienvenido al Analizador de Impacto Macro")
    st.markdown("""
    Esta aplicación es el resultado final de un proyecto integral de Ciencia de Datos aplicado a mercados financieros. 
    Nuestro objetivo fue cuantificar cómo reaccionan diferentes activos financieros ante las noticias económicas más importantes de EE.UU.
    """)
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("### 📐 El Concepto Clave: Sorpresa Estandarizada")
        st.markdown("En los mercados, el valor absoluto no importa. Lo que mueve el precio es la diferencia entre lo que salió y lo que se esperaba.")
    with col2:
        st.markdown("#### ¿Cómo lo calculamos?")
        st.code("Sorpresa = (Dato Real - Pronóstico) / Desviación Estándar Histórica")
        st.markdown("Esto nos da un Z-Score. Nos permite comparar peras con manzanas.")

    # --- SECCIÓN 1B: DISTRIBUCIÓN DE LA SORPRESA (justifica la metodología) ---
    st.markdown("#### 📊 ¿Por qué cuartiles y no un corte fijo en desvíos estándar?")

    if not df_completo.empty:
        paleta_hist = get_palette()
        serie_sorpresa = df_completo['sorpresa_std'].dropna()

        desvio_medido = serie_sorpresa.std()
        curtosis_medida = serie_sorpresa.kurt()  # curtosis de Fisher: una normal da 0
        minimo_medido = serie_sorpresa.min()
        maximo_medido = serie_sorpresa.max()
        limite_eje = 4

        histograma = (
            alt.Chart(df_completo)
            .mark_bar(color=paleta_hist['azul'], clip=True)
            .encode(
                x=alt.X(
                    'sorpresa_std:Q',
                    bin=alt.Bin(maxbins=60),
                    scale=alt.Scale(domain=[-limite_eje, limite_eje]),
                    title='Sorpresa Estandarizada (Z-Score)'
                ),
                y=alt.Y('count():Q', title='Frecuencia'),
                tooltip=[alt.Tooltip('count():Q', title='Frecuencia')]
            )
        )

        linea_q1 = (
            alt.Chart(pd.DataFrame({'x': [Q1_SORPRESA]}))
            .mark_rule(color=paleta_hist['tinta_atenuada'], strokeWidth=1.5, strokeDash=[4, 2])
            .encode(x='x:Q')
        )
        etiqueta_q1 = (
            alt.Chart(pd.DataFrame({'x': [Q1_SORPRESA], 'label': [f'Q1 = {Q1_SORPRESA:.3f}']}))
            .mark_text(align='right', dx=-4, dy=-4, color=paleta_hist['tinta_atenuada'], fontSize=11)
            .encode(x='x:Q', y=alt.value(0), text='label:N')
        )

        linea_q3 = (
            alt.Chart(pd.DataFrame({'x': [Q3_SORPRESA]}))
            .mark_rule(color=paleta_hist['tinta_atenuada'], strokeWidth=1.5, strokeDash=[4, 2])
            .encode(x='x:Q')
        )
        etiqueta_q3 = (
            alt.Chart(pd.DataFrame({'x': [Q3_SORPRESA], 'label': [f'Q3 = {Q3_SORPRESA:.3f}']}))
            .mark_text(align='left', dx=4, dy=-4, color=paleta_hist['tinta_atenuada'], fontSize=11)
            .encode(x='x:Q', y=alt.value(0), text='label:N')
        )

        chart_hist = (histograma + linea_q1 + etiqueta_q1 + linea_q3 + etiqueta_q3).properties(height=280)

        st.altair_chart(aplicar_estilo(chart_hist, paleta_hist), use_container_width=True)

        st.caption(
            f"Distribución medida sobre {len(serie_sorpresa)} observaciones: desvío estándar "
            f"{desvio_medido:.4f}, pero curtosis de Fisher {curtosis_medida:.1f} (una distribución "
            f"normal da 0). El rango real va de {minimo_medido:.2f} a {maximo_medido:.2f}; el eje se "
            f"recorta en ±{limite_eje} solo para que se pueda leer. Por eso la segmentación de "
            "'magnitud_sorpresa' usa cuartiles empíricos (no paramétricos) en lugar de un corte fijo "
            "en desvíos estándar, que asumiría una normalidad que estos datos claramente no tienen."
        )

    st.divider()

    # --- SECCIÓN 2: DICCIONARIO DE ACTIVOS ---
    st.subheader("🌍 Los 4 Grupos de Activos Analizados")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("### 🏢 Índices")
        st.markdown("Activo: S&P 500 (US500)")
        st.caption("Representa la salud de las empresas de EE.UU. Sensible al crecimiento y tasas.")
    with c2:
        st.markdown("### ₿ Cripto")
        st.markdown("Activo: Bitcoin (BTC)")
        st.caption("Activo de riesgo especulativo. Alta volatilidad. Reacciona fuerte a la liquidez.")
    with c3:
        st.markdown("### 💱 Forex")
        st.markdown("Activo: Euro/Dólar (EURUSD)")
        st.caption("El mercado más líquido del mundo. Reacciona directamente a la fuerza del Dólar.")
    with c4:
        st.markdown("### 🥇 Commodities")
        st.markdown("Activo: Oro (XAUUSD)")
        st.caption("Activo refugio y cobertura contra inflación. Reacciona opuesto a las tasas reales.")

    st.divider()

    # --- SECCIÓN 3: DICCIONARIO DE NOTICIAS ---
    st.subheader("📰 Guía de Eventos Macroeconómicos")
    
    with st.expander("Ver Diccionario de Noticias Detallado", expanded=True):
        st.markdown("""
        | Noticia (Evento) | Nombre Completo | ¿Qué mide? | Impacto Típico |
        | :--- | :--- | :--- | :--- |
        | NFP | Non-Farm Payrolls | Empleo creado en EE.UU. (excl. agro). | Muy Alto. Es el dato rey. Define la salud de la economía. |
        | CPI | Consumer Price Index | Inflación que pagan los consumidores. | Alto. Determina si la Fed sube o baja tasas. |
        | PPI | Producer Price Index | Inflación mayorista (costos). | Medio. Adelanta lo que pasará con el CPI. |
        | GDP | Gross Domestic Product | Crecimiento total de la economía. | Medio. Dato "retrasado", pero confirma recesión/expansión. |
        | Unemployment | Tasa de Desempleo | Porcentaje de gente sin trabajo. | Alto. Parte del mandato dual de la Fed. |
        | Avg Hourly Earnings| Average Hourly Earnings | Inflación de salarios (costo laboral). | **Alto. Si los sueldos suben, la inflación es difícil de bajar. |
        | PMI | Purchasing Managers' Index | Encuesta a gerentes de compras. | Medio. El mejor indicador adelantado de recesión. |
        | FOMC | Federal Funds Rate | Decisión de tipos de interés. | Extremo. El costo del dinero. Mueve todo. |
        """)
        
    st.info("👈 Instrucciones: Utilice el menú lateral para simular una noticia en tiempo real y ver la predicción de nuestros modelos.")

# --- PESTAÑA 2: RESULTADOS Y VALIDACIÓN (CON FILTRO DE CONFIANZA) ---
with tab2:
    st.header("🏆 Resultados Finales y Validación")
    
    # 1. GRÁFICO SUPERIOR: F1-SCORES
    st.subheader("1. Precisión del Modelo (F1-Score)")
    st.markdown("Este gráfico muestra qué tan bien clasifica el modelo (aciertos vs. errores) comparado con el azar.")

    data_campeones = {
        'Grupo': ['Índices', 'Cripto', 'Forex', 'Commodities'],
        'F1_Score': [0.5202, 0.5571, 0.5094, 0.5358],
        'Modelo': ['LogisticReg (Base)', 'LogisticReg (Optimizado)', 'RandomForest (Base)', 'GradientBoost (Reg)']
    }
    df_camp = pd.DataFrame(data_campeones)
    df_camp['Azar'] = 0.5  # punto de partida del vástago (borde sobre el azar)

    paleta = get_palette()
    orden_grupos = df_camp.sort_values('F1_Score', ascending=False)['Grupo'].tolist()
    escala_f1 = alt.Scale(zero=False, domain=[0.48, 0.58])
    eje_f1 = alt.Axis(values=[0.48, 0.50, 0.52, 0.54, 0.56, 0.58])

    # --- GRÁFICO DE LOLLIPOP: una barra codifica por longitud desde cero,   ---
    # --- y estos valores son casi idénticos entre sí; truncar el eje de    ---
    # --- una barra sería engañoso. En cambio, cambiamos la marca: el       ---
    # --- vástago codifica el margen sobre el azar (0,50) y el punto marca  ---
    # --- el valor exacto.                                                  ---

    # Vástago: del azar (0.5) al valor real
    vastago = (
        alt.Chart(df_camp)
        .mark_rule(color=paleta['azul'], strokeWidth=2)
        .encode(
            y=alt.Y('Grupo:N', sort=orden_grupos, title=None),
            x=alt.X('Azar:Q', scale=escala_f1, title='F1-Score (Macro)', axis=eje_f1),
            x2='F1_Score:Q'
        )
    )

    # Punto en el valor exacto
    puntos = (
        alt.Chart(df_camp)
        .mark_point(filled=True, size=140, color=paleta['azul'])
        .encode(
            y=alt.Y('Grupo:N', sort=orden_grupos, title=None),
            x=alt.X('F1_Score:Q', scale=escala_f1),
            tooltip=['Grupo', alt.Tooltip('F1_Score:Q', format='.4f'), 'Modelo']
        )
    )

    # Etiqueta directa del valor (en tinta atenuada, no en el color de la serie)
    etiquetas = (
        alt.Chart(df_camp)
        .mark_text(align='left', dx=10, color=paleta['tinta_atenuada'], fontSize=12)
        .encode(
            y=alt.Y('Grupo:N', sort=orden_grupos, title=None),
            x=alt.X('F1_Score:Q', scale=escala_f1),
            text=alt.Text('F1_Score:Q', format='.4f')
        )
    )

    # Línea de referencia del azar: hairline sólida y neutra, no roja/punteada
    referencia_azar = (
        alt.Chart(pd.DataFrame({'x': [0.5]}))
        .mark_rule(color=paleta['tinta_atenuada'], strokeWidth=1)
        .encode(x=alt.X('x:Q', scale=escala_f1))
    )
    texto_azar = (
        alt.Chart(pd.DataFrame({'x': [0.5], 'label': ['Azar (0,50)']}))
        .mark_text(align='left', dx=6, dy=-8, color=paleta['tinta_atenuada'], fontSize=11)
        .encode(x=alt.X('x:Q', scale=escala_f1), y=alt.value(0), text='label:N')
    )

    chart_f1 = (referencia_azar + texto_azar + vastago + puntos + etiquetas).properties(width=560, height=200)

    st.altair_chart(aplicar_estilo(chart_f1, paleta))
    st.divider()


    # 2. GRÁFICO INFERIOR (VALIDACIÓN)
    st.subheader("2. Calidad de la Señal (Retorno Real Promedio)")
    st.markdown("""
    Este gráfico responde: "Cuando el modelo apuesta con convicción, ¿gana dinero?"
    
    * Filtro de Calidad: Aplicamos un umbral de confianza eliminando el ruido donde el modelo es indeciso.
    * Azul (Long): Retorno promedio cuando predice SUBA.
    * Rojo (Short): Retorno promedio cuando predice BAJA.
    """)

    if not df_completo.empty:
        
        # Filtro datos extremos
        df_val = df_completo[df_completo['magnitud_sorpresa'].isin(['Negativa Extrema', 'Positiva Extrema'])].copy()
        
        res_validacion = []
        
        mapa_ret = {
            "Índices (S&P500)": "USA500IDXUSD_retorno_cuerpo_pct",
            "Cripto (BTC)": "BTCUSD_retorno_cuerpo_pct",
            "Forex (EURUSD)": "EURUSD_retorno_cuerpo_pct",
            "Commodities (Oro)": "XAUUSD_retorno_cuerpo_pct"
        }
        
        for name, model in models.items():
            if not model: continue
            col_ret = mapa_ret[name]
            
            # --- LÓGICA DE UMBRAL PERSONALIZADA ---
            # Índices es más ruidoso, requiere mayor confianza para operar
            umbral_confianza = 0.7 if "Índices" in name else 0.50
            
            try:
                X_val = df_val[numeric_features + categorical_features]
                probs = model.predict_proba(X_val)
                preds = model.predict(X_val)
                
                df_temp = df_val.copy()
                df_temp['Prediccion'] = preds
                df_temp['Retorno_Real'] = df_temp[col_ret]
                df_temp['Confianza'] = [max(p) for p in probs]
                
                # --- APLICAR FILTRO ESPECÍFICO ---
                df_temp_filtered = df_temp[df_temp['Confianza'] > umbral_confianza]
                
                avg_ret_long = df_temp_filtered[df_temp_filtered['Prediccion'] == 1]['Retorno_Real'].mean()
                avg_ret_short = df_temp_filtered[df_temp_filtered['Prediccion'] == 0]['Retorno_Real'].mean()
                
                if pd.isna(avg_ret_long): avg_ret_long = 0.0
                if pd.isna(avg_ret_short): avg_ret_short = 0.0
                
                res_validacion.append({'Grupo': name, 'Tipo': 'Predicción Alcista', 'Retorno Promedio (%)': avg_ret_long})
                res_validacion.append({'Grupo': name, 'Tipo': 'Predicción Bajista', 'Retorno Promedio (%)': avg_ret_short})

            except: pass

        if res_validacion:
            df_validacion = pd.DataFrame(res_validacion)

            escala_senal = alt.Scale(
                domain=['Predicción Alcista', 'Predicción Bajista'],
                range=[paleta['azul'], paleta['rojo']]
            )

            barras_val = (
                alt.Chart(df_validacion)
                .mark_bar(size=26)
                .encode(
                    x=alt.X('Grupo:N', title=None, axis=alt.Axis(labelAngle=0)),
                    xOffset=alt.XOffset('Tipo:N', sort=['Predicción Alcista', 'Predicción Bajista']),
                    y=alt.Y('Retorno Promedio (%):Q', title='Retorno Real Promedio (%)'),
                    color=alt.Color('Tipo:N', scale=escala_senal, legend=alt.Legend(title="Señal del Modelo")),
                    tooltip=['Grupo', 'Tipo', alt.Tooltip('Retorno Promedio (%):Q', format='.3f')]
                )
            )

            linea_cero = (
                alt.Chart(pd.DataFrame({'y': [0]}))
                .mark_rule(color=paleta['tinta_atenuada'], strokeWidth=1)
                .encode(y='y:Q')
            )

            chart_val = (linea_cero + barras_val).properties(width=620, height=300)

            st.altair_chart(aplicar_estilo(chart_val, paleta))
            st.info("💡 Interpretación: Barras Azules positivas y Rojas negativas confirman que el modelo genera Alpha (ganancia).")
        else:
            st.warning("No hay suficientes datos.")

# --- PESTAÑA 3: MATRIZ DE CUADRANTES (FINAL - UNO A LA VEZ) ---
with tab3:
    st.header("🧩 Mapa de Divergencia")
    st.markdown("¿Cómo leer esto?** Azules (Sube), Rojos (Baja). Use el selector para ver la reacción a noticias Malas vs. Buenas.")

    # --- 1. Selector de Escenario ---
    scenario = st.radio(
        "Seleccione el Escenario:", 
        ["📉 Sorpresas NEGATIVAS (Noticia Mala)", "📈 Sorpresas POSITIVAS (Noticia Buena)"],
        horizontal=True
    )
    
    target_magnitud = "Negativa Extrema" if "NEGATIVAS" in scenario else "Positiva Extrema"

    # --- 2. Filtrar Datos ---
    df_quad = df_long[df_long['magnitud_sorpresa'] == target_magnitud]
    
    # Agrupamos por 'noticia_clean' para que el gráfico use el nombre limpio
    # MODIFICADO: usamos 'noticia_clean' en el groupby
    heatmap_data = df_quad.groupby(['noticia_clean', 'Activo'], observed=True)['Retorno'].mean().reset_index()
    
    # --- 3. Configuración de Escala de Color (CLAMPED, tramos seguros en LAB) ---
    paleta = get_palette()
    domain_limit = 0.1

    # El color de texto se deriva del mismo domain_limit que la escala de
    # color (ver `condicion_texto_divergente` en src/viz.py): negativo =
    # rojo, cero = gris neutro, positivo = azul. Así ambos no pueden
    # desalinearse, y además el texto sigue la claridad del RELLENO (no la
    # magnitud del valor a secas), que es distinta en claro y en oscuro
    # (ver D2: antes el umbral era fijo en 0.5 mientras la escala saturaba
    # en 0.1, dejando texto negro sobre celdas ya saturadas de color; y en
    # oscuro el texto oscuro caía sobre relleno oscuro cerca del centro).
    escala_color = escala_divergente(paleta, domain_limit)

    # --- 3b. Grid completo: hace explícitas las celdas sin datos (A3) ---
    # Sin esto, una combinación noticia x activo sin filas (ej. Cripto x GDP en
    # este escenario) simplemente no aparece en heatmap_data y Vega pinta el
    # color de fondo ahí: se lee como un agujero negro, no como "sin datos".
    orden_x = sorted(df_long['noticia_clean'].unique())
    orden_y = sorted(df_long['Activo'].unique())
    grid_completo = pd.DataFrame(
        [(n, a) for n in orden_x for a in orden_y],
        columns=['noticia_clean', 'Activo']
    )
    heatmap_data = grid_completo.merge(heatmap_data, on=['noticia_clean', 'Activo'], how='left')
    heatmap_validos = heatmap_data.dropna(subset=['Retorno'])
    heatmap_nulos = heatmap_data[heatmap_data['Retorno'].isna()]

    # --- 4. Gráfico ---
    # labelLimit generoso + padding inferior para que nombres largos como
    # "Average Hourly Earnings" no se corten a -45° (A5).
    eje_x_heatmap = alt.Axis(labelAngle=-45, labelLimit=220, labelPadding=4)

    celdas = alt.Chart(heatmap_validos).mark_rect(stroke=paleta['superficie'], strokeWidth=2).encode(
        x=alt.X('noticia_clean:N', title='Noticia Económica', sort=orden_x, axis=eje_x_heatmap),
        y=alt.Y('Activo:N', title='Grupo de Activo', sort=orden_y),
        color=alt.Color('Retorno:Q', scale=escala_color, title='Retorno Promedio (%)'),
        tooltip=['noticia_clean', 'Activo', alt.Tooltip('Retorno', format='.2f')]
    )

    texto = celdas.mark_text(baseline='middle').encode(
        text=alt.Text('Retorno:Q', format='.2f'),
        color=condicion_texto_divergente(paleta, domain_limit, 'Retorno')
    )

    # Celdas sin datos: color de superficie + borde fino + etiqueta "s/d",
    # para que la ausencia se lea como ausencia, no como un agujero negro.
    celdas_nulas = alt.Chart(heatmap_nulos).mark_rect(
        fill=paleta['superficie'], stroke=paleta['grilla'], strokeWidth=1
    ).encode(
        x=alt.X('noticia_clean:N', sort=orden_x),
        y=alt.Y('Activo:N', sort=orden_y),
        tooltip=[alt.Tooltip('noticia_clean:N', title='Noticia'), alt.Tooltip('Activo:N')]
    )

    texto_nulo = celdas_nulas.mark_text(
        baseline='middle', fontStyle='italic', fontSize=11, color=paleta['tinta_atenuada']
    ).encode(text=alt.value('s/d'))

    heatmap_final = (celdas_nulas + texto_nulo + celdas + texto).properties(
        width=alt.Step(96), height=alt.Step(60), padding={"bottom": 40}
    )

    st.altair_chart(aplicar_estilo(heatmap_final, paleta), use_container_width=True)

    if "NEGATIVAS" in scenario:
        st.info("📉 Análisis: Si ve AZUL aquí, significa que el activo sube ante malas noticias (Risk-On / Esperanza de tasas bajas).")
    else:
        st.info("📈 Análisis: Si ve ROJO aquí, significa que el activo cae ante buenas noticias (Miedo a tasas altas).")

# --- PESTAÑA 4: DIAGNÓSTICO DEL MODELO (CALIBRACIÓN Y MATRIZ DE CONFUSIÓN) ---
with tab4:
    st.header("🔬 Diagnóstico del Modelo")

    st.warning(
        "⚠ Honestidad metodológica: estos diagnósticos se calculan sobre TODO el "
        "histórico de sorpresas extremas (Negativa/Positiva Extrema), no sobre el "
        "holdout cronológico de 140 filas del que salen los F1-Score reportados en "
        "la pestaña 'Resultados y Justificación'. Sirven para inspeccionar cómo se "
        "comporta el modelo, no como una métrica de test. Además, el N de casos "
        "difiere entre mercados porque cada activo tiene su propia cobertura "
        "histórica de precios (ej. BTC arranca más tarde que los demás): el N de "
        "cada mercado se muestra junto a sus gráficos y debe tenerse en cuenta al "
        "comparar un mercado contra otro."
    )

    if df_completo.empty:
        st.info("No hay datos cargados para calcular los diagnósticos.")
    else:
        df_extremos = df_completo[
            df_completo['magnitud_sorpresa'].isin(['Negativa Extrema', 'Positiva Extrema'])
        ].copy()

        mapa_ret_diag = {
            "Índices (S&P500)": "USA500IDXUSD_retorno_cuerpo_pct",
            "Cripto (BTC)": "BTCUSD_retorno_cuerpo_pct",
            "Forex (EURUSD)": "EURUSD_retorno_cuerpo_pct",
            "Commodities (Oro)": "XAUUSD_retorno_cuerpo_pct"
        }

        paleta_diag = get_palette()

        def _preparar_datos_mercado(nombre_mercado):
            """Filas con retorno válido (no nulo, no cero) + predicción/probabilidad del modelo."""
            col_ret = mapa_ret_diag[nombre_mercado]
            modelo = models.get(nombre_mercado)
            if modelo is None:
                return None

            datos = df_extremos.dropna(subset=[col_ret])
            datos = datos[datos[col_ret] != 0.0]
            if datos.empty:
                return None

            X = datos[numeric_features + categorical_features]
            try:
                probs = modelo.predict_proba(X)
                preds = modelo.predict(X)
            except Exception:
                return None

            clases = list(modelo.classes_)
            idx_1 = clases.index(1) if 1 in clases else (1 if probs.shape[1] > 1 else 0)

            return pd.DataFrame({
                'prob_1': probs[:, idx_1],
                'prediccion': preds,
                'real': (datos[col_ret] > 0).astype(int).values
            })

        # --- N real por mercado (nunca hardcodeado: sale de los datos en cada
        # corrida) para poder mostrarlo junto a los gráficos (ver D5). Difiere
        # entre mercados por los nulos de cada columna de precio (ej. el
        # histórico de BTC arranca más tarde que el resto).
        datos_por_mercado = {m: _preparar_datos_mercado(m) for m in models.keys()}
        n_por_mercado = {
            m: (0 if datos_por_mercado[m] is None else len(datos_por_mercado[m]))
            for m in models.keys()
        }
        etiqueta_mercado = {m: f"{m} — n={n_por_mercado[m]}" for m in models.keys()}

        # Filas con retorno exactamente 0: una vela que cierra en su apertura no
        # tiene dirección que acertar, así que se excluyen de estos diagnósticos.
        # Pero SÍ formaron parte del entrenamiento (plegadas a la clase 'baja'),
        # de modo que el diagnóstico corre sobre una población levemente distinta
        # a la de entrenamiento y corresponde declararlo.
        ceros_por_mercado = {
            m: int((df_extremos[mapa_ret_diag[m]] == 0.0).sum())
            for m in models.keys()
        }
        total_ceros = sum(ceros_por_mercado.values())
        if total_ceros:
            detalle = ", ".join(f"{m}: {c}" for m, c in ceros_por_mercado.items() if c)
            st.caption(
                f"Nota metodológica: se excluyen {total_ceros} filas con retorno "
                f"exactamente 0 ({detalle}). Una vela que cierra en su apertura no "
                "tiene dirección que acertar. Esas filas sí estuvieron en el "
                "entrenamiento, plegadas a la clase 'baja'."
            )

        st.markdown("---")

        # ================= B2: CURVA DE CALIBRACIÓN =================
        st.subheader("1. Curva de Calibración")
        st.markdown(
            "Para cada mercado se agrupan las predicciones en deciles según la "
            "probabilidad que el modelo le asignó a 'sube', y se compara contra la "
            "frecuencia real observada en ese grupo. Un punto sobre la diagonal "
            "significa que cuando el modelo dice '70% de confianza', efectivamente "
            "acierta ~70% de las veces: eso es lo que justifica (o no) usar los "
            "umbrales de confianza como filtro en la pestaña 2."
        )

        filas_calibracion = []
        filas_diagonal = []
        for nombre_mercado in models.keys():
            etiqueta = etiqueta_mercado[nombre_mercado]
            # La diagonal usa los mismos nombres de campo que la curva para poder
            # viajar en un unico DataFrame (ver nota en el armado del grafico).
            filas_diagonal.append({'Mercado': etiqueta, 'prob_media': 0.0,
                                   'frecuencia_observada': 0.0, 'tipo': 'diagonal'})
            filas_diagonal.append({'Mercado': etiqueta, 'prob_media': 1.0,
                                   'frecuencia_observada': 1.0, 'tipo': 'diagonal'})

            datos_mercado = datos_por_mercado[nombre_mercado]
            if datos_mercado is None or len(datos_mercado) < 10:
                continue

            n_bins = min(10, datos_mercado['prob_1'].nunique())
            if n_bins < 2:
                continue

            try:
                datos_mercado['decil'] = pd.qcut(datos_mercado['prob_1'], q=n_bins, duplicates='drop')
            except ValueError:
                continue

            resumen = datos_mercado.groupby('decil', observed=True).agg(
                prob_media=('prob_1', 'mean'),
                frecuencia_observada=('real', 'mean'),
                n=('real', 'size')
            ).reset_index(drop=True)
            resumen['Mercado'] = etiqueta
            filas_calibracion.append(resumen)

        if filas_calibracion:
            # IMPORTANTE: la diagonal y la curva viajan en UN SOLO DataFrame y se
            # separan con transform_filter. Si cada capa trae su propio DataFrame,
            # el facet no filtra las capas internas: crea los paneles pero dibuja
            # todos los mercados en cada uno, y los cuatro paneles salen identicos.
            df_calibracion = pd.concat(filas_calibracion, ignore_index=True)
            df_calibracion['tipo'] = 'curva'
            df_plot = pd.concat(
                [df_calibracion, pd.DataFrame(filas_diagonal)],
                ignore_index=True
            )

            base_calibracion = alt.Chart(df_plot).encode(
                x=alt.X('prob_media:Q', title='Probabilidad predicha (media)',
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y('frecuencia_observada:Q', title='Frecuencia real observada',
                        scale=alt.Scale(domain=[0, 1]))
            )

            linea_diagonal = (
                base_calibracion
                .transform_filter(alt.datum.tipo == 'diagonal')
                .mark_line(color=paleta_diag['tinta_atenuada'], strokeDash=[3, 3], strokeWidth=1)
            )

            linea_calibracion = (
                base_calibracion
                .transform_filter(alt.datum.tipo == 'curva')
                .mark_line(color=paleta_diag['azul'], point=alt.OverlayMarkDef(filled=True, size=50))
                .encode(
                    tooltip=[
                        alt.Tooltip('prob_media:Q', format='.2f', title='Prob. predicha'),
                        alt.Tooltip('frecuencia_observada:Q', format='.2f', title='Frecuencia real'),
                        alt.Tooltip('n:Q', title='Casos en el decil')
                    ]
                )
            )

            chart_calibracion = (
                (linea_diagonal + linea_calibracion)
                .properties(width=200, height=200)
                .facet(facet=alt.Facet('Mercado:N', title=None), columns=2)
            )

            st.altair_chart(aplicar_estilo(chart_calibracion, paleta_diag), use_container_width=True)
        else:
            st.info("No hay suficientes casos por mercado para construir la curva de calibración.")

        st.markdown("---")

        # ================= B3: MATRIZ DE CONFUSIÓN =================
        st.subheader("2. Matriz de Confusión por Mercado")
        st.markdown(
            "Conteo de aciertos y errores de dirección. El color codifica magnitud "
            "(cuántos casos cayeron en esa celda) con una única escala de azules: "
            "acá no hay polaridad que comunicar, así que no se usa la escala "
            "divergente rojo/azul. Correcto vs. incorrecto lo dicen las etiquetas de "
            "los ejes, no el color."
        )

        filas_confusion = []
        for nombre_mercado in models.keys():
            datos_mercado = datos_por_mercado[nombre_mercado]
            if datos_mercado is None:
                continue

            df_cm = pd.DataFrame({
                'Prediccion': datos_mercado['prediccion'].map({1: 'Predijo Alcista', 0: 'Predijo Bajista'}),
                'Real': datos_mercado['real'].map({1: 'Real Alcista', 0: 'Real Bajista'})
            })
            conteo = df_cm.groupby(['Prediccion', 'Real'], observed=True).size().reset_index(name='Conteo')
            totales_fila = conteo.groupby('Prediccion')['Conteo'].transform('sum')
            conteo['Porcentaje_Fila'] = conteo['Conteo'] / totales_fila
            conteo['Mercado'] = etiqueta_mercado[nombre_mercado]
            filas_confusion.append(conteo)

        if filas_confusion:
            df_confusion = pd.concat(filas_confusion, ignore_index=True)
            orden_pred = ['Predijo Alcista', 'Predijo Bajista']
            orden_real = ['Real Bajista', 'Real Alcista']

            # Umbral de texto y color derivados del mismo máximo (ver
            # `condicion_texto_secuencial` en src/viz.py, D4): en oscuro el
            # relleno más claro está en el máximo, no en el mínimo, así que
            # la condición se invierte respecto de clara.
            condicion_texto_cm = condicion_texto_secuencial(
                paleta_diag, df_confusion['Conteo'].max(), 'Conteo'
            )

            # eje_real: labelAngle=0 explícito (ver D3) — sin esto Vega
            # auto-rota "Real Alcista"/"Real Bajista" a 90° cuando no
            # entran horizontales en el ancho de la celda.
            eje_real = alt.Axis(labelAngle=0, labelPadding=4)

            celdas_cm = alt.Chart(df_confusion).mark_rect(stroke=paleta_diag['grilla'], strokeWidth=1).encode(
                x=alt.X('Real:N', title=None, sort=orden_real, axis=eje_real),
                y=alt.Y('Prediccion:N', title=None, sort=orden_pred),
                color=alt.Color('Conteo:Q', scale=escala_secuencial_azul(paleta_diag), legend=None),
                tooltip=[
                    'Prediccion', 'Real', 'Conteo',
                    alt.Tooltip('Porcentaje_Fila:Q', format='.0%', title='% de la fila')
                ]
            )

            texto_conteo = celdas_cm.mark_text(baseline='middle', dy=-8, fontSize=13, fontWeight='bold').encode(
                text=alt.Text('Conteo:Q', format='d'),
                color=condicion_texto_cm
            )
            texto_porcentaje = celdas_cm.mark_text(baseline='middle', dy=8, fontSize=10).encode(
                text=alt.Text('Porcentaje_Fila:Q', format='.0%'),
                color=condicion_texto_cm
            )

            chart_confusion = (
                (celdas_cm + texto_conteo + texto_porcentaje)
                # width ampliado (90 -> 130) para que "Real Alcista"/"Real
                # Bajista" quepan horizontales a labelAngle=0 sin solaparse
                # con la celda vecina (ver D3).
                .properties(width=alt.Step(130), height=alt.Step(60))
                .facet(facet=alt.Facet('Mercado:N', title=None), columns=2, spacing=30)
                .properties(padding={"bottom": 20})
            )

            st.altair_chart(aplicar_estilo(chart_confusion, paleta_diag), use_container_width=True)
        else:
            st.info("No hay suficientes casos por mercado para construir la matriz de confusión.")

# --- PESTAÑA 5: BACKTESTING (COMPLETA Y CORREGIDA) ---
with tab5:
    st.subheader("🧪 Simulación de Apuestas Históricas")
    st.markdown("Simule una estrategia de Interés Compuesto sobre los datos reales.")

    # --- 1. Controles de Usuario ---
    col_b1, col_b2, col_b3 = st.columns(3)
    
    with col_b1:
        sim_asset = st.selectbox("1. Seleccione Activo:", list(models.keys()), key='backtest_asset')
        mapa_ret = {
            "Índices (S&P500)": "USA500IDXUSD_retorno_cuerpo_pct",
            "Cripto (BTC)": "BTCUSD_retorno_cuerpo_pct",
            "Forex (EURUSD)": "EURUSD_retorno_cuerpo_pct",
            "Commodities (Oro)": "XAUUSD_retorno_cuerpo_pct"
        }
        col_retorno_sim = mapa_ret[sim_asset]

    with col_b2:
        # --- MODIFICADO: Uso de listas limpias para el filtro ---
        available_news_clean = sorted(df_completo['noticia_clean'].unique())
        default_news_clean = [n for n in ['Nfp', 'Cpi', 'Fomc'] if n in available_news_clean]
        sim_news_clean = st.multiselect("2. Noticias a Operar:", options=available_news_clean, default=default_news_clean)
    
    with col_b3:
        sim_capital = st.number_input("3. Capital Inicial ($)", value=10000)

    # --- 2. Configuración de Riesgo ---
    st.markdown("---")
    st.markdown("#### ⚙ Gestión de Riesgo (Risk Management)")
    col_risk1, col_risk2, col_risk3 = st.columns(3)
    
    with col_risk1:
        sim_umbral = st.slider("Confianza Mínima (%)", 50, 95, 55, key='backtest_conf') / 100.0
    
    with col_risk2:
        use_leverage = st.checkbox("🚀 Activar Apalancamiento Dinámico", help="Aumenta la exposición en operaciones de alta confianza.")
    
    leverage_mult = 1.0
    leverage_thresh = 1.0
    
    if use_leverage:
        with col_risk3:
            leverage_mult = st.slider("Multiplicador (x)", 1.5, 5.0, 2.0, step=0.5)
            leverage_thresh = st.slider("Activar si Confianza > (%)", int(sim_umbral*100), 95, 65) / 100.0
            st.caption(f"Se operará x{leverage_mult} si la confianza supera {leverage_thresh:.0%}")

    st.markdown("---")

    # --- 3. Ejecución ---
    if st.button("🚀 Ejecutar Backtest", type="primary"):
        
        # --- MODIFICADO: Filtrar usando nombres limpios ---
        df_sim = df_completo[
            (df_completo['noticia_clean'].isin(sim_news_clean)) & 
            (df_completo['magnitud_sorpresa'].isin(['Negativa Extrema', 'Positiva Extrema']))
        ].copy().sort_values('fecha')

        if df_sim.empty:
            st.error("⚠ No hay datos históricos para estos filtros.")
        else:
            model_sim = models[sim_asset]
            capital_history = [sim_capital]
            dates_history = [df_sim['fecha'].min()]
            trades_log = []
            wins = 0
            total_trades = 0
            aggr_trades = 0
            current_capital = sim_capital
            
            try:
                # Predicción en lote
                X_sim = df_sim[numeric_features + categorical_features]
                probs = model_sim.predict_proba(X_sim)
                preds = model_sim.predict(X_sim)

                progress_bar = st.progress(0)

                for i in range(len(df_sim)):
                    row = df_sim.iloc[i]
                    ret_real = row[col_retorno_sim]
                    if pd.isna(ret_real) or ret_real == 0.0: continue 
                    
                    prob_max = max(probs[i])
                    pred = preds[i]

                    # Lógica de Entrada
                    if prob_max >= sim_umbral:
                        total_trades += 1
                        
                        # Gestión de Riesgo (Apalancamiento)
                        active_leverage = 1.0
                        is_aggr = False
                        if use_leverage and prob_max >= leverage_thresh:
                            active_leverage = leverage_mult
                            is_aggr = True
                            aggr_trades += 1
                        
                        # Cálculo de PnL (Compuesto)
                        mult = 1 if pred == 1 else -1
                        trade_ret_pct = ret_real * mult * active_leverage
                        
                        old_capital = current_capital
                        current_capital = current_capital * (1 + trade_ret_pct/100)
                        
                        capital_history.append(current_capital)
                        dates_history.append(row['fecha'])

                        if trade_ret_pct > 0: wins += 1

                        trades_log.append({
                            "Fecha": row['fecha'].strftime('%Y-%m-%d'),
                            "Noticia": row['noticia_clean'], # Mostrar nombre limpio
                            "Pred": "🟢 Long" if pred == 1 else "🔴 Short",
                            "Confianza": f"{prob_max:.1%}",
                            "Lev": f"x{active_leverage}" if is_aggr else "x1",
                            "Retorno Activo": f"{ret_real:.2f}%",
                            "PnL Trade": f"{trade_ret_pct:.2f}%",
                            "Capital": f"${current_capital:,.2f}"
                        })

                    progress_bar.progress((i + 1) / len(df_sim))

                # --- Resultados Visuales ---
                total_return = ((current_capital - sim_capital) / sim_capital) * 100
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Capital Final", f"${current_capital:,.2f}")
                m2.metric("Retorno Total", f"{total_return:+.2f}%", delta_color="normal")
                m3.metric("Win Rate", f"{win_rate:.1f}%")
                m4.metric("Trades", f"{total_trades} ({aggr_trades})")

                st.subheader("📈 Crecimiento de Capital (Equity Curve)")
                chart_df = pd.DataFrame({'Fecha': dates_history, 'Capital': capital_history})
                paleta = get_palette()

                base_equity = alt.Chart(chart_df).encode(
                    x=alt.X('Fecha:T', title='Fecha'),
                    y=alt.Y('Capital:Q', scale=alt.Scale(zero=False), title='Capital')
                )

                linea_capital = base_equity.mark_line(color=paleta['azul'], strokeWidth=2)

                # Línea de referencia en el capital inicial: hace visible el punto de equilibrio.
                # La escala del eje no arranca en cero (una línea codifica por posición, no por
                # longitud, así que un eje no-cero es legítimo aquí).
                linea_capital_inicial = (
                    alt.Chart(pd.DataFrame({'y': [sim_capital]}))
                    .mark_rule(color=paleta['tinta_atenuada'], strokeWidth=1)
                    .encode(y='y:Q')
                )

                # Crosshair al pasar el mouse: selector invisible que engancha el punto más
                # cercano en Fecha, más una regla vertical y un punto visibles solo al hacer hover.
                seleccion_hover = alt.selection_point(
                    nearest=True, on='pointerover', fields=['Fecha'], empty=False
                )

                selectores_hover = base_equity.mark_point().encode(
                    opacity=alt.value(0)
                ).add_params(seleccion_hover)

                regla_hover = base_equity.mark_rule(color=paleta['tinta_atenuada']).encode(
                    opacity=alt.condition(seleccion_hover, alt.value(0.6), alt.value(0))
                ).transform_filter(seleccion_hover)

                punto_hover = base_equity.mark_point(color=paleta['azul'], size=60).encode(
                    opacity=alt.condition(seleccion_hover, alt.value(1), alt.value(0)),
                    tooltip=[
                        alt.Tooltip('Fecha:T', title='Fecha'),
                        alt.Tooltip('Capital:Q', title='Capital', format='$,.2f')
                    ]
                )

                chart_equity = (
                    linea_capital_inicial + linea_capital + selectores_hover + regla_hover + punto_hover
                ).properties(height=300)

                st.altair_chart(aplicar_estilo(chart_equity, paleta), use_container_width=True)

                with st.expander("📜 Ver Registro de Operaciones"):
                    st.dataframe(pd.DataFrame(trades_log))

            except Exception as e:
                st.error(f"Error en cálculo: {e}")