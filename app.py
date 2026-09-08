import streamlit as st
import pandas as pd
import altair as alt
import joblib
import pickle
import numpy as np
import os

from src.features import parse_numeric_value, obtener_umbrales_cuartiles, clasificar_magnitud

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

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Inicio y Conceptos", 
    "🏆 Resultados y Justificación", 
    "🧩 Matriz de Cuadrantes", 
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

    # --- BARRAS (esto ya sabemos que anda) ---
    bars = (
        alt.Chart(df_camp)
        .mark_bar(color='#4e79a7')
        .encode(
            x=alt.X('Grupo:N', title='Grupo'),
            y=alt.Y('F1_Score:Q', title='F1-Score (Macro)')
        )
    )

    # --- LÍNEA ROJA HORIZONTAL EN 0.5 ---
    rule_df = pd.DataFrame({'y': [0.5]})
    rule = (
        alt.Chart(rule_df)
        .mark_rule(color='red', strokeDash=[5, 5])
        .encode(
            y='y:Q'
        )
    )

    chart_f1 = (bars + rule).properties(height=300)

    st.altair_chart(chart_f1, use_container_width=True)
    st.divider()


    # 2. GRÁFICO INFERIOR (VALIDACIÓN)
    st.subheader("2. Calidad de la Señal (Retorno Real Promedio)")
    st.markdown("""
    Este gráfico responde: "Cuando el modelo apuesta con convicción, ¿gana dinero?"
    
    * Filtro de Calidad: Aplicamos un umbral de confianza eliminando el ruido donde el modelo es indeciso.
    * Verde (Long): Retorno promedio cuando predice SUBA.
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
                
                res_validacion.append({'Grupo': name, 'Tipo': '🟢 Predicción Alcista', 'Retorno Promedio (%)': avg_ret_long})
                res_validacion.append({'Grupo': name, 'Tipo': '🔴 Predicción Bajista', 'Retorno Promedio (%)': avg_ret_short})
                
            except: pass
            
        if res_validacion:
            df_validacion = pd.DataFrame(res_validacion)
            
            chart_val = alt.Chart(df_validacion).mark_bar().encode(
                x=alt.X('Grupo:N', title=None, axis=None),
                y=alt.Y('Retorno Promedio (%)', title='Retorno Real Promedio (%)'),
                color=alt.Color('Tipo', scale=alt.Scale(domain=['🟢 Predicción Alcista', '🔴 Predicción Bajista'], range=['#2ca02c', '#d62728']), legend=alt.Legend(title="Señal del Modelo")),
                column=alt.Column('Grupo:N', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
                tooltip=['Grupo', 'Tipo', alt.Tooltip('Retorno Promedio (%)', format='.3f')]
            ).properties(
                width=130, 
                height=300
            ).configure_view(stroke='transparent')

            st.altair_chart(chart_val)
            st.info("💡 Interpretación: Barras Verdes positivas y Rojas negativas confirman que el modelo genera Alpha (ganancia).")
        else:
            st.warning("No hay suficientes datos.")

# --- PESTAÑA 3: MATRIZ DE CUADRANTES (FINAL - UNO A LA VEZ) ---
with tab3:
    st.header("🧩 Mapa de Divergencia")
    st.markdown("¿Cómo leer esto?** Verdes (Sube), Rojos (Baja). Use el selector para ver la reacción a noticias Malas vs. Buenas.")

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
    
    # --- 3. Configuración de Escala de Color (CLAMPED) ---
    domain_limit = 0.1 
    
    # --- 4. Gráfico ---
    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X('noticia_clean:N', title='Noticia Económica', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Activo:N', title='Grupo de Activo'),
        color=alt.Color('Retorno:Q', 
                       scale=alt.Scale(scheme='redyellowgreen', domain=[-domain_limit, domain_limit], clamp=True),
                       title='Retorno Promedio (%)'
                      ),
        tooltip=['noticia_clean', 'Activo', alt.Tooltip('Retorno', format='.2f')]
    ).properties(
        height=500 
    )
    
    text = heatmap.mark_text(baseline='middle').encode(
        text=alt.Text('Retorno:Q', format='.2f'),
        color=alt.condition(
            (alt.datum.Retorno > 0.5) | (alt.datum.Retorno < -0.5),
            alt.value("white"),
            alt.value("black")
        )
    )

    st.altair_chart(heatmap + text, use_container_width=True)
    
    if "NEGATIVAS" in scenario:
        st.info("📉 Análisis: Si ve VERDE aquí, significa que el activo sube ante malas noticias (Risk-On / Esperanza de tasas bajas).")
    else:
        st.info("📈 Análisis: Si ve ROJO aquí, significa que el activo cae ante buenas noticias (Miedo a tasas altas).")

# --- PESTAÑA 4: BACKTESTING (COMPLETA Y CORREGIDA) ---
with tab4:
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
                chart = alt.Chart(chart_df).mark_line(color='#00FF00').encode(
                    x='Fecha:T', 
                    y=alt.Y('Capital:Q', scale=alt.Scale(zero=False)),
                    tooltip=['Fecha', alt.Tooltip('Capital', format='$,.2f')]
                ).properties(height=350)
                st.altair_chart(chart, use_container_width=True)

                with st.expander("📜 Ver Registro de Operaciones"):
                    st.dataframe(pd.DataFrame(trades_log))

            except Exception as e:
                st.error(f"Error en cálculo: {e}")