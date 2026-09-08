"""
Paleta y configuración visual compartida para los gráficos Altair de la app.

Centraliza los colores ya validados (contraste, daltonismo) y el cromatismo
recesivo (ejes, grillas) para que los cuatro gráficos hablen el mismo idioma
visual y no repitan configuración.

Semántica fija en toda la aplicación:
    - Azul  = alcista / retorno positivo / serie única.
    - Rojo  = bajista / retorno negativo (color de estado convencional).
    - Gris neutro = punto medio / referencia neutral (nunca un tercer matiz).
"""
import tomllib
from pathlib import Path

import streamlit as st
import altair as alt

# --- Superficies ---
SUPERFICIE_CLARA = "#ffffff"
SUPERFICIE_OSCURA = "#0e1117"

# --- Azul (alcista / serie única) ---
AZUL_CLARO = "#2a78d6"
AZUL_OSCURO = "#3987e5"

# --- Rojo (bajista) — mismo valor en ambos modos ---
ROJO = "#d03b3b"

# --- Gris neutro medio (punto medio de escalas divergentes) ---
GRIS_NEUTRO_CLARO = "#f0efec"
GRIS_NEUTRO_OSCURO = "#383835"

# --- Tramos intermedios de la escala divergente (ver A2) ---
# Sin estos tramos, Vega interpola en espacio RGB entre el neutro y los
# extremos: crema -> azul cruza por violeta, crema -> rojo cruza por rosa.
# Se fijan tramos intermedios que se mantienen dentro del mismo matiz
# (verificado en HLS: el rojo intermedio no se corre hacia magenta, el
# azul intermedio no se corre hacia violeta) y además se interpola en
# espacio LAB (interpolate='lab' en la escala) como segunda barrera.
#
# --- Regla que gobierna estos tramos (ver D1) ---
# La claridad (lightness) de los tramos debe ordenarse según la SUPERFICIE,
# no según el matiz:
#   - Sobre superficie OSCURA, la claridad debe AUMENTAR desde el punto
#     medio neutro hacia los dos polos (el neutro es el tramo más oscuro
#     de cada brazo).
#   - Sobre superficie CLARA, la claridad debe DISMINUIR desde el punto
#     medio neutro hacia los dos polos (el neutro es el tramo más claro).
# Elegir el matiz correcto no alcanza: si los tramos no se ordenan así, un
# valor intermedio puede leerse visualmente más intenso que el extremo
# (bug original: en oscuro, AZUL_INTERMEDIO_OSCURO era más claro que
# AZUL_OSCURO, así que +0.05 se veía más "fuerte" que +0.10).
# `AZUL_INTERMEDIO_OSCURO` usa un azul del banda 550/600 (más oscuro que
# el polo `AZUL_OSCURO`, banda 400) para que el brazo azul en oscuro quede
# neutro(oscuro) -> intermedio(azul oscuro) -> polo(azul más claro).
AZUL_INTERMEDIO_CLARO = "#86b6ef"
AZUL_INTERMEDIO_OSCURO = "#184f95"
ROJO_INTERMEDIO_CLARO = "#e09594"
ROJO_INTERMEDIO_OSCURO = "#843a38"

# --- Tinta atenuada: ejes, etiquetas, líneas de referencia (ambos modos) ---
TINTA_ATENUADA = "#898781"

# --- Líneas de grilla ---
GRILLA_CLARA = "#e1e0d9"
GRILLA_OSCURA = "#2c2c2a"


def _tema_fijado_en_config():
    """
    Devuelve 'light'/'dark' si el proyecto fija un tema en
    `.streamlit/config.toml`, o None si no lo fija.

    Hace falta leer el archivo directamente porque `st.get_option("theme.base")`
    devuelve "light" tanto cuando el tema está fijado en claro como cuando no
    está fijado en absoluto: no permite distinguir un valor elegido de un valor
    por defecto.
    """
    ruta = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
    if not ruta.exists():
        return None

    try:
        with ruta.open("rb") as archivo:
            config = tomllib.load(archivo)
    except Exception:
        return None

    base = config.get("theme", {}).get("base")
    return base if base in ("light", "dark") else None


def get_palette() -> dict:
    """
    Devuelve la paleta activa según el tema con el que la app se está
    renderizando (claro/oscuro).

    ORDEN DE PRECEDENCIA (importa, y esta es la razón):

    1. El tema fijado en `.streamlit/config.toml`, si existe. Cuando el
       proyecto fija un tema, ese tema es el que se renderiza para todos los
       visitantes, así que manda sobre cualquier preferencia del cliente.
    2. `st.context.theme.type`, que refleja el tema real del navegador. Es la
       fuente correcta cuando no hay config.toml (el caso de despliegue por
       defecto).
    3. Claro, como último recurso.

    El orden inverso produce un fallo silencioso: `st.context.theme.type` sigue
    la preferencia del sistema operativo del visitante, así que con un tema
    fijado en claro y un visitante con el sistema en oscuro, la página se
    renderiza clara y los gráficos salen con la paleta oscura.
    """
    tema = _tema_fijado_en_config()

    if tema is None:
        try:
            tema = st.context.theme.type
        except Exception:
            tema = None

    if tema not in ("dark", "light"):
        tema = "light"

    es_oscuro = tema == "dark"

    return {
        "es_oscuro": es_oscuro,
        "superficie": SUPERFICIE_OSCURA if es_oscuro else SUPERFICIE_CLARA,
        "azul": AZUL_OSCURO if es_oscuro else AZUL_CLARO,
        "azul_intermedio": AZUL_INTERMEDIO_OSCURO if es_oscuro else AZUL_INTERMEDIO_CLARO,
        "rojo": ROJO,
        "rojo_intermedio": ROJO_INTERMEDIO_OSCURO if es_oscuro else ROJO_INTERMEDIO_CLARO,
        "gris_neutro": GRIS_NEUTRO_OSCURO if es_oscuro else GRIS_NEUTRO_CLARO,
        "tinta_atenuada": TINTA_ATENUADA,
        "grilla": GRILLA_OSCURA if es_oscuro else GRILLA_CLARA,
    }


def escala_divergente(paleta: dict, domain_limit: float) -> alt.Scale:
    """
    Escala de color divergente rojo-gris-azul segura para polaridad
    (retornos negativos/positivos). Usa 5 tramos en lugar de 3 y fuerza
    interpolación en espacio LAB para que el camino cromático nunca
    atraviese violetas ni rosas (ver A2). No usar para magnitud pura:
    para eso está `escala_secuencial_azul()`.
    """
    return alt.Scale(
        domain=[-domain_limit, -domain_limit / 2, 0, domain_limit / 2, domain_limit],
        range=[
            paleta["rojo"],
            paleta["rojo_intermedio"],
            paleta["gris_neutro"],
            paleta["azul_intermedio"],
            paleta["azul"],
        ],
        interpolate="lab",
        clamp=True,
    )


def escala_secuencial_azul(paleta: dict, domain: list | None = None) -> alt.Scale:
    """
    Escala secuencial de un solo matiz (superficie -> azul) para codificar
    magnitud/conteo (ej. celdas de una matriz de confusión). A diferencia
    de `escala_divergente()`, aquí no hay polaridad que comunicar: mayor
    saturación de azul significa simplemente "más casos", nunca "mejor" o
    "peor" (eso lo dicen las etiquetas de los ejes).
    """
    kwargs = {"range": [paleta["superficie"], paleta["azul"]]}
    if domain is not None:
        kwargs["domain"] = domain
    return alt.Scale(**kwargs)


def condicion_texto_divergente(paleta: dict, domain_limit: float, campo: str):
    """
    Color de texto correcto para superponerse a `escala_divergente()`, tal
    que siga la claridad del RELLENO (ver la regla en la sección de tramos
    intermedios, D1/D2) en lugar de la magnitud del valor por sí sola.

    - Modo claro: el neutro central es claro y los polos son saturados ->
      texto oscuro cerca del punto medio, texto claro en los extremos.
    - Modo oscuro: el neutro central es oscuro y los polos son más claros
      (por la regla de D1) -> texto claro cerca del punto medio, texto
      oscuro en los extremos.

    `domain_limit` debe ser el mismo valor pasado a `escala_divergente()`
    para esta misma escala: así el umbral de texto nunca se desalinea del
    umbral de color. `campo` es el nombre de la columna cuantitativa
    codificada en la escala de color (ej. 'Retorno').
    """
    umbral = domain_limit / 2
    valor = getattr(alt.datum, campo)
    es_extremo = (valor >= umbral) | (valor <= -umbral)

    if paleta.get("es_oscuro"):
        # Relleno oscuro en el centro -> texto claro; polos más claros -> texto oscuro.
        return alt.condition(es_extremo, alt.value("black"), alt.value("white"))
    # Relleno claro en el centro -> texto oscuro; polos saturados -> texto claro.
    return alt.condition(es_extremo, alt.value("white"), alt.value("black"))


def condicion_texto_secuencial(paleta: dict, domain_max: float, campo: str):
    """
    Color de texto correcto para superponerse a `escala_secuencial_azul()`,
    hermana de `condicion_texto_divergente()` pero para un dominio 0..max
    en lugar de uno simétrico alrededor de un punto medio.

    El relleno va de `paleta["superficie"]` (valor 0) a `paleta["azul"]`
    (domain_max):
    - Modo claro: superficie clara en 0 -> texto oscuro para valores bajos;
      polo azul saturado en el máximo -> texto claro para valores altos.
    - Modo oscuro: superficie oscura en 0 -> texto claro para valores
      bajos; polo azul más claro en el máximo -> texto oscuro para valores
      altos.

    `domain_max` debe ser el mismo máximo que alimenta la escala (o su
    dominio explícito) para que el umbral de texto no se desalinee del
    color. `campo` es la columna cuantitativa codificada en el color.
    """
    umbral = domain_max / 2
    valor = getattr(alt.datum, campo)
    es_alto = valor > umbral

    if paleta.get("es_oscuro"):
        # Bajo -> relleno oscuro -> texto claro. Alto -> relleno más claro -> texto oscuro.
        return alt.condition(es_alto, alt.value("black"), alt.value("white"))
    # Bajo -> relleno claro -> texto oscuro. Alto -> relleno saturado -> texto claro.
    return alt.condition(es_alto, alt.value("white"), alt.value(paleta["tinta_atenuada"]))


def aplicar_estilo(chart, paleta: dict | None = None):
    """
    Aplica el cromatismo recesivo estándar a un gráfico Altair ya armado
    (ejes y grillas discretos, sin borde de vista). Llamar sobre la capa
    final, justo antes de pasarla a st.altair_chart.
    """
    if paleta is None:
        paleta = get_palette()

    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            grid=True,
            gridColor=paleta["grilla"],
            gridDash=[1, 0],  # sólida: nunca punteada
            domainColor=paleta["grilla"],
            tickColor=paleta["grilla"],
            labelColor=paleta["tinta_atenuada"],
            titleColor=paleta["tinta_atenuada"],
        )
    )
