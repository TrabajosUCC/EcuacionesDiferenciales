"""Simulador interactivo de sistemas de oscilación.  Ejecutar:  python -m streamlit run app.py"""
import streamlit as st

from src.servicio import ErrorValidacion, config_defecto, ejecutar, tabla_resultados
from src.sistemas import listar
from src.solucionadores.integradores import EULER, METODOS, RK4, SCIPY
from src.visualizaciones.render import a_plotly

st.set_page_config(page_title="Simulador de oscilaciones", page_icon="🌊", layout="wide")
st.title("🌊 Simulador interactivo de sistemas de oscilación")
st.caption("Ecuaciones diferenciales ordinarias · soluciones analíticas y numéricas (Euler, RK4, SciPy)")

SISTEMAS = {s.nombre: s for s in listar()}
nombre = st.sidebar.selectbox("Seleccionar sistema de oscilación", list(SISTEMAS))
sis = SISTEMAS[nombre]
st.sidebar.markdown(sis.descripcion)

if "resultado" not in st.session_state:
    st.session_state["resultado"] = None


def mostrar_figuras(figs, prefijo):
    cols = st.columns(2)
    for n, spec in enumerate(figs):
        with cols[n % 2]:
            st.plotly_chart(a_plotly(spec), key=f"{prefijo}_{sis.clave}_{n}")


def mostrar_bloques(bloques):
    for tipo, txt in bloques:
        if tipo == "h":
            st.markdown(f"#### {txt}")
        elif tipo == "t":
            st.markdown(txt)
        else:
            st.latex(txt)


def resultado_vigente():
    r = st.session_state["resultado"]
    if r is None or r["clave"] != sis.clave:
        st.info("Configura los parámetros en la pestaña **Configuración** y pulsa **Calcular**.")
        return None
    for a in r["avisos"]:
        st.warning(a)
    return r


tab_cfg, tab_dev, tab_res, tab_gra, tab_cmp = st.tabs(
    ["Configuración", "Desarrollo matemático", "Resultados", "Gráficas", "Comparación"])

with tab_cfg:
    st.subheader(sis.nombre)
    st.markdown(sis.descripcion)
    with st.form(f"form_{sis.clave}"):
        st.markdown("**Parámetros físicos y condiciones iniciales**")
        cols = st.columns(3)
        valores = {}
        for n, c in enumerate(sis.campos):
            with cols[n % 3]:
                valores[c.clave] = st.number_input(
                    f"{c.etiqueta} ({c.unidad})", value=float(c.defecto), format="%.6g",
                    key=f"{sis.clave}_{c.clave}", help=c.ayuda or None)
        st.markdown("**Simulación numérica**")
        d = config_defecto(sis)
        c1, c2, c3, c4, c5 = st.columns(5)
        metodo = c1.selectbox("Método numérico", METODOS, index=METODOS.index(RK4), key=f"{sis.clave}_met")
        T = c2.number_input("Tiempo de simulación (s)", value=float(d["T"]), format="%.6g", key=f"{sis.clave}_T")
        h = c3.number_input("Paso temporal h (s)", value=float(d["h"]), format="%.6g", key=f"{sis.clave}_h",
                            help="Usado por Euler y RK4 y como malla de salida de solve_ivp.")
        rtol = c4.number_input("Tolerancia relativa", value=d["rtol"], format="%.3g", key=f"{sis.clave}_rt",
                               help="Solo aplica a SciPy solve_ivp.")
        atol = c5.number_input("Tolerancia absoluta", value=d["atol"], format="%.3g", key=f"{sis.clave}_at",
                               help="Solo aplica a SciPy solve_ivp.")
        enviado = st.form_submit_button("Calcular", type="primary")
    if enviado:
        try:
            with st.spinner("Calculando..."):
                st.session_state["resultado"] = ejecutar(
                    sis, valores, {"T": T, "h": h, "metodo": metodo, "rtol": rtol, "atol": atol})
            st.success("Cálculo completado. Revisa las demás pestañas.")
        except ErrorValidacion as e:
            st.session_state["resultado"] = None
            for msg in e.errores:
                st.error(msg)

with tab_dev:
    r = resultado_vigente()
    if r:
        mostrar_bloques(r["desarrollo"])

with tab_res:
    r = resultado_vigente()
    if r:
        st.markdown(f"Método usado: **{r['cfg']['metodo']}** · h = {r['cfg']['h']:.6g} s · T = {r['cfg']['T']:.6g} s")
        st.markdown("#### Magnitudes calculadas")
        cols = st.columns(3)
        for n, (et, val, un) in enumerate(r["derivadas"]):
            with cols[n % 3]:
                txt = f"{val:.6g}" if isinstance(val, (int, float)) else str(val)
                st.metric(et, f"{txt} {un}".strip())
        st.markdown("#### Interpretación física")
        for linea in r["interpretacion"]:
            st.markdown(f"- {linea}")
        st.markdown("#### Tabla de resultados")
        st.dataframe(tabla_resultados(r), hide_index=True)

with tab_gra:
    r = resultado_vigente()
    if r:
        mostrar_figuras(r["figuras"], "g")

with tab_cmp:
    r = resultado_vigente()
    if r:
        st.markdown(f"Referencia de comparación: **{r['ref_nombre']}**."
                    + ("" if r["tiene_analitica"] else " Este modelo no tiene solución cerrada sencilla: se usa una solución numérica."))
        st.dataframe(r["comparacion"], hide_index=True)
        mostrar_figuras(r["figuras_comparacion"], "c")
