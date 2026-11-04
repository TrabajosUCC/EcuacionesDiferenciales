"""
Demo interactiva opcional del modelo de autoescalado.

Esta aplicación NO forma parte de los entregables obligatorios ni de
`python main.py`: es un complemento para la presentación oral. Permite mover
los parámetros con sliders y ver en vivo el régimen, las métricas y la
comparación entre la solución analítica y los métodos numéricos.

Ejecutar desde la carpeta proyecto/:

    pip install -r requirements-app.txt
    streamlit run app_streamlit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Streamlit puede ejecutar el script desde otro directorio; se añade la carpeta
# del proyecto al path para que `import src...` funcione siempre.
RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.analitica import (
    frecuencia_resonancia,
    metricas_escalon,
    respuesta_frecuencia,
    x_escalon,
    x_sinusoidal,
    y_escalon,
    y_sinusoidal,
)
from src.modelo import BASE, Parametros, entrada_sinusoidal
from src.numerico import euler_explicito, error_max_estado, rk4

st.set_page_config(
    page_title="Autoescalado - Simulación interactiva",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Utilidades de presentación
# ---------------------------------------------------------------------------

def paso_adecuado(p: Parametros) -> float:
    """Elige un paso h que divida T y mantenga estable a Euler.

    Euler explícito es condicionalmente estable: para un par de polos
    complejos necesita h*wn < 2*zeta. Se toma la mitad de ese límite como
    margen y se limita además h*wn <= 0.2 por precisión. Los candidatos
    dividen exactamente cualquier horizonte múltiplo de 50 s.
    """
    limite = 0.5 * p.zeta / p.wn
    for h in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        if h <= limite and h * p.wn <= 0.2:
            return h
    return 0.01


def texto_tiempo(valor: float) -> str:
    """Formatea un tiempo; avisa si no se alcanzó dentro del horizonte."""
    if valor is None or not np.isfinite(valor):
        return "no alcanzado"
    return f"{valor:.1f} s"


def texto_numero(valor: float, decimales: int = 2) -> str:
    """Formatea un número y usa un guion si es NaN."""
    if valor is None or not np.isfinite(valor):
        return "—"
    return f"{valor:.{decimales}f}"


# ---------------------------------------------------------------------------
# Panel lateral
# ---------------------------------------------------------------------------

st.sidebar.header("Parámetros del sistema")
mu = st.sidebar.slider(
    "μ: capacidad por réplica [pet./(s·réplica)]", 1.0, 30.0, 10.0, 0.5
)
K = st.sidebar.slider(
    "K: ganancia del autoescalador [réplicas/petición]",
    0.001, 0.030, 0.005, 0.0005, format="%.4f",
)
tau = st.sidebar.slider(
    "τ: retardo de aprovisionamiento [s]", 1.0, 100.0, 20.0, 1.0
)
d0 = st.sidebar.slider(
    "d0: amplitud de la carga [peticiones/s]", 1.0, 50.0, 20.0, 1.0
)
T = st.sidebar.slider(
    "T: horizonte de simulación [s]", 100.0, 1000.0, 600.0, 50.0
)

p = BASE.con_parametros(mu=mu, K=K, tau=tau, d0=d0, T=T)

st.sidebar.header("Entrada de carga")
tipo = st.sidebar.radio("Forma de d(t)", ("Escalón", "Sinusoidal"))

w_res = frecuencia_resonancia(p)
w = 0.0
usar_resonancia = False
if tipo == "Sinusoidal":
    if w_res > 0.0:
        usar_resonancia = st.sidebar.checkbox(
            "Fijar ω en la frecuencia de resonancia", value=False
        )
    w_max = max(0.5, 2.0 * p.wn)
    w_defecto = float(np.clip(w_res if w_res > 0.0 else 0.05, 0.001, w_max))
    w = st.sidebar.slider(
        "ω: frecuencia de la carga [rad/s]",
        0.001, float(w_max), w_defecto, 0.001, format="%.3f",
    )
    if usar_resonancia:
        w = w_res

st.sidebar.caption(
    "Modelo lineal con parámetros positivos: siempre estable. "
    "La inestabilidad real aparece solo con retardo puro (extensión)."
)


# ---------------------------------------------------------------------------
# Encabezado y régimen
# ---------------------------------------------------------------------------

st.title("Autoescalado de servicios: simulación interactiva")
st.latex(r"x' = -\mu y + d(t), \qquad \tau y' = -y + Kx, \qquad x(0)=y(0)=0")

st.subheader("Régimen y valores estacionarios")
c1, c2, c3, c4 = st.columns(4)
c1.metric("ωn [rad/s]", f"{p.wn:.4f}")
c2.metric("ζ", f"{p.zeta:.4f}", p.regimen)
c3.metric("x_ss [peticiones]", f"{p.x_ss:.1f}")
c4.metric("y_ss [réplicas]", f"{p.y_ss:.2f}")

h = paso_adecuado(p)
t = np.linspace(0.0, p.T, 2001)

# Se define la entrada y se calculan las tres soluciones una sola vez.
if tipo == "Escalón":
    entrada = lambda tt: float(np.where(tt >= 0.0, p.d0, 0.0))
    x_a, y_a = x_escalon(t, p), y_escalon(t, p)
else:
    entrada = lambda tt: float(entrada_sinusoidal(tt, p.d0, w))
    x_a, y_a = x_sinusoidal(t, p, w), y_sinusoidal(t, p, w)

t_num, x_e, y_e = euler_explicito(p, h, entrada=entrada)
_, x_r, y_r = rk4(p, h, entrada=entrada)

# El error se mide en la malla numérica, no en la malla fina de la gráfica.
if tipo == "Escalón":
    x_a_num, y_a_num = x_escalon(t_num, p), y_escalon(t_num, p)
else:
    x_a_num, y_a_num = x_sinusoidal(t_num, p, w), y_sinusoidal(t_num, p, w)

error_e = error_max_estado(x_e, y_e, x_a_num, y_a_num)
error_r = error_max_estado(x_r, y_r, x_a_num, y_a_num)


# ---------------------------------------------------------------------------
# Pestaña 1: respuesta temporal
# ---------------------------------------------------------------------------

tab_temporal, tab_polos, tab_sensibilidad = st.tabs(
    ["Respuesta temporal", "Plano de polos", "Sensibilidad de ζ"]
)

with tab_temporal:
    if tipo == "Escalón":
        m = metricas_escalon(p, n_muestras=40_001)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Sobrepaso máximo [%]", f"{m['sobrepaso_pct']:.2f}")
        d2.metric("Tiempo de pico", texto_numero(m["t_pico_s"], 1) + " s"
                  if np.isfinite(m["t_pico_s"]) else "—")
        d3.metric("Asentamiento (2 %)", texto_tiempo(m["t_asentamiento_s"]))
        d4.metric("Periodo amortiguado", texto_numero(m["periodo_s"], 1) + " s"
                  if np.isfinite(m["periodo_s"]) else "—")
    else:
        rf = respuesta_frecuencia(p, w)
        amplitud_x = p.d0 * float(rf["mag_x"])
        st.caption(
            f"ω = {w:.3f} rad/s · |G(jω)| = {float(rf['mag_x']):.4f} s · "
            f"amplitud estacionaria de x = {amplitud_x:.2f} peticiones"
        )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Cola de peticiones x(t)", "Capacidad adicional y(t)"),
        vertical_spacing=0.13,
    )
    fig.add_trace(
        go.Scatter(x=t, y=x_a, name="Analítica",
                   line=dict(color="black", width=2.5)), 1, 1)
    fig.add_trace(
        go.Scatter(x=t_num, y=x_r, name=f"RK4 (h={h:g} s)",
                   line=dict(color="#1f77b4", dash="dot")), 1, 1)
    fig.add_trace(
        go.Scatter(x=t_num, y=x_e, name=f"Euler (h={h:g} s)",
                   line=dict(color="#d62728", dash="dash")), 1, 1)

    fig.add_trace(
        go.Scatter(x=t, y=y_a, name="Analítica", showlegend=False,
                   line=dict(color="black", width=2.5)), 2, 1)
    fig.add_trace(
        go.Scatter(x=t_num, y=y_r, name=f"RK4 (h={h:g} s)", showlegend=False,
                   line=dict(color="#1f77b4", dash="dot")), 2, 1)
    fig.add_trace(
        go.Scatter(x=t_num, y=y_e, name=f"Euler (h={h:g} s)", showlegend=False,
                   line=dict(color="#d62728", dash="dash")), 2, 1)

    if tipo == "Escalón":
        fig.add_hline(y=p.x_ss, line_dash="dash", line_color="gray",
                      line_width=1, row=1, col=1)
        fig.add_hline(y=p.y_ss, line_dash="dash", line_color="gray",
                      line_width=1, row=2, col=1)

    fig.update_yaxes(title_text="peticiones", row=1, col=1)
    fig.update_yaxes(title_text="réplicas", row=2, col=1)
    fig.update_xaxes(title_text="tiempo t [s]", row=2, col=1)
    fig.update_layout(
        height=680, hovermode="x unified",
        title="Solución analítica vs. métodos numéricos",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig)

    st.caption(
        f"Error máximo en el estado: Euler = {error_e:.3e}, "
        f"RK4 = {error_r:.3e}. Paso elegido h = {h:g} s "
        f"(Euler necesita h·ωn < 2ζ)."
    )


# ---------------------------------------------------------------------------
# Pestaña 2: plano de polos
# ---------------------------------------------------------------------------

with tab_polos:
    polos = p.valores_propios
    reales = [float(np.real(z)) for z in polos]
    imag = [float(np.imag(z)) for z in polos]

    fig_p = go.Figure()
    fig_p.add_trace(
        go.Scatter(
            x=reales, y=imag, mode="markers+text",
            text=[f"{r:.3f}{i:+.3f}j" for r, i in zip(reales, imag)],
            textposition="top center",
            marker=dict(size=14, color="#1f77b4",
                        line=dict(color="black", width=1)),
            name="polos",
        )
    )
    fig_p.add_vline(x=0.0, line_dash="dash", line_color="black")
    fig_p.add_hline(y=0.0, line_color="gray", line_width=1)
    fig_p.update_layout(
        height=520,
        title="Polos de τs² + s + μK = 0 en el plano complejo",
        xaxis_title="Parte real [1/s]",
        yaxis_title="Parte imaginaria [rad/s]",
        showlegend=False,
    )
    st.plotly_chart(fig_p)
    st.caption(
        "Con μ, K, τ > 0 todos los polos tienen parte real negativa: el "
        "sistema es asintóticamente estable. ζ controla cuánto oscila."
    )


# ---------------------------------------------------------------------------
# Pestaña 3: sensibilidad de ζ
# ---------------------------------------------------------------------------

with tab_sensibilidad:
    taus = np.linspace(1.0, 100.0, 300)
    zeta_tau = 1.0 / (2.0 * np.sqrt(p.mu * p.K * taus))
    ks = np.linspace(0.001, 0.030, 300)
    zeta_k = 1.0 / (2.0 * np.sqrt(p.mu * ks * p.tau))

    fig_z = make_subplots(
        rows=1, cols=2,
        subplot_titles=("ζ frente a τ (con K actual)",
                        "ζ frente a K (con τ actual)"),
    )
    fig_z.add_trace(
        go.Scatter(x=taus, y=zeta_tau, name="ζ(τ)",
                   line=dict(color="#1f77b4", width=2)), 1, 1)
    fig_z.add_trace(
        go.Scatter(x=[p.tau], y=[p.zeta], mode="markers", name="caso actual",
                   marker=dict(size=14, color="gold",
                               line=dict(color="black", width=1))), 1, 1)
    fig_z.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)

    fig_z.add_trace(
        go.Scatter(x=ks, y=zeta_k, name="ζ(K)", showlegend=False,
                   line=dict(color="#d62728", width=2)), 1, 2)
    fig_z.add_trace(
        go.Scatter(x=[p.K], y=[p.zeta], mode="markers", showlegend=False,
                   marker=dict(size=14, color="gold",
                               line=dict(color="black", width=1))), 1, 2)
    fig_z.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)

    fig_z.update_xaxes(title_text="τ [s]", row=1, col=1)
    fig_z.update_xaxes(title_text="K [réplicas/petición]", row=1, col=2)
    fig_z.update_yaxes(title_text="ζ (adimensional)", row=1, col=1)
    fig_z.update_layout(height=460, title="Cómo cambia el amortiguamiento")
    st.plotly_chart(fig_z)
    st.caption(
        "Más τ o más K aumentan μKτ y, por tanto, reducen ζ: el lazo se "
        "vuelve más oscilatorio aunque siga siendo estable."
    )
