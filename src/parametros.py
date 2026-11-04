"""
Barridos de parámetros, tablas y figuras del análisis de sensibilidad.

Se estudian dos parámetros con tres valores cada uno, manteniendo mu=10 fijo:

    tau en {5, 20, 60} s   con K = 0.005
    K   en {0.0025, 0.005, 0.02} con tau = 20 s

Además se construye la rejilla 3x3 completa, el mapa de zeta en el plano
(K, tau), la respuesta en frecuencia y la ubicación de polos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .analitica import (
    amplitud_resonancia,
    frecuencia_resonancia,
    metricas_escalon,
    respuesta_frecuencia,
    x_escalon,
    x_sinusoidal,
    y_escalon,
)
from .modelo import BASE, Parametros

TAU_VALORES: Sequence[float] = (5.0, 20.0, 60.0)
K_VALORES: Sequence[float] = (0.0025, 0.005, 0.02)


# ---------------------------------------------------------------------------
# 1. Construcción de casos y tabla de parámetros
# ---------------------------------------------------------------------------

def casos_tau(base: Parametros = BASE) -> List[Parametros]:
    """Casos del barrido de tau con K fijo."""
    return [base.con_parametros(tau=tau) for tau in TAU_VALORES]


def casos_K(base: Parametros = BASE) -> List[Parametros]:
    """Casos del barrido de K con tau fijo."""
    return [base.con_parametros(K=K) for K in K_VALORES]


def casos_rejilla(base: Parametros = BASE) -> List[Parametros]:
    """Los 9 pares (tau, K) de la rejilla 3x3, en orden tau-mayor."""
    return [
        base.con_parametros(tau=tau, K=K)
        for tau in TAU_VALORES
        for K in K_VALORES
    ]


def _formato_polo(z: complex) -> str:
    """Formatea un polo de forma legible y estable para el CSV."""
    if abs(z.imag) < 1.0e-8:
        return f"{z.real:.8f}"
    return f"{z.real:.8f}{z.imag:+.8f}j"


def tabla_parametros(base: Parametros = BASE) -> pd.DataFrame:
    """Calcula la tabla de los 9 casos con sus métricas de respuesta."""
    filas: List[Dict[str, object]] = []
    for i, p in enumerate(casos_rejilla(base), start=1):
        polos = p.raices_caracteristicas()
        metricas = metricas_escalon(p, n_muestras=120_001)
        filas.append(
            {
                "caso": f"C{i:02d}",
                "tau_s": p.tau,
                "K_replicas_por_peticion": p.K,
                "mu_peticiones_por_s_replica": p.mu,
                "wn_rad_s": p.wn,
                "zeta": p.zeta,
                "regimen": p.regimen,
                "polo_1": _formato_polo(polos[0]),
                "polo_2": _formato_polo(polos[1]),
                "x_ss_peticiones": metricas["x_ss"],
                "y_ss_replicas": metricas["y_ss"],
                "sobrepaso_pct": metricas["sobrepaso_pct"],
                "t_pico_s": metricas["t_pico_s"],
                "t_asentamiento_s": metricas["t_asentamiento_s"],
                "periodo_s": metricas["periodo_s"],
            }
        )
    return pd.DataFrame(filas)


def guardar_tabla_parametros(df: pd.DataFrame, ruta: Path) -> None:
    """Guarda la tabla de parámetros en CSV."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, float_format="%.10g")


# ---------------------------------------------------------------------------
# 2. Barridos de tau y K
# ---------------------------------------------------------------------------

def _dibujar_respuestas(
    casos: Iterable[Parametros],
    etiqueta_parametro: str,
    ruta: Path,
    titulo: str,
) -> None:
    """Dibuja x(t) e y(t) para una lista de casos.

    Se usan dos paneles porque x se mide en peticiones e y en réplicas; así
    cada eje conserva sus unidades y no se mezclan escalas.
    """
    casos = list(casos)
    colores = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(9.2, 8.2), sharex=True)

    for color, p in zip(colores, casos):
        t = np.linspace(0.0, p.T, 6001)
        x = x_escalon(t, p)
        y = y_escalon(t, p)
        texto = (
            f"{etiqueta_parametro}={p.tau:g} s"
            if etiqueta_parametro == "τ"
            else f"{etiqueta_parametro}={p.K:g}"
        )
        texto += f" (ζ={p.zeta:.3f})"
        ax_x.plot(t, x, color=color, lw=1.8, label=texto)
        ax_y.plot(t, y, color=color, lw=1.8, label=texto)
        ax_x.axhline(p.x_ss, color=color, ls=":", lw=0.9, alpha=0.65)
        ax_y.axhline(p.y_ss, color=color, ls=":", lw=0.9, alpha=0.65)

    ax_x.set_title(titulo)
    ax_x.set_ylabel("Cola x(t) [peticiones]")
    ax_y.set_ylabel("Capacidad y(t) [réplicas]")
    ax_y.set_xlabel("Tiempo t [s]")
    for ax in (ax_x, ax_y):
        ax.grid(True, ls=":", alpha=0.65)
        ax.legend(fontsize=9)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


def graficar_barrido_tau(ruta: Path, base: Parametros = BASE) -> None:
    """Figura del barrido de tau (3 valores, K fijo)."""
    _dibujar_respuestas(
        casos_tau(base),
        "τ",
        ruta,
        "Barrido de τ: efecto del retardo de aprovisionamiento (K=0.005)",
    )


def graficar_barrido_K(ruta: Path, base: Parametros = BASE) -> None:
    """Figura del barrido de K (3 valores, tau fijo)."""
    _dibujar_respuestas(
        casos_K(base),
        "K",
        ruta,
        "Barrido de K: efecto de la ganancia del autoescalador (τ=20 s)",
    )


# ---------------------------------------------------------------------------
# 3. Rejilla 3x3
# ---------------------------------------------------------------------------

def graficar_rejilla(ruta: Path, base: Parametros = BASE) -> None:
    """Rejilla 3x3 con los nueve pares (tau, K).

    Cada panel muestra x(t) en el eje izquierdo e y(t) en el derecho. Las
    líneas punteadas horizontales son los valores estacionarios teóricos.
    """
    casos = casos_rejilla(base)
    fig, ejes = plt.subplots(3, 3, figsize=(15.0, 11.0), sharex=True)
    t = np.linspace(0.0, base.T, 4001)
    azul, naranja = "#1f77b4", "#ff7f0e"

    for idx, (ax, p) in enumerate(zip(ejes.ravel(), casos)):
        x = x_escalon(t, p)
        y = y_escalon(t, p)
        ax.plot(t, x, color=azul, lw=1.6)
        ax.axhline(p.x_ss, color=azul, ls=":", lw=1.0)
        ax.set_title(
            f"τ={p.tau:g} s, K={p.K:g}\nζ={p.zeta:.3f} ({p.regimen})",
            fontsize=10,
        )
        ax.set_ylabel("x [peticiones]", fontsize=8)
        ax.grid(True, ls=":", alpha=0.55)

        ax2 = ax.twinx()
        ax2.plot(t, y, color=naranja, lw=1.4)
        ax2.axhline(p.y_ss, color=naranja, ls=":", lw=1.0)
        ax2.set_ylabel("y [réplicas]", fontsize=8)

        # Se etiquetan solo los ejes externos para no saturar la figura.
        if idx % 3 != 0:
            ax.set_ylabel("")
        if idx % 3 != 2:
            ax2.set_ylabel("")
        if idx >= 6:
            ax.set_xlabel("t [s]")

    leyenda = [
        Line2D([0], [0], color=azul, lw=1.6, label="x(t): cola"),
        Line2D([0], [0], color=naranja, lw=1.4, label="y(t): réplicas"),
        Line2D([0], [0], color="gray", lw=1.0, ls=":",
               label="valor estacionario"),
    ]
    fig.legend(handles=leyenda, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(
        "Rejilla 3×3 de respuestas al escalón para los pares (τ, K)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Mapa de zeta en el plano (K, tau)
# ---------------------------------------------------------------------------

def graficar_mapa_zeta(ruta: Path, base: Parametros = BASE) -> None:
    """Mapa de contorno de zeta=1/(2*sqrt(mu*K*tau)) con las tres zonas.

    La frontera crítica es zeta=1. Para este modelo es la hipérbola
    K*tau = 1/(4*mu). Por encima de ella (mayor K*tau) el sistema es
    subamortiguado; por debajo, sobreamortiguado.
    """
    K = np.linspace(0.001, 0.03, 500)
    tau = np.linspace(2.0, 100.0, 500)
    KK, TT = np.meshgrid(K, tau)
    Z = 1.0 / (2.0 * np.sqrt(base.mu * KK * TT))

    fig, ax = plt.subplots(figsize=(10.0, 7.2))
    niveles = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    relleno = ax.contourf(KK, TT, Z, levels=niveles, cmap="coolwarm",
                          extend="max", alpha=0.85)
    cbar = fig.colorbar(relleno, ax=ax, pad=0.02)
    cbar.set_label("Factor de amortiguamiento ζ (adimensional)")

    # Frontera crítica zeta=1 y referencia zeta=1/sqrt(2).
    ax.contour(KK, TT, Z, levels=[1.0], colors="black", linewidths=2.2)
    ax.contour(KK, TT, Z, levels=[1.0 / np.sqrt(2.0)], colors="black",
               linewidths=1.0, linestyles="--")

    # Puntos de los nueve casos y del caso base.
    for p in casos_rejilla(base):
        ax.plot(p.K, p.tau, "X", color="black", ms=7, mew=1.2)
    ax.plot(base.K, base.tau, "*", color="gold", ms=16, mec="black",
            mew=1.0, zorder=5, label="Caso base (τ=20, K=0.005)")

    ax.text(0.013, 78, "SUBAMORTIGUADO\n(ζ < 1, oscila)", fontsize=11,
            ha="center", color="#4a0c0c", fontweight="bold")
    ax.text(0.0030, 2.7, "ζ > 1\n(no oscila)",
            fontsize=9, ha="center", color="#0c234a", fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", alpha=0.75, ec="none"))
    # La frontera es la hipérbola K*tau=1/(4*mu). Se señala con una flecha
    # porque en la esquina inferior izquierda las curvas quedan muy juntas.
    ax.annotate(
        "FRONTERA\nζ = 1",
        xy=(0.010, 2.5), xytext=(0.016, 8.0),
        fontsize=9, ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    )

    leyenda = [
        Line2D([0], [0], marker="X", color="w", markerfacecolor="black",
               markeredgecolor="black", markersize=8,
               label="Casos de la rejilla 3×3"),
        Line2D([0], [0], color="black", lw=2.2, label="ζ = 1 (crítico)"),
        Line2D([0], [0], color="black", lw=1.0, ls="--",
               label="ζ = 1/√2 (referencia)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=13, label="Caso base"),
    ]
    ax.legend(handles=leyenda, loc="upper right", fontsize=8, framealpha=0.95)
    ax.set_xlabel("Ganancia del autoescalador K [réplicas/petición]")
    ax.set_ylabel("Retardo de aprovisionamiento τ [s]")
    ax.set_title("Mapa del amortiguamiento ζ en el plano (K, τ), con μ=10 fijo")
    ax.grid(True, ls=":", alpha=0.35)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Respuesta en frecuencia y simulación sinusoidal
# ---------------------------------------------------------------------------

def graficar_respuesta_frecuencia(ruta: Path, base: Parametros = BASE) -> None:
    """|G(jw)| para los tres zeta del barrido de K y simulación sinusoidal.

    El panel izquierdo muestra la ganancia y marca el pico de resonancia de
    cada caso (solo existe si ζ < 0.7769 para esta G, que tiene un cero).
    El panel derecho compara la respuesta temporal en la frecuencia de
    resonancia del caso base con una frecuencia alejada.
    """
    casos = casos_K(base)
    colores = ["#1f77b4", "#d62728", "#2ca02c"]
    w = np.linspace(1.0e-4, 0.25, 4000)

    fig, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(15.0, 6.2))

    for color, p in zip(colores, casos):
        mag = respuesta_frecuencia(p, w)["mag_x"]
        wr = frecuencia_resonancia(p)
        mr = amplitud_resonancia(p)
        ax_f.plot(w, mag, color=color, lw=1.9,
                  label=f"K={p.K:g}, ζ={p.zeta:.3f}")
        if wr > 0.0:
            ax_f.plot([wr], [mr], "o", color=color, ms=7, mec="black", mew=0.6)
            ax_f.axvline(wr, color=color, ls=":", lw=1.0, alpha=0.7)

    ax_f.set_title("Respuesta en frecuencia |G(jω)|")
    ax_f.set_xlabel("Frecuencia angular ω [rad/s]")
    ax_f.set_ylabel("|G(jω)| [s]")
    ax_f.grid(True, ls=":", alpha=0.65)
    ax_f.legend(fontsize=9)
    ax_f.text(
        0.98, 0.72,
        "Los círculos marcan\nel pico de resonancia",
        transform=ax_f.transAxes, ha="right", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )

    # Simulación del caso base: resonancia frente a una frecuencia lejana.
    wr = frecuencia_resonancia(base)
    w_lejos = 0.20
    rf_r = respuesta_frecuencia(base, wr)
    rf_l = respuesta_frecuencia(base, w_lejos)
    t = np.linspace(0.0, base.T, 12001)
    ax_t.plot(t, x_sinusoidal(t, base, wr), color="#d62728", lw=1.4,
              label=f"ω=ωr={wr:.4f} rad/s, |G|={float(rf_r['mag_x']):.2f}")
    ax_t.plot(t, x_sinusoidal(t, base, w_lejos), color="#1f77b4", lw=1.4,
              label=f"ω={w_lejos:.2f} rad/s, |G|={float(rf_l['mag_x']):.2f}")
    ax_t.set_title("Carga sinusoidal: resonancia vs. frecuencia alejada")
    ax_t.set_xlabel("Tiempo t [s]")
    ax_t.set_ylabel("Cola x(t) [peticiones]")
    ax_t.grid(True, ls=":", alpha=0.65)
    ax_t.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Análisis frecuencial del caso base y de los tres valores de ζ",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Ubicación de polos
# ---------------------------------------------------------------------------

def graficar_polos(ruta: Path, base: Parametros = BASE) -> None:
    """Ubica los polos de los nueve casos en el plano complejo.

    Todos los polos tienen parte real negativa, lo que confirma la
    estabilidad asintótica del modelo lineal con parámetros positivos.
    """
    casos = casos_rejilla(base)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    marcadores = {5.0: "o", 20.0: "s", 60.0: "^"}
    for p in casos:
        polos = p.raices_caracteristicas()
        reales = [z.real for z in polos]
        imaginarias = [z.imag for z in polos]
        ax.scatter(
            reales, imaginarias,
            marker=marcadores.get(p.tau, "o"),
            s=62, edgecolor="black", linewidth=0.6,
            label=f"τ={p.tau:g} s, K={p.K:g}, ζ={p.zeta:.3f}",
        )

    ax.axvline(0.0, color="black", ls="--", lw=1.2)
    ax.axhline(0.0, color="gray", lw=0.8, alpha=0.7)
    ax.set_xlabel("Parte real [1/s]")
    ax.set_ylabel("Parte imaginaria [rad/s]")
    ax.set_title("Ubicación de polos de los nueve casos del barrido")
    ax.grid(True, ls=":", alpha=0.65)
    ax.legend(fontsize=7.5, loc="lower left", ncol=3, framealpha=0.95)
    ax.text(0.02, 0.98, "semiplano izquierdo = estable",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Función de conveniencia para ejecutar todo el análisis de parámetros
# ---------------------------------------------------------------------------

def ejecutar_analisis_parametros(
    directorio_figuras: Path,
    ruta_tabla: Path,
    base: Parametros = BASE,
) -> pd.DataFrame:
    """Genera tabla y figuras de los barridos, el mapa y los polos."""
    df = tabla_parametros(base)
    guardar_tabla_parametros(df, ruta_tabla)
    graficar_barrido_tau(directorio_figuras / "05a_barrido_tau.png", base)
    graficar_barrido_K(directorio_figuras / "05b_barrido_K.png", base)
    graficar_rejilla(directorio_figuras / "06_rejilla_3x3.png", base)
    graficar_mapa_zeta(directorio_figuras / "07_mapa_zeta.png", base)
    graficar_respuesta_frecuencia(
        directorio_figuras / "08_respuesta_frecuencia.png", base
    )
    graficar_polos(directorio_figuras / "09_polos.png", base)
    return df
