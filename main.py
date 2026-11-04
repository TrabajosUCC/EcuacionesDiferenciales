"""
Ejecutor principal del proyecto de autoescalado.

Uso:
    python main.py

Genera, sin abrir ventanas (backend Agg), todas las figuras y tablas en
resultados/. El script es idempotente: se puede ejecutar tantas veces como
se quiera y sobrescribe los archivos con los mismos resultados.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Se selecciona un backend sin interfaz gráfica ANTES de importar pyplot.
# Esto permite ejecutar el proyecto en servidores, contenedores o CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from src.analitica import (
    comparar_con_solve_ivp,
    metricas_escalon,
    residuo_simbolico,
    respuesta_frecuencia,
    x_escalon,
    y_escalon,
)
from src.convergencia import ejecutar_convergencia
from src.extension_retardo import (
    ejecutar_extension,
    estable_pade,
    theta_critico_exacto,
    theta_critico_pade,
)
from src.modelo import BASE, Parametros, verificar_dimensiones
from src.numerico import euler_explicito, error_max_estado, rk4
from src.parametros import ejecutar_analisis_parametros, tabla_parametros

RAIZ = Path(__file__).resolve().parent
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"


# ===========================================================================
# Figuras generales del sistema
# ===========================================================================

def _bloque(ax, x, y, ancho, alto, texto, color="#dbe9f6"):
    """Dibuja un bloque rectangular con texto centrado."""
    caja = FancyBboxPatch(
        (x - ancho / 2, y - alto / 2), ancho, alto,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor="#1f3b57", facecolor=color, zorder=2,
    )
    ax.add_patch(caja)
    ax.text(x, y, texto, ha="center", va="center", fontsize=10, zorder=3)


def _flecha(ax, inicio, fin, color="#1f3b57", estilo="-|>", lw=1.5):
    """Dibuja una flecha entre dos puntos del diagrama."""
    ax.add_patch(
        FancyArrowPatch(
            inicio, fin, arrowstyle=estilo, mutation_scale=14,
            linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=4,
        )
    )


def figura_diagrama_bloques(ruta: Path) -> None:
    """Diagrama de bloques del modelo x'=-mu*y+d, tau*y'=-y+K*x."""
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    ax.set_title(
        "Diagrama de bloques del sistema de autoescalado",
        fontsize=15, pad=12,
    )

    # Nodo sumador de la entrada de la cola.
    sumador = Circle((2.3, 4.0), 0.28, fill=False, lw=1.5,
                     edgecolor="#1f3b57", zorder=5)
    ax.add_patch(sumador)
    ax.text(2.3, 4.0, "+", fontsize=12, ha="center", va="center", zorder=6)
    ax.text(2.1, 3.55, "−", fontsize=12, ha="center", va="center", zorder=6)

    # Entrada de perturbación d(t).
    ax.text(0.55, 4.28, "d(t)", fontsize=11, ha="left")
    ax.text(0.55, 3.72, "[peticiones/s]", fontsize=8, ha="left", color="#444")
    _flecha(ax, (0.95, 4.0), (2.02, 4.0))

    # Integrador, ganancia K y dinámica de aprovisionamiento.
    _bloque(ax, 3.55, 4.0, 1.35, 0.85, "1/s\nintegrador")
    _flecha(ax, (2.58, 4.0), (2.87, 4.0))
    _flecha(ax, (4.22, 4.0), (5.17, 4.0))
    ax.text(4.65, 4.72, "x(t) [peticiones]", fontsize=9, ha="center")

    _bloque(ax, 5.85, 4.0, 1.35, 0.85, "K\n[rep./pet.]")
    _flecha(ax, (6.52, 4.0), (7.12, 4.0))

    _bloque(ax, 7.95, 4.0, 1.65, 0.85, "1/(τs+1)\narranque")
    _flecha(ax, (8.78, 4.0), (9.55, 4.0))
    ax.text(9.15, 4.72, "y(t) [réplicas]", fontsize=9, ha="center")

    # Realimentación de capacidad: y -> mu -> entrada negativa.
    _bloque(ax, 8.2, 1.7, 1.65, 0.85, "μ\n[pet./(s·rep.)]",
            color="#f6e2d3")
    _flecha(ax, (8.9, 4.0), (8.9, 2.13), estilo="-")
    _flecha(ax, (7.37, 1.7), (2.3, 1.7), estilo="-")
    _flecha(ax, (2.3, 1.7), (2.3, 3.72))
    ax.text(5.9, 1.95, "realimentación de capacidad", fontsize=9,
            ha="center", color="#444")

    # Leyenda esquemática.
    leyenda = [
        Line2D([0], [0], color="#1f3b57", lw=1.5,
               label="señal / flujo de información"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#dbe9f6",
               markeredgecolor="#1f3b57", markersize=12,
               label="bloque dinámico o ganancia"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#1f3b57", markersize=10,
               label="nodo sumador"),
    ]
    ax.legend(handles=leyenda, loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.text(
        0.02, 0.03,
        "Ecuaciones: x' = −μy + d(t),   τy' = −y + Kx",
        transform=ax.transAxes, fontsize=9, color="#333",
    )

    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figura_comparacion(ruta: Path, base: Parametros = BASE) -> None:
    """Compara analítica, Euler y RK4 en la misma figura para el escalón."""
    h = 2.0
    t, x_e, y_e = euler_explicito(base, h)
    _, x_r, y_r = rk4(base, h)
    x_a = x_escalon(t, base)
    y_a = y_escalon(t, base)

    error_e = error_max_estado(x_e, y_e, x_a, y_a)
    error_r = error_max_estado(x_r, y_r, x_a, y_a)

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)

    ax_x.plot(t, x_a, color="black", lw=2.5, label="Analítica")
    ax_x.plot(t, x_e, color="#d62728", lw=1.4, ls="--",
              label=f"Euler (h={h:g} s)")
    ax_x.plot(t, x_r, color="#1f77b4", lw=1.6, ls=":",
              label=f"RK4 (h={h:g} s)")
    ax_x.set_ylabel("Cola x(t) [peticiones]")
    ax_x.set_title("Caso base: analítica vs. Euler vs. RK4 (escalón d0=20)")
    ax_x.grid(True, ls=":", alpha=0.65)
    ax_x.legend(fontsize=9)
    ax_x.text(
        0.99, 0.04,
        f"Error máx. Euler = {error_e:.3e}\nError máx. RK4 = {error_r:.3e}",
        transform=ax_x.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )

    ax_y.plot(t, y_a, color="black", lw=2.5, label="Analítica")
    ax_y.plot(t, y_e, color="#d62728", lw=1.4, ls="--",
              label=f"Euler (h={h:g} s)")
    ax_y.plot(t, y_r, color="#1f77b4", lw=1.6, ls=":",
              label=f"RK4 (h={h:g} s)")
    ax_y.set_xlabel("Tiempo t [s]")
    ax_y.set_ylabel("Capacidad y(t) [réplicas]")
    ax_y.grid(True, ls=":", alpha=0.65)
    ax_y.legend(fontsize=9)

    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figura_retrato_fase(ruta: Path, base: Parametros = BASE) -> None:
    """Retrato de fase x-y con la trayectoria hacia el equilibrio."""
    t = np.linspace(0.0, base.T, 6001)
    x = x_escalon(t, base)
    y = y_escalon(t, base)

    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    ax.plot(x, y, color="#1f77b4", lw=2.0, label="trayectoria (t creciente)")

    # Flechas para indicar el sentido del recorrido.
    for fraccion in (0.03, 0.12, 0.30, 0.60):
        i = int(fraccion * (len(t) - 1))
        j = min(i + 10, len(t) - 1)
        ax.annotate(
            "", xy=(x[j], y[j]), xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle="-|>", color="#1f77b4",
                            lw=1.4, mutation_scale=14),
        )

    # Punto inicial y punto de equilibrio.
    ax.plot([base.x0], [base.y0], "o", color="#d62728", ms=8,
            label="inicio (0, 0)")
    ax.plot([base.x_ss], [base.y_ss], "*", color="gold", ms=18,
            mec="black", mew=1.0, label="equilibrio (x_ss, y_ss)")

    # Isoclinas: x'=0 -> y=d0/mu; y'=0 -> y=K*x.
    ax.axhline(base.d0 / base.mu, color="gray", ls="--", lw=1.0,
               label="isoclina x'=0: y=d0/μ")
    xx = np.linspace(0.0, 1.05 * base.x_ss, 200)
    ax.plot(xx, base.K * xx, color="#2ca02c", ls=":", lw=1.2,
            label="isoclina y'=0: y=Kx")

    ax.set_xlabel("Cola x [peticiones]")
    ax.set_ylabel("Capacidad y [réplicas]")
    ax.set_title("Retrato de fase del caso base (escalón)")
    ax.grid(True, ls=":", alpha=0.65)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Flujo principal por fases
# ===========================================================================

def main() -> None:
    """Ejecuta todas las fases y verifica que los entregables existan."""
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FASE 1: modelo y solución analítica")
    print("=" * 70)
    print(f"Parámetros base: {BASE}")
    print(f"wn = {BASE.wn:.6f} rad/s, zeta = {BASE.zeta:.6f}, "
          f"régimen = {BASE.regimen}")
    print(f"x_ss = {BASE.x_ss:.3f} peticiones, "
          f"y_ss = {BASE.y_ss:.3f} réplicas")
    print("Chequeo dimensional:", verificar_dimensiones(BASE))
    print("Métricas del caso base:", metricas_escalon(BASE, 120_001))
    print("Residuo simbólico del caso base:", residuo_simbolico(BASE))
    t_val = np.linspace(0.0, BASE.T, 3001)
    print("Analítica vs. solve_ivp:",
          comparar_con_solve_ivp(BASE, t_val))
    w_res = respuesta_frecuencia(BASE, 0.0)  # evaluación en w=0
    print(f"|G(0)| = {float(w_res['mag_x']):.6f}")

    print("\nGenerando figuras generales...")
    figura_diagrama_bloques(DIR_FIGURAS / "01_diagrama_bloques.png")
    figura_comparacion(DIR_FIGURAS / "02_comparacion_analitica_numerica.png")
    figura_retrato_fase(DIR_FIGURAS / "04_retrato_fase.png")

    print("\n" + "=" * 70)
    print("FASE 2: simulación numérica y convergencia")
    print("=" * 70)
    df_conv = ejecutar_convergencia(
        BASE,
        DIR_RESULTADOS / "tabla_convergencia.csv",
        DIR_FIGURAS / "03_convergencia.png",
    )
    print(df_conv.to_string(index=False))

    print("\n" + "=" * 70)
    print("FASE 3: análisis de parámetros y figuras")
    print("=" * 70)
    df_par = ejecutar_analisis_parametros(DIR_FIGURAS, DIR_RESULTADOS /
                                          "tabla_parametros.csv")
    print(df_par.to_string(index=False))

    print("\n" + "=" * 70)
    print("FASE 4: extensión opcional con retardo puro")
    print("=" * 70)
    theta_c = theta_critico_exacto(BASE)
    theta_p = theta_critico_pade(BASE)
    print(f"Retardo crítico exacto del caso base: {theta_c:.6f} s")
    print(f"Retardo crítico con Padé 1:          {theta_p:.6f} s")
    print(f"Padé estable en theta=10 s: {estable_pade(BASE, 10.0)}")
    print(f"Padé estable en theta=35 s: {estable_pade(BASE, 35.0)}")
    df_ret = ejecutar_extension(DIR_FIGURAS, DIR_RESULTADOS /
                                "tabla_retardo.csv", BASE)
    print(df_ret.to_string(index=False))

    # Comprobación explícita de entregables. Si algo falta, el programa falla
    # con un mensaje claro en lugar de terminar silenciosamente.
    figuras_esperadas = [
        "01_diagrama_bloques.png",
        "02_comparacion_analitica_numerica.png",
        "03_convergencia.png",
        "04_retrato_fase.png",
        "05a_barrido_tau.png",
        "05b_barrido_K.png",
        "06_rejilla_3x3.png",
        "07_mapa_zeta.png",
        "08_respuesta_frecuencia.png",
        "09_polos.png",
        "10_retardo_critico.png",
        "11_retardo_simulacion.png",
    ]
    faltantes = [n for n in figuras_esperadas
                 if not (DIR_FIGURAS / n).is_file()]
    if faltantes:
        raise RuntimeError(f"Faltan figuras: {faltantes}")
    if not (DIR_RESULTADOS / "tabla_convergencia.csv").is_file():
        raise RuntimeError("Falta tabla_convergencia.csv")
    if not (DIR_RESULTADOS / "tabla_parametros.csv").is_file():
        raise RuntimeError("Falta tabla_parametros.csv")
    if not (DIR_RESULTADOS / "tabla_retardo.csv").is_file():
        raise RuntimeError("Falta tabla_retardo.csv")

    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)
    print(f"Figuras en: {DIR_FIGURAS}")
    print(f"Tablas en:  {DIR_RESULTADOS}")
    print(f"Total de figuras: {len(figuras_esperadas)}")


if __name__ == "__main__":
    main()
