"""
Análisis de convergencia de Euler explícito y RK4.

Para cada tamaño de paso h se compara la solución numérica con la solución
analítica exacta y se calcula el error en norma infinita. Después se estima
el orden observado mediante

    p = log(e_{i-1}/e_i) / log(h_{i-1}/h_i).

Con un integrador correcto se espera p ~ 1 para Euler y p ~ 4 para RK4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analitica import x_escalon, y_escalon
from .modelo import Parametros
from .numerico import euler_explicito, error_max_estado, rk4

# Pasos exigidos por el enunciado. Todos dividen exactamente T=600 s.
PASOS_CONVERGENCIA = (2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01)


def tabla_convergencia(
    p: Parametros,
    pasos: Iterable[float] = PASOS_CONVERGENCIA,
) -> pd.DataFrame:
    """Construye la tabla de error y orden observado.

    Columnas:
        h, error_euler, error_rk4, orden_euler, orden_rk4

    El error se mide sobre el estado completo [x, y] con la norma infinita.
    """
    pasos = tuple(float(h) for h in pasos)
    errores_euler = []
    errores_rk4 = []

    for h in pasos:
        # Se genera la malla con cada método; ambos usan la misma malla.
        t, x_e, y_e = euler_explicito(p, h)
        _, x_r, y_r = rk4(p, h)

        # Solución analítica exacta en los mismos instantes.
        x_a = x_escalon(t, p)
        y_a = y_escalon(t, p)

        errores_euler.append(error_max_estado(x_e, y_e, x_a, y_a))
        errores_rk4.append(error_max_estado(x_r, y_r, x_a, y_a))

    orden_euler = [np.nan]
    orden_rk4 = [np.nan]
    for i in range(1, len(pasos)):
        razon_h = np.log(pasos[i - 1] / pasos[i])
        orden_euler.append(
            float(np.log(errores_euler[i - 1] / errores_euler[i]) / razon_h)
        )
        orden_rk4.append(
            float(np.log(errores_rk4[i - 1] / errores_rk4[i]) / razon_h)
        )

    return pd.DataFrame(
        {
            "h_s": pasos,
            "error_euler": errores_euler,
            "error_rk4": errores_rk4,
            "orden_euler": orden_euler,
            "orden_rk4": orden_rk4,
        }
    )


def guardar_tabla_convergencia(df: pd.DataFrame, ruta: Path) -> None:
    """Guarda la tabla en CSV con suficiente precisión para reproducir."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, float_format="%.10e")


def graficar_convergencia(df: pd.DataFrame, ruta: Path) -> None:
    """Gráfica log-log del error frente a h con pendientes de referencia.

    Se dibujan rectas auxiliares de pendiente 1 y 4 ancladas al primer punto
    de Euler y de RK4, respectivamente. Si la implementación es correcta,
    las curvas deben ser paralelas a esas rectas.
    """
    h = df["h_s"].to_numpy(dtype=float)
    e_euler = df["error_euler"].to_numpy(dtype=float)
    e_rk4 = df["error_rk4"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.loglog(h, e_euler, "o-", color="#d62728", lw=1.8, ms=6,
              label="Euler explícito")
    ax.loglog(h, e_rk4, "s-", color="#1f77b4", lw=1.8, ms=6,
              label="RK4")

    # Rectas de referencia: C*h^1 y C*h^4, ancladas al primer dato.
    h_ref = np.array([h[0], h[-1]])
    ax.loglog(h_ref, e_euler[0] * (h_ref / h[0]) ** 1,
              "--", color="#d62728", lw=1.0, alpha=0.65,
              label="Referencia pendiente 1")
    ax.loglog(h_ref, e_rk4[0] * (h_ref / h[0]) ** 4,
              "--", color="#1f77b4", lw=1.0, alpha=0.65,
              label="Referencia pendiente 4")

    ax.set_title("Convergencia de los métodos numéricos (caso base)")
    ax.set_xlabel("Tamaño de paso h [s]")
    ax.set_ylabel("Error máximo en el estado (norma infinito) [unidades mixtas]")
    ax.grid(True, which="both", ls=":", alpha=0.65)
    ax.legend(fontsize=9)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ejecutar_convergencia(
    p: Parametros,
    ruta_csv: Path,
    ruta_figura: Path,
    pasos: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """Calcula, guarda y grafica el estudio de convergencia."""
    if pasos is None:
        pasos = PASOS_CONVERGENCIA
    df = tabla_convergencia(p, pasos)
    guardar_tabla_convergencia(df, ruta_csv)
    graficar_convergencia(df, ruta_figura)
    return df
