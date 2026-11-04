"""
Extensión opcional: retardo puro en el controlador de autoescalado.

El núcleo del trabajo usa

    tau*y' = -y + K*x.

En un sistema distribuido real, la decisión de escalar se toma con la cola
medida hace theta segundos (latencia de métricas, cola de mensajes, etc.):

    tau*y' = -y + K*x(t - theta).

Linealizando alrededor del equilibrio se obtiene la ecuación característica

    tau*s^2 + s + mu*K*exp(-s*theta) = 0.

Con theta>0 pueden aparecer raíces con parte real positiva, es decir,
oscilaciones crecientes. Este módulo calcula el retardo crítico, compara la
predicción exacta con la aproximación de Padé de primer orden y simula un
caso estable y otro inestable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .modelo import BASE, Parametros


# ---------------------------------------------------------------------------
# 1. Retardo crítico exacto
# ---------------------------------------------------------------------------

def theta_critico_exacto(p: Parametros) -> float:
    """Retardo crítico exacto [s] del sistema con retardo puro.

    En la frontera de estabilidad s=j*omega. Sustituyendo en
    tau*s^2+s+mu*K*exp(-s*theta)=0 e igualando partes real e imaginaria:

        mu*K*cos(omega*theta) = tau*omega^2
        mu*K*sin(omega*theta) = omega

    De ahí

        omega^2 = [-1 + sqrt(1 + 4*tau^2*(mu*K)^2)] / (2*tau^2)
        theta_c = atan(1/(tau*omega)) / omega.

    Es el primer cruce de la frontera al aumentar theta desde cero.
    """
    c = p.mu * p.K
    tau = p.tau
    omega_cuadrado = (
        -1.0 + np.sqrt(1.0 + 4.0 * tau**2 * c**2)
    ) / (2.0 * tau**2)
    omega = float(np.sqrt(omega_cuadrado))
    return float(np.arctan(1.0 / (tau * omega)) / omega)


# ---------------------------------------------------------------------------
# 2. Aproximación de Padé de primer orden
# ---------------------------------------------------------------------------

def coeficientes_pade(p: Parametros, theta: float) -> np.ndarray:
    """Coeficientes [a3, a2, a1, a0] del polinomio característico aproximado.

    Con exp(-s*theta) ~= (1-theta*s/2)/(1+theta*s/2):

        (tau*theta/2)s^3 + (tau+theta/2)s^2
        + (1-mu*K*theta/2)s + mu*K = 0.
    """
    return np.array(
        [
            p.tau * theta / 2.0,
            p.tau + theta / 2.0,
            1.0 - p.mu * p.K * theta / 2.0,
            p.mu * p.K,
        ],
        dtype=float,
    )


def theta_critico_pade(p: Parametros) -> float:
    """Retardo crítico [s] según el criterio de Routh-Hurwitz con Padé 1.

    Para a3*s^3+a2*s^2+a1*s+a0=0 con a3>0, la estabilidad requiere
    a1>0, a2>0, a0>0 y a2*a1 > a3*a0. La igualdad conduce a

        mu*K*theta^2 + (4*tau*mu*K-2)*theta - 4*tau = 0,

    cuya raíz positiva es el retardo crítico aproximado.
    """
    a = p.mu * p.K
    b = 4.0 * p.tau * p.mu * p.K - 2.0
    c = -4.0 * p.tau
    discriminante = b**2 - 4.0 * a * c
    return float((-b + np.sqrt(discriminante)) / (2.0 * a))


def estable_pade(p: Parametros, theta: float) -> bool:
    """Evalúa la estabilidad del modelo aproximado con Padé 1."""
    a3, a2, a1, a0 = coeficientes_pade(p, theta)
    return bool(a3 >= 0.0 and a2 > 0.0 and a1 > 0.0 and a0 > 0.0
                and a2 * a1 > a3 * a0)


# ---------------------------------------------------------------------------
# 3. Simulación del sistema con retardo (método de pasos)
# ---------------------------------------------------------------------------

def simular_retardo(
    p: Parametros,
    theta: float,
    h: float = 0.01,
    T: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integra el sistema con retardo mediante RK4 y método de pasos.

    Para t<theta se usa la historia x(t-theta)=0. Para t>=theta se interpola
    linealmente el valor pasado de x, que ya está almacenado. Se exige
    h<theta para que la interpolación use solo valores ya calculados.
    """
    if theta <= h:
        raise ValueError("Para el método de pasos se requiere h < theta.")
    T_final = p.T if T is None else float(T)
    n = int(round(T_final / h))
    t = np.arange(n + 1, dtype=float) * h
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    mu, K, tau = p.mu, p.K, p.tau

    def x_retrasada(td: float, n_actual: int) -> float:
        """Historia x(td) con interpolación lineal (td <= t_n)."""
        if td <= 0.0:
            return 0.0
        posicion = td / h
        j = int(np.floor(posicion))
        j = max(0, min(j, n_actual - 1))
        fraccion = posicion - j
        return float(x[j] * (1.0 - fraccion) + x[j + 1] * fraccion)

    # Funciones auxiliares para no repetir la expresión en cada etapa.
    def f(tt: float, estado: np.ndarray, n_actual: int) -> np.ndarray:
        xx, yy = estado
        return np.array(
            [
                -mu * yy + p.d0,
                (-yy + K * x_retrasada(tt - theta, n_actual)) / tau,
            ],
            dtype=float,
        )

    for k in range(n):
        tn = t[k]
        estado = np.array([x[k], y[k]], dtype=float)
        k1 = f(tn, estado, k)
        k2 = f(tn + 0.5 * h, estado + 0.5 * h * k1, k)
        k3 = f(tn + 0.5 * h, estado + 0.5 * h * k2, k)
        k4 = f(tn + h, estado + h * k3, k)
        estado = estado + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x[k + 1], y[k + 1] = estado

    return t, x, y


# ---------------------------------------------------------------------------
# 4. Tabla y figuras
# ---------------------------------------------------------------------------

def tabla_retardo(p: Parametros = BASE) -> pd.DataFrame:
    """Tabla de theta crítico para los barridos de tau y K."""
    filas: List[Dict[str, object]] = []

    for tau in (2.0, 5.0, 20.0, 60.0, 100.0):
        caso = p.con_parametros(tau=tau)
        exacto = theta_critico_exacto(caso)
        pade = theta_critico_pade(caso)
        filas.append(
            {
                "barrido": "tau",
                "valor": tau,
                "theta_critico_exacto_s": exacto,
                "theta_critico_pade_s": pade,
                "error_relativo_pade_pct": abs(pade - exacto) / exacto * 100.0,
            }
        )

    for K in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03):
        caso = p.con_parametros(K=K)
        exacto = theta_critico_exacto(caso)
        pade = theta_critico_pade(caso)
        filas.append(
            {
                "barrido": "K",
                "valor": K,
                "theta_critico_exacto_s": exacto,
                "theta_critico_pade_s": pade,
                "error_relativo_pade_pct": abs(pade - exacto) / exacto * 100.0,
            }
        )
    return pd.DataFrame(filas)


def graficar_theta_critico(ruta: Path, p: Parametros = BASE) -> None:
    """Curvas de theta crítico frente a tau y frente a K."""
    fig, (ax_tau, ax_K) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    taus = np.linspace(2.0, 100.0, 300)
    exactos = [theta_critico_exacto(p.con_parametros(tau=float(t))) for t in taus]
    pades = [theta_critico_pade(p.con_parametros(tau=float(t))) for t in taus]
    ax_tau.plot(taus, exactos, color="#1f77b4", lw=2.0, label="Exacto")
    ax_tau.plot(taus, pades, color="#d62728", lw=1.5, ls="--",
                label="Padé de 1er orden")
    ax_tau.plot([p.tau], [theta_critico_exacto(p)], "*", color="gold", ms=16,
                mec="black", mew=1.0, label="Caso base")
    ax_tau.set_xlabel("Retardo de aprovisionamiento τ [s]")
    ax_tau.set_ylabel("Retardo crítico θc [s]")
    ax_tau.set_title("θc frente a τ (K=0.005)")
    ax_tau.grid(True, ls=":", alpha=0.65)
    ax_tau.legend(fontsize=9)

    Ks = np.linspace(0.001, 0.03, 300)
    exactos = [theta_critico_exacto(p.con_parametros(K=float(k))) for k in Ks]
    pades = [theta_critico_pade(p.con_parametros(K=float(k))) for k in Ks]
    ax_K.plot(Ks, exactos, color="#1f77b4", lw=2.0, label="Exacto")
    ax_K.plot(Ks, pades, color="#d62728", lw=1.5, ls="--",
              label="Padé de 1er orden")
    ax_K.plot([p.K], [theta_critico_exacto(p)], "*", color="gold", ms=16,
              mec="black", mew=1.0, label="Caso base")
    ax_K.set_xlabel("Ganancia K [réplicas/petición]")
    ax_K.set_ylabel("Retardo crítico θc [s]")
    ax_K.set_title("θc frente a K (τ=20 s)")
    ax_K.grid(True, ls=":", alpha=0.65)
    ax_K.legend(fontsize=9)

    fig.suptitle(
        "Margen de estabilidad con retardo puro: exacto vs. Padé 1",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


def graficar_simulacion_retardo(ruta: Path, p: Parametros = BASE) -> None:
    """Simula un retardo estable y uno inestable (θ < θc y θ > θc)."""
    theta_c = theta_critico_exacto(p)
    theta_estable = 0.5 * theta_c
    theta_inestable = 1.5 * theta_c

    t_e, x_e, y_e = simular_retardo(p, theta_estable, h=0.01)
    t_i, x_i, y_i = simular_retardo(p, theta_inestable, h=0.01)

    fig, (ax_e, ax_i) = plt.subplots(1, 2, figsize=(14.0, 5.8))

    # Caso estable: oscilaciones que se atenúan hacia el equilibrio.
    ax_e.plot(t_e, x_e, color="#1f77b4", lw=1.6, label="x(t)")
    ax_e.axhline(p.x_ss, color="#1f77b4", ls=":", lw=1.0,
                 label="x_ss teórico")
    ax_e.set_xlabel("Tiempo t [s]")
    ax_e.set_ylabel("Cola x(t) [peticiones]")
    ax_e.set_title(
        f"Estable: θ={theta_estable:.2f} s < θc={theta_c:.2f} s",
        fontsize=11,
    )
    ax_e.grid(True, ls=":", alpha=0.65)
    ax_e.legend(fontsize=9)
    ax_e2 = ax_e.twinx()
    ax_e2.plot(t_e, y_e, color="#ff7f0e", lw=1.2, alpha=0.9)
    ax_e2.set_ylabel("Capacidad y(t) [réplicas]", color="#b35300")
    ax_e2.tick_params(axis="y", labelcolor="#b35300")

    # Caso inestable: la amplitud crece con el tiempo.
    ax_i.plot(t_i, x_i, color="#d62728", lw=1.6, label="x(t)")
    ax_i.set_xlabel("Tiempo t [s]")
    ax_i.set_ylabel("Cola x(t) [peticiones]")
    ax_i.set_title(
        f"Inestable: θ={theta_inestable:.2f} s > θc={theta_c:.2f} s",
        fontsize=11,
    )
    ax_i.grid(True, ls=":", alpha=0.65)
    ax_i.legend(fontsize=9)
    ax_i2 = ax_i.twinx()
    ax_i2.plot(t_i, y_i, color="#ff7f0e", lw=1.2, alpha=0.9)
    ax_i2.set_ylabel("Capacidad y(t) [réplicas]", color="#b35300")
    ax_i2.tick_params(axis="y", labelcolor="#b35300")

    fig.suptitle(
        "Extensión: efecto del retardo puro en la estabilidad del autoescalado",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ejecutar_extension(
    directorio_figuras: Path,
    ruta_tabla: Path,
    p: Parametros = BASE,
) -> pd.DataFrame:
    """Genera tabla y figuras de la extensión de retardo."""
    df = tabla_retardo(p)
    ruta_tabla.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_tabla, index=False, float_format="%.10g")
    graficar_theta_critico(directorio_figuras / "10_retardo_critico.png", p)
    graficar_simulacion_retardo(
        directorio_figuras / "11_retardo_simulacion.png", p
    )
    return df
