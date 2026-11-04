"""
Métodos numéricos implementados a mano para el sistema de autoescalado.

El sistema es

    x' = -mu*y + d(t)
    y' = (-y + K*x)/tau

y se integra con Euler explícito y con Runge-Kutta clásico de cuarto orden
(RK4). La entrada d(t) se evalúa en el tiempo exacto de cada etapa, lo cual
es imprescindible para que el orden de convergencia sea el teórico.

solve_ivp se usa únicamente como referencia de validación; no es el método
principal de simulación.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .modelo import Parametros, campo_vectorial, entrada_escalon


Entrada = Callable[[float], float]


def _malla_temporal(p: Parametros, h: float, T: Optional[float] = None) -> np.ndarray:
    """Construye la malla uniforme t=0,h,2h,...,N*h.

    Se exige que h sea positivo y que N*h coincida con T salvo error de
    redondeo. Los valores de h usados en el proyecto dividen exactamente a T.
    """
    if h <= 0:
        raise ValueError("El paso h debe ser positivo.")
    T_final = p.T if T is None else float(T)
    n = int(round(T_final / h))
    if n < 1:
        raise ValueError("El paso h es demasiado grande para el horizonte T.")
    t = np.arange(n + 1, dtype=float) * h
    # Si por redondeo el último punto no coincide con T, se ajusta el paso
    # para conservar exactamente el horizonte solicitado.
    if not np.isclose(t[-1], T_final, rtol=0.0, atol=1e-9 * max(1.0, T_final)):
        t = np.linspace(0.0, T_final, n + 1)
    return t


def _entrada_por_defecto(p: Parametros, entrada: Optional[Entrada]) -> Entrada:
    """Devuelve la entrada del usuario o el escalón d0 por defecto."""
    if entrada is not None:
        return entrada
    return lambda tt: float(entrada_escalon(tt, p.d0))


def euler_explicito(
    p: Parametros,
    h: float,
    T: Optional[float] = None,
    entrada: Optional[Entrada] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integra el sistema con Euler explícito.

    Fórmula: z_{n+1} = z_n + h*f(t_n, z_n).

    Devuelve (t, x, y). La entrada se evalúa en t_n, el único instante que
    utiliza Euler.
    """
    t = _malla_temporal(p, h, T)
    f = _entrada_por_defecto(p, entrada)
    x = np.empty_like(t)
    y = np.empty_like(t)
    x[0], y[0] = p.x0, p.y0

    estado = np.array([p.x0, p.y0], dtype=float)
    for n in range(len(t) - 1):
        derivada = campo_vectorial(t[n], estado, p, f)
        estado = estado + h * derivada
        x[n + 1], y[n + 1] = estado
    return t, x, y


def rk4(
    p: Parametros,
    h: float,
    T: Optional[float] = None,
    entrada: Optional[Entrada] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integra el sistema con Runge-Kutta clásico de cuarto orden.

    En cada paso se evalúa la entrada en t_n, t_n+h/2 (dos veces) y t_n+h.
    Esa evaluación correcta en las etapas es lo que produce orden global 4.
    """
    t = _malla_temporal(p, h, T)
    f = _entrada_por_defecto(p, entrada)
    x = np.empty_like(t)
    y = np.empty_like(t)
    x[0], y[0] = p.x0, p.y0

    estado = np.array([p.x0, p.y0], dtype=float)
    for n in range(len(t) - 1):
        tn = t[n]
        k1 = campo_vectorial(tn, estado, p, f)
        k2 = campo_vectorial(tn + 0.5 * h, estado + 0.5 * h * k1, p, f)
        k3 = campo_vectorial(tn + 0.5 * h, estado + 0.5 * h * k2, p, f)
        k4 = campo_vectorial(tn + h, estado + h * k3, p, f)
        estado = estado + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x[n + 1], y[n + 1] = estado
    return t, x, y


def resolver_solve_ivp(
    p: Parametros,
    entrada: Optional[Entrada] = None,
    t_eval: Optional[np.ndarray] = None,
) -> Any:
    """Referencia numérica independiente con tolerancias estrictas.

    Se usa DOP853 (Runge-Kutta de orden 8) con rtol=1e-10 y atol=1e-12.
    El resultado se emplea solo para validar la solución analítica.
    """
    f = _entrada_por_defecto(p, entrada)
    return solve_ivp(
        fun=lambda tt, z: campo_vectorial(tt, z, p, f),
        t_span=(0.0, p.T),
        y0=np.array([p.x0, p.y0], dtype=float),
        method="DOP853",
        t_eval=t_eval,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def error_max_estado(
    x_num: np.ndarray,
    y_num: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
) -> float:
    """Error en norma infinita para el estado completo [x, y].

    Se toma el máximo entre el error de la cola y el de las réplicas, de modo
    que un solo número resuma la discrepancia de ambas variables.
    """
    return float(max(np.max(np.abs(np.asarray(x_num) - x_ref)),
                     np.max(np.abs(np.asarray(y_num) - y_ref))))


def resumen_metodos(p: Parametros, h: float) -> Dict[str, float]:
    """Compara Euler y RK4 con solve_ivp para un paso dado (diagnóstico)."""
    from .analitica import x_escalon, y_escalon

    t = _malla_temporal(p, h)
    ref = resolver_solve_ivp(p, t_eval=t)
    _, xe, ye = euler_explicito(p, h)
    _, xr, yr = rk4(p, h)
    return {
        "error_euler_vs_solveivp": error_max_estado(xe, ye, ref.y[0], ref.y[1]),
        "error_rk4_vs_solveivp": error_max_estado(xr, yr, ref.y[0], ref.y[1]),
        "error_analitica_vs_solveivp": error_max_estado(
            x_escalon(t, p), y_escalon(t, p), ref.y[0], ref.y[1]
        ),
    }


if __name__ == "__main__":
    from .modelo import BASE

    for paso in (1.0, 0.5, 0.1):
        print(f"h={paso}: {resumen_metodos(BASE, paso)}")
