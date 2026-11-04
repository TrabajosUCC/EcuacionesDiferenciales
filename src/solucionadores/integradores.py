"""Integradores genéricos para sistemas de EDOs de primer orden z' = f(t, z).

Euler y RK4 usan los mismos algoritmos ya validados en ``src/numerico.py``,
pero aceptan cualquier campo vectorial (cualquier número de variables).
"""
from __future__ import annotations

import math
import time
from typing import Callable, Tuple

import numpy as np
from scipy.integrate import solve_ivp

MAX_PASOS = 200_000
EULER = "Euler explícito"
RK4 = "Runge-Kutta 4 (RK4)"
SCIPY = "SciPy solve_ivp"
METODOS = (EULER, RK4, SCIPY)

Campo = Callable[[float, np.ndarray], np.ndarray]


def malla(T: float, h: float) -> np.ndarray:
    """Malla uniforme t = 0, h, ..., n*h con n = floor(T/h).

    El paso NO se modifica: si h no divide a T, el último instante es n*h < T.
    """
    if not (math.isfinite(T) and T > 0):
        raise ValueError("El tiempo de simulación debe ser un número positivo.")
    if not (math.isfinite(h) and h > 0):
        raise ValueError("El paso temporal debe ser un número positivo.")
    if h > T:
        raise ValueError("El paso temporal no puede ser mayor que el tiempo de simulación.")
    n = int(math.floor(T / h + 1e-9))
    if n > MAX_PASOS:
        raise ValueError(
            f"T/h = {n} pasos supera el máximo permitido ({MAX_PASOS}). Aumenta h o reduce T."
        )
    return np.arange(n + 1, dtype=float) * h


def euler(f: Campo, z0, t: np.ndarray) -> np.ndarray:
    Z = np.empty((len(t), len(z0)))
    z = np.array(z0, dtype=float)
    Z[0] = z
    for n in range(len(t) - 1):
        z = z + (t[n + 1] - t[n]) * f(t[n], z)
        Z[n + 1] = z
    return Z


def rk4(f: Campo, z0, t: np.ndarray) -> np.ndarray:
    Z = np.empty((len(t), len(z0)))
    z = np.array(z0, dtype=float)
    Z[0] = z
    for n in range(len(t) - 1):
        h = t[n + 1] - t[n]
        tn = t[n]
        k1 = f(tn, z)
        k2 = f(tn + 0.5 * h, z + 0.5 * h * k1)
        k3 = f(tn + 0.5 * h, z + 0.5 * h * k2)
        k4 = f(tn + h, z + h * k3)
        z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Z[n + 1] = z
    return Z


def integrar(f: Campo, z0, t: np.ndarray, metodo: str,
             rtol: float = 1e-8, atol: float = 1e-10) -> Tuple[np.ndarray, float]:
    """Integra con el método indicado. Devuelve (Z[nt, n_estados], segundos)."""
    t0 = time.perf_counter()
    with np.errstate(all="ignore"):
        if metodo == EULER:
            Z = euler(f, z0, t)
        elif metodo == RK4:
            Z = rk4(f, z0, t)
        elif metodo == SCIPY:
            sol = solve_ivp(f, (0.0, float(t[-1])), np.array(z0, float), method="RK45",
                            t_eval=t, rtol=rtol, atol=atol)
            if not sol.success:
                raise ArithmeticError(f"solve_ivp falló: {sol.message}")
            Z = sol.y.T
        else:
            raise ValueError(f"Método desconocido: {metodo}")
    dt = time.perf_counter() - t0
    if not np.all(np.isfinite(Z)):
        raise ArithmeticError(
            f"{metodo} produjo valores no finitos (inestabilidad numérica). Reduce el paso temporal."
        )
    return Z, dt


def referencia(f: Campo, z0, t: np.ndarray) -> np.ndarray:
    """Solución numérica de referencia (DOP853, tolerancias estrictas)."""
    sol = solve_ivp(f, (0.0, float(t[-1])), np.array(z0, float), method="DOP853",
                    t_eval=t, rtol=1e-11, atol=1e-13)
    if not sol.success:
        raise ArithmeticError(f"La referencia numérica falló: {sol.message}")
    return sol.y.T
