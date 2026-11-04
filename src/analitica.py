"""
Solución analítica del modelo de autoescalado.

Se trabaja con la EDO de segundo orden obtenida al eliminar y(t):

    x'' + (1/tau)*x' + (mu*K/tau)*x = d'(t) + d(t)/tau

y con la función de transferencia

    G(s) = X(s)/D(s) = (tau*s + 1) / (tau*s^2 + s + mu*K).

El módulo contiene:
  * la derivación simbólica con SymPy (Laplace inversa),
  * fórmulas cerradas numéricamente estables para los tres regímenes,
  * la solución completa para carga sinusoidal,
  * la respuesta en frecuencia y la resonancia,
  * las métricas de respuesta al escalón.

Convención: los parámetros son siempre positivos, por lo que el sistema es
asintóticamente estable. Lo que cambia con tau y K es el amortiguamiento.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import brentq, minimize_scalar

from .modelo import Parametros, entrada_escalon, entrada_sinusoidal

# Tolerancia para decidir si el régimen es exactamente crítico. Es mucho menor
# que cualquier diferencia físicamente relevante entre parámetros.
TOL_CRITICO = 1.0e-9


# ===========================================================================
# 1. Derivación simbólica con SymPy
# ===========================================================================

@lru_cache(maxsize=1)
def simbolos_modelo() -> Dict[str, sp.Symbol]:
    """Crea los símbolos positivos usados en la derivación simbólica."""
    s = sp.symbols("s", positive=True)
    t = sp.symbols("t", positive=True)
    tau, mu, K, d0 = sp.symbols("tau mu K d0", positive=True)
    return {"s": s, "t": t, "tau": tau, "mu": mu, "K": K, "d0": d0}


@lru_cache(maxsize=1)
def funcion_transferencia_simbolica() -> sp.Expr:
    """Devuelve G(s) = (tau*s+1)/(tau*s^2+s+mu*K) con SymPy."""
    sy = simbolos_modelo()
    s, tau, mu, K = sy["s"], sy["tau"], sy["mu"], sy["K"]
    return (tau * s + 1) / (tau * s**2 + s + mu * K)


@lru_cache(maxsize=1)
def laplace_escalon_x_simbolica() -> sp.Expr:
    """Transformada de Laplace de x(t) ante un escalón d0."""
    sy = simbolos_modelo()
    s, d0 = sy["s"], sy["d0"]
    return d0 * funcion_transferencia_simbolica() / s


@lru_cache(maxsize=1)
def laplace_escalon_y_simbolica() -> sp.Expr:
    """Transformada de Laplace de y(t) ante un escalón d0.

    De tau*y' = -y + K*x se obtiene Y(s) = K*X(s)/(tau*s+1), es decir,
    Y(s) = d0*K / (s*(tau*s^2+s+mu*K)).
    """
    sy = simbolos_modelo()
    s, tau, mu, K, d0 = sy["s"], sy["tau"], sy["mu"], sy["K"], sy["d0"]
    return d0 * K / (s * (tau * s**2 + s + mu * K))


def laplace_sinusoidal_x_simbolica(w: float) -> sp.Expr:
    """Transformada de x(t) para d(t)=d0*sin(w*t), condiciones iniciales nulas.

    L{sin(w*t)} = w/(s^2+w^2). Por tanto X(s)=G(s)*d0*w/(s^2+w^2).
    """
    sy = simbolos_modelo()
    s, d0 = sy["s"], sy["d0"]
    w_sym = sp.Float(w)
    return d0 * funcion_transferencia_simbolica() * w_sym / (s**2 + w_sym**2)


def formulas_escalon_simbolicas(p: Parametros) -> Dict[str, sp.Expr]:
    """Deriva x(t) e y(t) con Laplace inversa para los parámetros dados.

    Se sustituyen los parámetros numéricos como números racionales exactos
    para que SymPy pueda simplificar correctamente los tres regímenes. El
    resultado es una expresión simbólica en t, útil para verificar el residuo
    de la EDO.

    Esta función es de verificación: las gráficas y simulaciones usan las
    fórmulas cerradas vectorizadas de la sección 2 por rapidez y estabilidad.
    """
    sy = simbolos_modelo()
    s, t, tau, mu, K, d0 = (
        sy["s"], sy["t"], sy["tau"], sy["mu"], sy["K"], sy["d0"],
    )
    # Rational(str(...)) convierte 0.005 en 1/200 sin arrastrar el error
    # binario del punto flotante.
    valores = {
        tau: sp.Rational(str(p.tau)),
        mu: sp.Rational(str(p.mu)),
        K: sp.Rational(str(p.K)),
        d0: sp.Rational(str(p.d0)),
    }
    X = laplace_escalon_x_simbolica().subs(valores)
    Y = laplace_escalon_y_simbolica().subs(valores)

    x_t = sp.simplify(sp.inverse_laplace_transform(X, s, t))
    y_t = sp.simplify(sp.inverse_laplace_transform(Y, s, t))

    # Para t>0, Heaviside(t)=1. Se elimina explícitamente para poder derivar
    # sin generar deltas de Dirac en el origen.
    x_t = x_t.replace(sp.Heaviside(t), sp.Integer(1))
    y_t = y_t.replace(sp.Heaviside(t), sp.Integer(1))
    return {"x": sp.simplify(x_t), "y": sp.simplify(y_t), "t": t}


def residuo_simbolico(p: Parametros, n_muestras: int = 200) -> Dict[str, float]:
    """Sustituye la solución simbólica en la EDO y mide el residuo máximo.

    Se comprueban las dos ecuaciones del sistema de primer orden:
        r_x = x' + mu*y - d(t)
        r_y = tau*y' + y - K*x
    y, de forma equivalente, la EDO de segundo orden. Se evalúa en una malla
    de t>0 para evitar el punto t=0, donde la derivada de Heaviside es una
    distribución.
    """
    formulas = formulas_escalon_simbolicas(p)
    t = formulas["t"]
    x_t, y_t = formulas["x"], formulas["y"]

    # RHS del escalón en la EDO de segundo orden y residuos del sistema 2x2.
    rhs_segundo_orden = sp.Rational(str(p.d0)) / sp.Rational(str(p.tau))
    r_x = sp.diff(x_t, t, 2) + sp.diff(x_t, t) / sp.Rational(str(p.tau)) \
        + sp.Rational(str(p.mu * p.K / p.tau)) * x_t - rhs_segundo_orden
    r_y = sp.Rational(str(p.tau)) * sp.diff(y_t, t) + y_t \
        - sp.Rational(str(p.K)) * x_t

    # Lambdify permite evaluar rápido sin volver a simplificar.
    f_rx = sp.lambdify(t, sp.simplify(r_x), modules="numpy")
    f_ry = sp.lambdify(t, sp.simplify(r_y), modules="numpy")

    t_eval = np.linspace(1.0e-6, p.T, n_muestras)
    rx = np.asarray(f_rx(t_eval), dtype=float) * np.ones_like(t_eval)
    ry = np.asarray(f_ry(t_eval), dtype=float) * np.ones_like(t_eval)

    return {
        "residuo_max_x": float(np.max(np.abs(rx))),
        "residuo_max_y": float(np.max(np.abs(ry))),
        "regimen": p.regimen,
    }


# ===========================================================================
# 2. Fórmulas cerradas para el escalón
# ===========================================================================

def _como_arreglo(t: np.ndarray | float) -> Tuple[np.ndarray, bool]:
    """Convierte t a arreglo y recuerda si la entrada era escalar."""
    t_arr = np.asarray(t, dtype=float)
    return t_arr, bool(t_arr.ndim == 0)


def _devolver(valor: np.ndarray, escalar: bool) -> np.ndarray | float:
    """Devuelve un float si la entrada original era escalar."""
    return float(valor) if escalar else valor


def _omega_d(p: Parametros) -> float:
    """Frecuencia amortiguada [rad/s]: wd = wn*sqrt(1-zeta^2)."""
    return float(p.wn * np.sqrt(max(0.0, 1.0 - p.zeta**2)))


def _alfa(p: Parametros) -> float:
    """Atenuación [1/s]: alfa = zeta*wn = 1/(2*tau)."""
    return float(p.zeta * p.wn)


def _homogenea(
    t: np.ndarray | float, p: Parametros, x0: float, v0: float
) -> np.ndarray | float:
    """Solución homogénea de la EDO de 2do orden.

    Resuelve x'' + (1/tau)x' + (mu*K/tau)x = 0 con x(0)=x0, x'(0)=v0.
    Se elige automáticamente el régimen según zeta.
    """
    t_arr, escalar = _como_arreglo(t)
    alfa = _alfa(p)
    zeta = p.zeta

    # Caso 1: subamortiguado (zeta < 1).
    if zeta < 1.0 - TOL_CRITICO:
        wd = _omega_d(p)
        A = x0
        B = (v0 + alfa * A) / wd
        sol = np.exp(-alfa * t_arr) * (A * np.cos(wd * t_arr) + B * np.sin(wd * t_arr))
        return _devolver(sol, escalar)

    # Caso 2: crítico (zeta = 1). Se usa la forma t*exp(-alfa*t).
    if abs(zeta - 1.0) <= TOL_CRITICO:
        A = x0
        B = v0 + alfa * A
        sol = np.exp(-alfa * t_arr) * (A + B * t_arr)
        return _devolver(sol, escalar)

    # Caso 3: sobreamortiguado (zeta > 1). Dos exponenciales reales.
    # Las raíces se toman reales para que todas las operaciones se mantengan
    # en punto flotante real (en este régimen el discriminante es positivo).
    r1_c, r2_c = p.raices_caracteristicas()
    r1, r2 = float(np.real(r1_c)), float(np.real(r2_c))
    C1 = (v0 - r2 * x0) / (r1 - r2)
    C2 = (r1 * x0 - v0) / (r1 - r2)
    sol = C1 * np.exp(r1 * t_arr) + C2 * np.exp(r2 * t_arr)
    return _devolver(sol, escalar)


def x_escalon(t: np.ndarray | float, p: Parametros) -> np.ndarray | float:
    """Cola x(t) ante d(t)=d0, con x(0)=y(0)=0.

    La solución se escribe como x(t) = x_ss + x_h(t), donde x_h es la
    homogénea con x_h(0) = -x_ss y x_h'(0) = d0 (porque x'(0)=-mu*y(0)+d0).
    """
    t_arr, escalar = _como_arreglo(t)
    sol = p.x_ss + _homogenea(t_arr, p, -p.x_ss, p.d0)
    return _devolver(sol, escalar)


def y_escalon(t: np.ndarray | float, p: Parametros) -> np.ndarray | float:
    """Capacidad y(t) ante d(t)=d0, con y(0)=0 y y'(0)=0.

    y'(0) = (K*x(0)-y(0))/tau = 0. Por eso la homogénea de y usa
    y_h(0) = -y_ss y y_h'(0) = 0.
    """
    t_arr, escalar = _como_arreglo(t)
    sol = p.y_ss + _homogenea(t_arr, p, -p.y_ss, 0.0)
    return _devolver(sol, escalar)


# ===========================================================================
# 3. Solución completa para la carga sinusoidal
# ===========================================================================

def respuesta_frecuencia(p: Parametros, w: np.ndarray | float) -> Dict[str, np.ndarray]:
    """Evalúa G(jw) y las amplitudes/fases de x e y en régimen permanente.

    Para d(t)=d0*sin(w*t):
        x_p(t) = d0*|G(jw)|*sin(w*t + fase_x)
        y_p(t) = d0*|G_y(jw)|*sin(w*t + fase_y)
    con G_y(s) = K/(tau*s^2+s+mu*K).
    """
    w_arr, escalar = _como_arreglo(w)
    A = p.mu * p.K - p.tau * w_arr**2      # parte real del denominador
    B = w_arr                               # parte imaginaria
    den = A**2 + B**2

    # |G(jw)| = sqrt(1+(tau*w)^2) / sqrt(A^2+B^2)
    mag_x = np.sqrt(1.0 + (p.tau * w_arr) ** 2) / np.sqrt(den)
    fase_x = np.arctan2(p.tau * w_arr, 1.0) - np.arctan2(B, A)

    mag_y = p.K / np.sqrt(den)
    fase_y = -np.arctan2(B, A)

    g_compleja = (1.0 + 1j * p.tau * w_arr) / (A + 1j * B)
    resultado = {
        "G": g_compleja,
        "mag_x": mag_x,
        "fase_x": fase_x,
        "mag_y": mag_y,
        "fase_y": fase_y,
    }
    if escalar:
        # G es compleja; no debe convertirse a float. Las magnitudes y fases
        # sí son reales.
        return {
            "G": complex(np.asarray(g_compleja).item()),
            "mag_x": float(np.asarray(mag_x).item()),
            "fase_x": float(np.asarray(fase_x).item()),
            "mag_y": float(np.asarray(mag_y).item()),
            "fase_y": float(np.asarray(fase_y).item()),
        }
    return resultado


def frecuencia_resonancia(p: Parametros) -> float:
    """Frecuencia [rad/s] del máximo de |G(jw)|, si existe.

    Derivando |G(jw)|^2 respecto de u=w^2 se obtiene

        u_res = [sqrt(c*(c+2)) - 1] / tau^2,   c = mu*K*tau = 1/(4*zeta^2).

    Hay pico para w>0 cuando c*(c+2)>1, es decir,
    zeta < sqrt((sqrt(2)+1)/4) ~= 0.7769. Nótese que, por el cero del
    numerador, el umbral exacto no es 1/sqrt(2); 1/sqrt(2) es la condición
    clásica para un sistema de segundo orden sin cero.
    Devuelve 0.0 si |G| decrece monótonamente desde w=0.
    """
    c = p.mu * p.K * p.tau
    discriminante = np.sqrt(c * (c + 2.0)) - 1.0
    if discriminante <= 0.0:
        return 0.0
    return float(np.sqrt(discriminante) / p.tau)


def amplitud_resonancia(p: Parametros) -> float:
    """Valor máximo de |G(jw)| (o |G(0)| si no hay pico)."""
    wr = frecuencia_resonancia(p)
    return float(respuesta_frecuencia(p, wr)["mag_x"])


def x_sinusoidal(
    t: np.ndarray | float, p: Parametros, w: float
) -> np.ndarray | float:
    """Solución completa de x(t) para d=d0*sin(w*t), x(0)=y(0)=0.

    Se suma la respuesta permanente (particular) y la homogénea. Los
    coeficientes de la homogénea se eligen para cancelar el valor y la
    pendiente de la particular en t=0.
    """
    t_arr, escalar = _como_arreglo(t)
    rf = respuesta_frecuencia(p, w)
    mag_x, fase_x = float(rf["mag_x"]), float(rf["fase_x"])

    # Particular y sus condiciones iniciales.
    x_p0 = p.d0 * mag_x * np.sin(fase_x)
    v_p0 = p.d0 * mag_x * w * np.cos(fase_x)

    x_p = p.d0 * mag_x * np.sin(w * t_arr + fase_x)
    x_h = _homogenea(t_arr, p, -x_p0, -v_p0)
    return _devolver(x_h + x_p, escalar)


def y_sinusoidal(
    t: np.ndarray | float, p: Parametros, w: float
) -> np.ndarray | float:
    """Solución completa de y(t) para d=d0*sin(w*t), y(0)=0."""
    t_arr, escalar = _como_arreglo(t)
    rf = respuesta_frecuencia(p, w)
    mag_y, fase_y = float(rf["mag_y"]), float(rf["fase_y"])

    y_p0 = p.d0 * mag_y * np.sin(fase_y)
    v_p0 = p.d0 * mag_y * w * np.cos(fase_y)

    y_p = p.d0 * mag_y * np.sin(w * t_arr + fase_y)
    y_h = _homogenea(t_arr, p, -y_p0, -v_p0)
    return _devolver(y_h + y_p, escalar)


# ===========================================================================
# 4. Comparación con solve_ivp (referencia independiente)
# ===========================================================================

def comparar_con_solve_ivp(
    p: Parametros, t: np.ndarray, w: float | None = None
) -> Dict[str, float]:
    """Compara la fórmula analítica con solve_ivp de tolerancia estricta.

    Si w es None se usa el escalón; en caso contrario, la sinusoidal.
    Se usa DOP853 con rtol=1e-10 y atol=1e-12. solve_ivp se emplea solo como
    referencia de validación, nunca como método principal del proyecto.
    """
    from .numerico import resolver_solve_ivp

    if w is None:
        entrada = lambda tt: float(entrada_escalon(tt, p.d0))
        x_ref, y_ref = x_escalon(t, p), y_escalon(t, p)
    else:
        entrada = lambda tt: float(entrada_sinusoidal(tt, p.d0, w))
        x_ref, y_ref = x_sinusoidal(t, p, w), y_sinusoidal(t, p, w)

    # Se pide la solución exactamente en la malla t. Si se dejara que
    # solve_ivp eligiera sus propios nodos, la interpolación lineal entre
    # nodos grandes introduciría un error ajeno al método.
    sol = resolver_solve_ivp(p, entrada, t_eval=t)
    x_num = sol.y[0]
    y_num = sol.y[1]

    return {
        "error_max_x": float(np.max(np.abs(x_num - x_ref))),
        "error_max_y": float(np.max(np.abs(y_num - y_ref))),
    }


# ===========================================================================
# 5. Métricas de respuesta al escalón
# ===========================================================================

def metricas_escalon(p: Parametros, n_muestras: int = 200_001) -> Dict[str, float]:
    """Calcula sobrepaso, tiempo de pico, asentamiento (2 %) y periodo.

    Las métricas se obtienen de la solución analítica evaluada en una malla
    densa (paso <= 0.005 s para T=600 s) y se refinan con optimización
    escalar. No se usan las fórmulas clásicas de segundo orden sin cero
    porque G(s) tiene un cero en s=-1/tau y el sobrepaso real es distinto.
    Para regímenes sin oscilación, el periodo y el tiempo de pico se
    devuelven como NaN.
    """
    t = np.linspace(0.0, p.T, n_muestras)
    x = np.asarray(x_escalon(t, p), dtype=float)
    x_ss = p.x_ss

    # ---- Sobrepaso máximo y tiempo de pico --------------------------------
    idx_max = int(np.argmax(x))
    if idx_max == 0 or x[idx_max] <= x_ss * (1.0 + 1.0e-12):
        sobrepaso = 0.0
        t_pico = float("nan")
    else:
        # Refinamiento del máximo en el intervalo vecino.
        a = max(0.0, t[max(0, idx_max - 1)])
        b = min(p.T, t[min(len(t) - 1, idx_max + 1)])
        optimo = minimize_scalar(
            lambda tt: -float(x_escalon(tt, p)),
            bounds=(a, b),
            method="bounded",
            options={"xatol": 1.0e-9},
        )
        x_pico = -float(optimo.fun)
        t_pico = float(optimo.x)
        sobrepaso = max(0.0, (x_pico - x_ss) / x_ss * 100.0)

    # ---- Tiempo de asentamiento con banda del 2 % -------------------------
    banda = 0.02 * abs(x_ss)
    fuera = np.abs(x - x_ss) > banda
    if not np.any(fuera):
        t_asentamiento = 0.0
    else:
        ultimo = int(np.where(fuera)[0][-1])
        if ultimo >= len(t) - 1:
            # No alcanzó la banda dentro del horizonte simulado.
            t_asentamiento = float("inf")
        else:
            # Se refina la última salida de la banda con brentq.
            f = lambda tt: abs(float(x_escalon(tt, p)) - x_ss) - banda
            t_asentamiento = float(brentq(f, t[ultimo], t[ultimo + 1]))

    # ---- Periodo de oscilación amortiguada -------------------------------
    if p.zeta < 1.0 - TOL_CRITICO:
        periodo = float(2.0 * np.pi / _omega_d(p))
    else:
        periodo = float("nan")

    return {
        "x_ss": float(x_ss),
        "y_ss": float(p.y_ss),
        "sobrepaso_pct": float(sobrepaso),
        "t_pico_s": float(t_pico),
        "t_asentamiento_s": float(t_asentamiento),
        "periodo_s": float(periodo),
    }


if __name__ == "__main__":
    from .modelo import BASE

    print("G(s) simbólica =", funcion_transferencia_simbolica())
    print("x_ss =", BASE.x_ss, " y_ss =", BASE.y_ss)
    print("métricas base =", metricas_escalon(BASE))
    print("residuo simbólico base =", residuo_simbolico(BASE))
    t = np.linspace(0.0, BASE.T, 2000)
    print("comparación solve_ivp =", comparar_con_solve_ivp(BASE, t))
