"""Soluciones cerradas del oscilador lineal  m x'' + c x' + k x = F0 cos(W t)."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

TOL_CRITICO = 1e-9


def omega0(m: float, k: float) -> float:
    return math.sqrt(k / m)


def zeta(m: float, c: float, k: float) -> float:
    return c / (2.0 * math.sqrt(k * m))


def regimen(m: float, c: float, k: float) -> str:
    z = zeta(m, c, k)
    if c == 0:
        return "no amortiguado"
    if abs(z - 1.0) <= TOL_CRITICO:
        return "críticamente amortiguado"
    return "subamortiguado" if z < 1 else "sobreamortiguado"


def constantes(m: float, c: float, k: float, x0: float, v0: float) -> Dict[str, float]:
    """Constantes de la solución homogénea según el régimen."""
    w0, z, a = omega0(m, k), zeta(m, c, k), c / (2.0 * m)
    d = {"w0": w0, "zeta": z, "alpha": a}
    if c == 0 or (z < 1.0 and abs(z - 1.0) > TOL_CRITICO):
        wd = w0 * math.sqrt(1.0 - z * z)
        d.update(tipo="sub", wd=wd, C1=x0, C2=(v0 + a * x0) / wd)
    elif abs(z - 1.0) <= TOL_CRITICO:
        d.update(tipo="crit", C1=x0, C2=v0 + a * x0)
    else:
        s = w0 * math.sqrt(z * z - 1.0)
        r1, r2 = -a + s, -a - s
        C2 = (v0 - r1 * x0) / (r2 - r1)
        d.update(tipo="sobre", r1=r1, r2=r2, C1=x0 - C2, C2=C2)
    return d


def homogenea(m, c, k, x0, v0, t):
    """Devuelve (x_h(t), v_h(t)) con las condiciones iniciales dadas."""
    t = np.asarray(t, float)
    d = constantes(m, c, k, x0, v0)
    if d["tipo"] == "sub":
        a, wd, C1, C2 = d["alpha"], d["wd"], d["C1"], d["C2"]
        e = np.exp(-a * t)
        cs, sn = np.cos(wd * t), np.sin(wd * t)
        x = e * (C1 * cs + C2 * sn)
        v = e * ((-a * C1 + wd * C2) * cs + (-a * C2 - wd * C1) * sn)
    elif d["tipo"] == "crit":
        a, C1, C2 = d["alpha"], d["C1"], d["C2"]
        e = np.exp(-a * t)
        x = e * (C1 + C2 * t)
        v = e * (C2 - a * (C1 + C2 * t))
    else:
        r1, r2, C1, C2 = d["r1"], d["r2"], d["C1"], d["C2"]
        x = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)
        v = C1 * r1 * np.exp(r1 * t) + C2 * r2 * np.exp(r2 * t)
    return x, v


def es_resonancia_pura(m, c, k, W) -> bool:
    return c == 0 and math.isclose(W, omega0(m, k), rel_tol=1e-9)


def amplitud(m, c, k, F0, W) -> float:
    den = (k - m * W * W) ** 2 + (c * W) ** 2
    return F0 / math.sqrt(den) if den > 0 else math.inf


def fase(m, c, k, W) -> float:
    """Desfase φ ∈ [0, π] de x_p = A cos(Wt - φ)."""
    return math.atan2(c * W, k - m * W * W)


def frecuencia_pico(m, c, k):
    """(Ω_r, A_max/F0) si existe pico de amplitud, si no (None, None)."""
    r2 = k / m - c * c / (2.0 * m * m)
    if r2 <= 0:
        return None, None
    wd = math.sqrt(k / m - c * c / (4.0 * m * m))
    return math.sqrt(r2), 1.0 / (c * wd) if c > 0 else math.inf


def forzada(m, c, k, F0, W, x0, v0, t) -> Dict[str, np.ndarray]:
    """Respuesta completa = transitoria (homogénea) + estacionaria (particular)."""
    t = np.asarray(t, float)
    if es_resonancia_pura(m, c, k, W):
        w0 = omega0(m, k)
        g = F0 / (2.0 * m * w0)
        xp = g * t * np.sin(w0 * t)
        vp = g * (np.sin(w0 * t) + w0 * t * np.cos(w0 * t))
        xh, vh = homogenea(m, c, k, x0, v0, t)
    else:
        A, ph = amplitud(m, c, k, F0, W), fase(m, c, k, W)
        xp = A * np.cos(W * t - ph)
        vp = -A * W * np.sin(W * t - ph)
        xh, vh = homogenea(m, c, k, x0 - A * math.cos(ph), v0 - A * W * math.sin(ph), t)
    return {"x_tr": xh, "v_tr": vh, "x_ss": xp, "v_ss": vp, "x": xh + xp, "v": vh + vp}
