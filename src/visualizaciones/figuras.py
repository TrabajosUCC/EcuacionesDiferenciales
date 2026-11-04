"""Especificaciones de figuras (datos puros). El render con Plotly está aparte."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

MAX_PUNTOS = 4000


def serie(nombre, x, y, dash: Optional[str] = None, modo: str = "lines") -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) > MAX_PUNTOS:
        idx = np.unique(np.linspace(0, len(x) - 1, MAX_PUNTOS).astype(int))
        x, y = x[idx], y[idx]
    return {"nombre": nombre, "x": x, "y": y, "dash": dash, "modo": modo}


def figura(titulo, xl, yl, series: List[dict], ylog: bool = False, igual: bool = False) -> dict:
    return {"titulo": titulo, "xl": xl, "yl": yl, "series": series, "ylog": ylog, "igual": igual}


def figuras_oscilador(t, x, v, a, energias: Dict[str, np.ndarray], nombres: dict,
                      xa=None, va=None) -> List[dict]:
    """Posición, velocidad, aceleración, retrato de fase y energía.

    ``nombres``: dict con claves x, v, a -> (titulo, etiqueta_eje).
    ``xa``, ``va``: solución analítica (se dibuja punteada si existe).
    """
    figs = []
    for clave, y, ya in (("x", x, xa), ("v", v, va), ("a", a, None)):
        tit, eje = nombres[clave]
        s = [serie("Numérica", t, y)]
        if ya is not None:
            s.append(serie("Analítica", t, ya, dash="dash"))
        figs.append(figura(f"{tit} frente al tiempo", "t (s)", eje, s))
    s = [serie("Numérica", x, v)]
    if xa is not None:
        s.append(serie("Analítica", xa, va, dash="dash"))
    figs.append(figura("Retrato de fase", nombres["x"][1], nombres["v"][1], s))
    figs.append(figura("Energía", "t (s)", nombres.get("E", "Energía (J)"),
                       [serie(n, t, e) for n, e in energias.items()]))
    return figs
