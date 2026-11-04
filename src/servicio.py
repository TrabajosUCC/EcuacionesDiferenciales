"""Orquestación: valida, integra con los 3 métodos, compara y arma los resultados."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from .solucionadores import integradores as ig
from .visualizaciones.figuras import figura, serie


class ErrorValidacion(Exception):
    def __init__(self, errores: List[str]):
        super().__init__("; ".join(errores))
        self.errores = errores


def config_defecto(sis) -> dict:
    return {"T": sis.tiempo_defecto, "h": sis.paso_defecto, "metodo": ig.RK4,
            "rtol": 1e-8, "atol": 1e-10}


def validar_config(cfg: dict) -> List[str]:
    errs = []
    for k, n in (("T", "Tiempo de simulación"), ("h", "Paso temporal"),
                 ("rtol", "Tolerancia relativa"), ("atol", "Tolerancia absoluta")):
        v = cfg.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
            errs.append(f"{n}: debe ser un número positivo.")
    if cfg.get("metodo") not in ig.METODOS:
        errs.append("Método numérico no válido.")
    if not errs:
        try:
            ig.malla(cfg["T"], cfg["h"])
        except ValueError as e:
            errs.append(str(e))
    return errs


def ejecutar(sis, p: dict, cfg: dict) -> dict:
    errs = sis.validar(p) + validar_config(cfg)
    if errs:
        raise ErrorValidacion(errs)
    t = ig.malla(cfg["T"], cfg["h"])
    f, z0 = sis.rhs(p), sis.z0(p)
    avisos = []
    if abs(t[-1] - cfg["T"]) > 1e-9 * cfg["T"]:
        avisos.append(f"h no divide a T: la simulación termina en t = {t[-1]:.6g} s (no en {cfg['T']:.6g} s).")

    Za = sis.analitica(p, t)
    ref = Za if Za is not None else ig.referencia(f, z0, t)
    ref_nombre = "solución analítica" if Za is not None else "solve_ivp DOP853 (rtol=1e-11)"

    Zs, tiempos = {}, {}
    for m in ig.METODOS:
        try:
            Zs[m], tiempos[m] = ig.integrar(f, z0, t, m, cfg["rtol"], cfg["atol"])
        except ArithmeticError as e:
            if m == cfg["metodo"]:
                raise ErrorValidacion([str(e)])
            avisos.append(str(e))
    Z = Zs[cfg["metodo"]]

    i = sis.magnitud_principal()
    escala = float(np.max(np.abs(ref[:, i]))) or 1.0
    comp = []
    for m, Zm in Zs.items():
        err = np.abs(Zm[:, i] - ref[:, i])
        comp.append({"Método": m, "Error absoluto máx": float(err.max()),
                     "Error relativo máx": float(err.max() / escala),
                     "Tiempo de ejecución (s)": tiempos[m]})
    figs = sis.figuras(p, t, Z, Za)
    fig_err = figura(f"Error absoluto de «{sis.estados[i]}» respecto a la {ref_nombre}", "t (s)", "|error|",
                     [serie(m, t, np.maximum(np.abs(Zm[:, i] - ref[:, i]), 1e-18)) for m, Zm in Zs.items()],
                     ylog=True)
    return {"clave": sis.clave, "p": dict(p), "cfg": dict(cfg), "t": t, "Z": Z, "Zs": Zs, "Za": Za,
            "ref": ref, "ref_nombre": ref_nombre, "tiene_analitica": Za is not None,
            "comparacion": pd.DataFrame(comp), "derivadas": sis.derivadas(p),
            "desarrollo": sis.desarrollo(p, cfg), "figuras": figs,
            "figuras_comparacion": [fig_err] + sis.figuras_comparacion(p, t, Zs, ref),
            "interpretacion": sis.interpretacion(p, t, Z), "avisos": avisos,
            "estados": sis.estados}


def tabla_resultados(res: dict, max_filas: int = 2000) -> pd.DataFrame:
    n = len(res["t"])
    idx = np.unique(np.linspace(0, n - 1, min(n, max_filas)).astype(int))
    d = {"t (s)": res["t"][idx]}
    for j, nombre in enumerate(res["estados"]):
        d[f"{nombre} numérica"] = res["Z"][idx, j]
        if res["tiene_analitica"]:
            d[f"{nombre} analítica"] = res["Za"][idx, j]
            d[f"error abs {nombre}"] = np.abs(res["Z"][idx, j] - res["Za"][idx, j])
    return pd.DataFrame(d)
