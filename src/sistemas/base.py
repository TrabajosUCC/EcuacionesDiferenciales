"""Interfaz común de los sistemas de oscilación."""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class Campo:
    clave: str
    etiqueta: str
    unidad: str
    defecto: float
    regla: str = "pos"      # 'pos' (>0), 'nonneg' (>=0) o 'any'
    ayuda: str = ""


class Sistema:
    clave = ""
    nombre = ""
    descripcion = ""
    campos: tuple = ()
    estados: tuple = ()
    tiempo_defecto = 10.0
    paso_defecto = 0.01

    # ---- validación -------------------------------------------------
    def defectos(self) -> Dict[str, float]:
        return {c.clave: c.defecto for c in self.campos}

    def validar(self, p: dict) -> List[str]:
        errs: List[str] = []
        for c in self.campos:
            v = p.get(c.clave)
            if (not isinstance(v, numbers.Real) or isinstance(v, bool)
                    or not math.isfinite(v)):
                errs.append(f"{c.etiqueta}: debe ser un número finito.")
            elif c.regla == "pos" and v <= 0:
                errs.append(f"{c.etiqueta}: debe ser estrictamente positivo.")
            elif c.regla == "nonneg" and v < 0:
                errs.append(f"{c.etiqueta}: no puede ser negativo.")
        return errs if errs else self.validar_extra(p)

    def validar_extra(self, p: dict) -> List[str]:
        return []

    # ---- modelo -------------------------------------------------------
    def rhs(self, p: dict):
        raise NotImplementedError

    def z0(self, p: dict) -> np.ndarray:
        raise NotImplementedError

    def analitica(self, p: dict, t: np.ndarray) -> Optional[np.ndarray]:
        """Matriz (nt, n_estados) con la solución cerrada, o None si no existe."""
        return None

    # ---- salidas ------------------------------------------------------
    def derivadas(self, p: dict) -> List[tuple]:
        """Lista de (etiqueta, valor, unidad)."""
        return []

    def desarrollo(self, p: dict, cfg: dict) -> List[tuple]:
        return []

    def figuras(self, p, t, Z, Za) -> List[dict]:
        return []

    def figuras_comparacion(self, p, t, Zs: dict, ref: np.ndarray) -> List[dict]:
        return []

    def interpretacion(self, p, t, Z) -> List[str]:
        return []

    def magnitud_principal(self) -> int:
        """Índice del estado usado para medir errores."""
        return 0
