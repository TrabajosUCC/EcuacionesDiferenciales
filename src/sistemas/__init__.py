"""Registro de sistemas de oscilación disponibles."""
from .masa_resorte import MasaResorteAmortiguado, MasaResorteForzado, MasaResorteSimple


def _construir():
    from .pendulo import PenduloNoLineal
    from .acoplado import DosMasasAcopladas
    from .rlc import CircuitoRLC
    return [MasaResorteSimple(), MasaResorteAmortiguado(), MasaResorteForzado(),
            PenduloNoLineal(), DosMasasAcopladas(), CircuitoRLC()]


def listar():
    return _construir()


def por_nombre(nombre):
    for s in _construir():
        if s.nombre == nombre:
            return s
    raise KeyError(nombre)


def por_clave(clave):
    for s in _construir():
        if s.clave == clave:
            return s
    raise KeyError(clave)
