"""Formato de números y bloques de texto/LaTeX para el desarrollo matemático."""
from __future__ import annotations


def fm(x: float) -> str:
    """Número con 6 cifras significativas."""
    return f"{x:.6g}"


def fp(x: float) -> str:
    """Número entre paréntesis si es negativo (para sustituir en fórmulas)."""
    return f"({x:.6g})" if x < 0 else f"{x:.6g}"


def sg(x: float) -> str:
    """Término con signo explícito: '+ 3' o '- 3'."""
    return f"{'+' if x >= 0 else '-'} {abs(x):.6g}"


def H(texto: str):
    return ("h", texto)


def T(texto: str):
    return ("t", texto)


def L(texto: str):
    return ("l", texto)
