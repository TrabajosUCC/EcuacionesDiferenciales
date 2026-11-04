"""
Prueba de humo de la demo opcional de Streamlit.

Se salta automáticamente si `streamlit` o `plotly` no están instalados, para
que la suite principal (`pytest`) siga funcionando solo con las dependencias
obligatorias. Si están instalados, ejecuta el script completo con AppTest y
comprueba que no haya excepciones, incluso al cambiar parámetros y al pasar a
la carga sinusoidal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest

# AppTest resuelve rutas relativas contra este archivo, así que se usa la ruta
# absoluta del script de la app (un nivel arriba de tests/).
RUTA_APP = Path(__file__).resolve().parents[1] / "app_streamlit.py"


def _app() -> AppTest:
    """Crea la app de prueba con un tiempo de espera holgado."""
    return AppTest.from_file(str(RUTA_APP), default_timeout=120)


def test_app_arranca_sin_excepciones() -> None:
    """El script completo se ejecuta sin errores con los valores por defecto."""
    at = _app()
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_responde_a_cambio_de_parametros() -> None:
    """Mover los sliders vuelve a ejecutar el cálculo sin fallar."""
    at = _app()
    at.run()
    # Orden de los sliders del panel lateral: mu, K, tau, d0, T.
    assert len(at.sidebar.slider) == 5
    at.sidebar.slider[0].set_value(20.0)   # mu
    at.sidebar.slider[2].set_value(60.0)   # tau
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_carga_sinusoidal() -> None:
    """La opción de carga sinusoidal también se ejecuta sin errores."""
    at = _app()
    at.run()
    assert len(at.sidebar.radio) == 1
    at.sidebar.radio[0].set_value("Sinusoidal")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
