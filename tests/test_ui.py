"""Prueba de la interfaz con streamlit.testing (se omite si streamlit no está instalado)."""
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def test_app_carga_y_calcula_todos_los_sistemas():
    at = st_testing.AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    opciones = list(at.sidebar.selectbox[0].options)
    assert len(opciones) == 6
    for op in opciones:
        at.sidebar.selectbox[0].select(op).run()
        assert not at.exception
        at.button[0].click().run()          # botón «Calcular»
        assert not at.exception, op
        assert len(at.error) == 0, op


def test_valores_invalidos_muestran_error():
    at = st_testing.AppTest.from_file("../app.py", default_timeout=60).run()
    at.number_input[0].set_value(-1.0)       # masa negativa
    at.button[0].click().run()
    assert len(at.error) >= 1

