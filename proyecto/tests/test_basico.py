"""
Pruebas básicas del proyecto de autoescalado.

Se verifica:
  * condiciones iniciales y valor estacionario de la solución analítica,
  * residuo de la EDO con la solución simbólica en los tres regímenes,
  * comparación con solve_ivp de tolerancia estricta,
  * orden de convergencia de Euler (~1) y RK4 (~4),
  * monotonía de zeta al variar tau y K,
  * consistencia de la resonancia y de la solución sinusoidal.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analitica import (
    amplitud_resonancia,
    comparar_con_solve_ivp,
    frecuencia_resonancia,
    metricas_escalon,
    residuo_simbolico,
    respuesta_frecuencia,
    x_escalon,
    x_sinusoidal,
    y_escalon,
    y_sinusoidal,
)
from src.convergencia import tabla_convergencia
from src.modelo import BASE, Parametros, verificar_dimensiones


# Casos que cubren los tres regímenes:
#   BASE (zeta=0.5), tau=5,K=0.005 (zeta=1), tau=5,K=0.0025 (zeta=1.414)
CASOS_REGIMENES = [
    BASE,
    BASE.con_parametros(tau=5.0, K=0.005),
    BASE.con_parametros(tau=5.0, K=0.0025),
]


def test_parametros_base_correctos() -> None:
    """Comprueba los valores teóricos del caso base."""
    assert BASE.wn == pytest.approx(0.05, rel=1e-12)
    assert BASE.zeta == pytest.approx(0.5, rel=1e-12)
    assert BASE.regimen == "subamortiguado"
    assert BASE.x_ss == pytest.approx(400.0, rel=1e-12)
    assert BASE.y_ss == pytest.approx(2.0, rel=1e-12)
    assert verificar_dimensiones(BASE)["estable"] is True


@pytest.mark.parametrize("p", CASOS_REGIMENES)
def test_condiciones_iniciales_y_estacionario(p: Parametros) -> None:
    """x(0)=y(0)=0 y la solución tiende a x_ss, y_ss."""
    assert float(x_escalon(0.0, p)) == pytest.approx(0.0, abs=1e-12)
    assert float(y_escalon(0.0, p)) == pytest.approx(0.0, abs=1e-12)

    # Un tiempo muy posterior al horizonte permite verificar el límite.
    t_lejano = 50.0 * p.T
    assert float(x_escalon(t_lejano, p)) == pytest.approx(p.x_ss, rel=1e-7)
    assert float(y_escalon(t_lejano, p)) == pytest.approx(p.y_ss, rel=1e-7)


@pytest.mark.parametrize("p", CASOS_REGIMENES)
def test_residuo_edo_simbolica(p: Parametros) -> None:
    """El residuo de la EDO con la solución simbólica debe ser ~0."""
    residuo = residuo_simbolico(p, n_muestras=80)
    assert residuo["residuo_max_x"] < 1e-8
    assert residuo["residuo_max_y"] < 1e-8


@pytest.mark.parametrize("p", CASOS_REGIMENES)
def test_analitica_contra_solve_ivp(p: Parametros) -> None:
    """La fórmula cerrada coincide con solve_ivp con tolerancias estrictas."""
    t = np.linspace(0.0, p.T, 1201)
    error = comparar_con_solve_ivp(p, t)
    assert error["error_max_x"] < 1e-5
    assert error["error_max_y"] < 1e-6


def test_solucion_sinusoidal_condiciones_y_solve_ivp() -> None:
    """La solución completa sinusoidal cumple x(0)=y(0)=0 y la EDO."""
    p = BASE
    w = 0.071
    assert float(x_sinusoidal(0.0, p, w)) == pytest.approx(0.0, abs=1e-10)
    assert float(y_sinusoidal(0.0, p, w)) == pytest.approx(0.0, abs=1e-10)

    t = np.linspace(0.0, p.T, 1501)
    error = comparar_con_solve_ivp(p, t, w=w)
    assert error["error_max_x"] < 1e-5
    assert error["error_max_y"] < 1e-6


def test_orden_convergencia() -> None:
    """Euler debe mostrar orden ~1 y RK4 orden ~4."""
    df = tabla_convergencia(BASE, pasos=(1.0, 0.5, 0.25))
    orden_euler = df["orden_euler"].to_numpy()[1:]
    orden_rk4 = df["orden_rk4"].to_numpy()[1:]

    assert np.all(np.abs(orden_euler - 1.0) < 0.1)
    assert np.all(np.abs(orden_rk4 - 4.0) < 0.15)

    # Para h pequeno, ambos métodos deben acercarse a la analítica.
    assert df["error_euler"].iloc[-1] < 3.0
    assert df["error_rk4"].iloc[-1] < 1e-5


def test_zeta_disminuye_con_tau_y_K() -> None:
    """Más tau o más K implican menos amortiguamiento (zeta menor)."""
    zetas_tau = [BASE.con_parametros(tau=t).zeta for t in (5.0, 20.0, 60.0)]
    zetas_K = [BASE.con_parametros(K=k).zeta for k in (0.0025, 0.005, 0.02)]

    assert zetas_tau[0] > zetas_tau[1] > zetas_tau[2]
    assert zetas_K[0] > zetas_K[1] > zetas_K[2]


def test_resonancia_es_el_maximo() -> None:
    """La frecuencia de resonancia debe maximizar |G(jw)|."""
    p = BASE
    wr = frecuencia_resonancia(p)
    mr = amplitud_resonancia(p)
    assert wr > 0.0
    # |G| en la resonancia no puede ser menor que en frecuencias cercanas.
    for w in np.linspace(0.5 * wr, 1.5 * wr, 31):
        assert mr >= float(respuesta_frecuencia(p, float(w))["mag_x"]) - 1e-10

    # En el caso crítico no existe un máximo interior.
    p_critico = BASE.con_parametros(tau=5.0, K=0.005)
    assert frecuencia_resonancia(p_critico) == 0.0


def test_metricas_base_coherentes() -> None:
    """Las métricas del caso base deben ser finitas y físicamente válidas."""
    m = metricas_escalon(BASE, n_muestras=40_001)
    assert m["sobrepaso_pct"] == pytest.approx(29.8436, rel=1e-3)
    assert 40.0 < m["t_pico_s"] < 60.0
    assert 100.0 < m["t_asentamiento_s"] < 250.0
    assert m["periodo_s"] == pytest.approx(2.0 * np.pi / (BASE.wn * np.sqrt(0.75)),
                                           rel=1e-9)


# ---------------------------------------------------------------------------
# Extensión opcional con retardo puro
# ---------------------------------------------------------------------------

def test_retardo_critico_y_pade() -> None:
    """El retardo crítico es positivo y Padé 1 sobreestima ligeramente."""
    from src.extension_retardo import (
        estable_pade,
        theta_critico_exacto,
        theta_critico_pade,
    )

    theta_e = theta_critico_exacto(BASE)
    theta_p = theta_critico_pade(BASE)
    assert theta_e > 0.0
    assert theta_p > theta_e
    assert estable_pade(BASE, 0.0) is True
    assert estable_pade(BASE, 0.5 * theta_e) is True
    assert estable_pade(BASE, 1.5 * theta_e) is False


def test_simulacion_retardo_estable_e_inestable() -> None:
    """La amplitud decae por debajo de θc y crece por encima de θc."""
    from src.extension_retardo import simular_retardo, theta_critico_exacto

    theta_c = theta_critico_exacto(BASE)
    # Horizontes cortos para mantener la prueba rápida.
    t_e, x_e, _ = simular_retardo(BASE, 0.6 * theta_c, h=0.02, T=300.0)
    t_i, x_i, _ = simular_retardo(BASE, 1.5 * theta_c, h=0.02, T=500.0)

    # En el caso estable la amplitud del último tramo es menor que la del
    # transitorio inicial; en el inestable ocurre lo contrario.
    mitad_e = len(t_e) // 2
    mitad_i = len(t_i) // 2
    amplitud_inicial_e = np.max(np.abs(x_e[:mitad_e] - BASE.x_ss))
    amplitud_final_e = np.max(np.abs(x_e[mitad_e:] - BASE.x_ss))
    assert amplitud_final_e < amplitud_inicial_e

    amplitud_inicial_i = np.max(np.abs(x_i[:mitad_i]))
    amplitud_final_i = np.max(np.abs(x_i[mitad_i:]))
    assert amplitud_final_i > 1.8 * amplitud_inicial_i
