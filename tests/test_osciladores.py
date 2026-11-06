"""Pruebas de los sistemas de oscilación, solucionadores y servicio."""
import math

import numpy as np

from src.analitica_osc import lineal as lin
from src.servicio import ErrorValidacion, config_defecto, ejecutar, tabla_resultados
from src.sistemas import listar, por_clave
from src.sistemas.acoplado import matrices, modos
from src.sistemas.pendulo import barrido_aproximacion, periodo_exacto
from src.solucionadores import integradores as ig


def corre(clave, cambios=None, **cfg):
    s = por_clave(clave)
    p = {**s.defectos(), **(cambios or {})}
    c = {**config_defecto(s), **cfg}
    return s, p, ejecutar(s, p, c)


def lanza(fn):
    try:
        fn()
    except (ErrorValidacion, ValueError):
        return True
    return False


# ---------- solucionadores ----------
def test_orden_euler_y_rk4():
    f = lambda t, z: np.array([z[1], -z[0]])
    errs = {}
    for m, fn in (("e", ig.euler), ("r", ig.rk4)):
        errs[m] = []
        for h in (0.1, 0.05):
            t = ig.malla(2.0, h)
            Z = fn(f, [1.0, 0.0], t)
            errs[m].append(abs(Z[-1, 0] - math.cos(2.0)))
    assert 1.8 < errs["e"][0] / errs["e"][1] < 2.2      # orden 1
    assert 14 < errs["r"][0] / errs["r"][1] < 18         # orden 4


def test_rk4_generico_igual_al_original():
    from src.modelo import BASE
    from src.numerico import rk4 as rk4_orig
    from src.modelo import campo_vectorial, entrada_escalon
    t, x, y = rk4_orig(BASE, 1.0)
    d = lambda tt: float(entrada_escalon(tt, BASE.d0))
    Z = ig.rk4(lambda tt, z: campo_vectorial(tt, z, BASE, d), [0.0, 0.0], t)
    assert np.allclose(Z[:, 0], x, atol=1e-12) and np.allclose(Z[:, 1], y, atol=1e-12)


def test_malla_no_modifica_paso():
    t = ig.malla(10.0, 3.0)
    assert np.isclose(t[1], 3.0) and np.isclose(t[-1], 9.0)


# ---------- masa-resorte simple ----------
def test_simple_analitica_vs_numerica_y_energia():
    s, p, r = corre("simple", T=20.0, h=0.005)
    assert r["comparacion"].set_index("Método").loc[ig.RK4, "Error absoluto máx"] < 1e-6
    E = 0.5 * p["m"] * r["Z"][:, 1] ** 2 + 0.5 * p["k"] * r["Z"][:, 0] ** 2
    assert (E.max() - E.min()) / E[0] < 1e-6


def test_simple_frecuencias():
    s = por_clave("simple")
    d = {e: v for e, v, u in s.derivadas({**s.defectos(), "m": 2.0, "k": 50.0})}
    assert math.isclose(d["Frecuencia angular natural ω₀"], 5.0)
    assert math.isclose(d["Período T"], 2 * math.pi / 5.0)
    assert math.isclose(d["Frecuencia natural f₀"], 5.0 / (2 * math.pi))


def test_euler_gana_energia_y_rk4_no():
    _, _, r = corre("simple", T=20.0, h=0.01)
    p = por_clave("simple").defectos()
    en = lambda Z: 0.5 * p["m"] * Z[:, 1] ** 2 + 0.5 * p["k"] * Z[:, 0] ** 2
    Ee, Er = en(r["Zs"][ig.EULER]), en(r["Zs"][ig.RK4])
    assert Ee[-1] > 1.05 * Ee[0]
    assert abs(Er[-1] / Er[0] - 1) < 1e-6
    # Euler explícito: E_n = (1 + w0^2 h^2)^n E_0 (crecimiento exacto, no oscilatorio)
    n, h = len(r["t"]) - 1, 0.01
    w0 = math.sqrt(p["k"] / p["m"])
    assert math.isclose(Ee[-1] / Ee[0], (1 + w0 * w0 * h * h) ** n, rel_tol=1e-10)
    # La solución analítica conserva la energía
    Ea = en(r["Za"])
    assert (Ea.max() - Ea.min()) / Ea[0] < 1e-12


# ---------- amortiguado ----------
def test_tres_regimenes_y_energia_decreciente():
    for c, esperado in ((0.8, "subamortiguado"), (4.0, "críticamente amortiguado"), (10.0, "sobreamortiguado")):
        s, p, r = corre("amortiguado", {"c": c}, T=15.0, h=0.005)
        assert lin.regimen(p["m"], p["c"], p["k"]) == esperado
        assert r["comparacion"].set_index("Método").loc[ig.RK4, "Error absoluto máx"] < 1e-6
        assert dict((e, v) for e, v, u in r["derivadas"])["Régimen"] == esperado
        E = 0.5 * r["Z"][:, 1] ** 2 + 2.0 * r["Z"][:, 0] ** 2
        assert np.all(np.diff(E) <= 1e-9)


def test_amortiguado_condiciones_iniciales():
    for c in (0.8, 4.0, 10.0):
        x, v = lin.homogenea(1.0, c, 4.0, 0.7, -0.3, np.array([0.0]))
        assert math.isclose(x[0], 0.7) and math.isclose(v[0], -0.3)


# ---------- forzado ----------
def test_forzado_analitica_y_cambio_con_frecuencia():
    _, _, r1 = corre("forzado", {"W": 1.0}, T=30.0, h=0.005)
    _, _, r2 = corre("forzado", {"W": 3.0}, T=30.0, h=0.005)
    assert r1["comparacion"].set_index("Método").loc[ig.RK4, "Error absoluto máx"] < 1e-6
    assert not np.allclose(r1["Z"][:, 0], r2["Z"][:, 0])
    A = lambda r: dict((e, v) for e, v, u in r["derivadas"])["Amplitud estacionaria A"]
    assert A(r1) > 0 and not math.isclose(A(r1), A(r2))


def test_amplitud_y_pico_de_resonancia():
    m, c, k, F0 = 1.0, 0.4, 4.0, 1.0
    W = np.linspace(0.01, 6, 200000)
    A = F0 / np.sqrt((k - m * W ** 2) ** 2 + (c * W) ** 2)
    Wr, Mr = lin.frecuencia_pico(m, c, k)
    assert abs(W[A.argmax()] - Wr) < 1e-3
    assert math.isclose(A.max(), F0 * Mr, rel_tol=1e-6)


def test_resonancia_pura_crece_linealmente():
    s, p, r = corre("forzado", {"c": 0.0, "W": 2.0, "x0": 0.0, "v0": 0.0}, T=30.0, h=0.005)
    assert r["comparacion"].set_index("Método").loc[ig.RK4, "Error absoluto máx"] < 1e-5
    assert np.abs(r["Z"][:, 0]).max() > 10 * np.abs(r["Z"][:200, 0]).max()


def test_mas_amortiguamiento_menos_amplitud():
    a = [lin.amplitud(1.0, c, 4.0, 1.0, 2.0) for c in (0.1, 0.5, 2.0)]
    assert a[0] > a[1] > a[2]


# ---------- péndulo ----------
def test_pendulo_energia_conservada_y_periodo():
    s, p, r = corre("pendulo", {"th0": 60.0}, T=10.0, h=0.002)
    th, w = r["Z"][:, 0], r["Z"][:, 1]
    E = 0.5 * w ** 2 + 9.81 * (1 - np.cos(th))
    assert (E.max() - E.min()) / E[0] < 1e-6
    T = periodo_exacto(1.0, 9.81, math.radians(60.0))
    i = np.where((th[:-1] > 0) & (th[1:] <= 0))[0]            # cruces descendentes por cero
    assert abs((r["t"][i[1]] - r["t"][i[0]]) - T) < 0.01


def test_pendulo_error_lineal_crece_con_angulo():
    d = barrido_aproximacion(1.0, 9.81, [5, 20, 45, 90, 150])
    assert np.all(np.diff(d) > 0) and d[0] < 0.01 and d[-1] > 0.5


def test_pendulo_lineal_coincide_si_angulo_pequeno():
    _, _, r = corre("pendulo", {"th0": 1.0}, T=5.0, h=0.005)
    from src.sistemas.pendulo import solucion_lineal
    th, _ = solucion_lineal(1.0, 9.81, math.radians(1.0), 0.0, r["t"])
    assert np.abs(r["Z"][:, 0] - th).max() < 1e-4


# ---------- acoplado ----------
def test_acoplado_frecuencias_conocidas():
    p = {**por_clave("acoplado").defectos(), "m1": 1.0, "m2": 1.0, "k1": 4.0, "k2": 2.0, "k3": 4.0}
    w, Phi = modos(p)
    assert np.allclose(w, [2.0, math.sqrt(8.0)])              # ω1²=k=4, ω2²=k+2k2=8
    M, K = matrices(p)
    assert np.allclose(Phi.T @ M @ Phi, np.eye(2))
    assert np.allclose(K @ Phi, M @ Phi @ np.diag(w ** 2))   # K φ = ω² M φ


def test_acoplado_analitica_numerica_y_energia():
    s, p, r = corre("acoplado", {"m2": 2.0, "k3": 3.0}, T=20.0, h=0.005)
    assert r["comparacion"].set_index("Método").loc[ig.RK4, "Error absoluto máx"] < 1e-6
    Ta, U = s._energias(p, r["Z"])
    E = Ta + U
    assert (E.max() - E.min()) / E[0] < 1e-6


def test_acoplado_un_solo_modo():
    p = {**por_clave("acoplado").defectos()}
    w, Phi = modos(p)
    p.update(x1=Phi[0, 1], x2=Phi[1, 1], v1=0.0, v2=0.0)
    s = por_clave("acoplado")
    t = ig.malla(5.0, 0.01)
    Z = s.analitica(p, t)
    assert np.allclose(Z[:, 0], p["x1"] * np.cos(w[1] * t))


# ---------- RLC ----------
def test_rlc_analitica_y_balance_energia():
    s, p, r = corre("rlc", T=3.0, h=0.0005)
    assert r["comparacion"].set_index("Método").loc[ig.RK4, "Error absoluto máx"] < 1e-6
    ser = s._series(p, r["t"], r["Z"])
    bal = ser["Esum"] - (ser["Ealm"] - ser["Ealm"][0]) - ser["Edis"]
    assert np.abs(bal).max() < 1e-3 * ser["Ealm"].max()


def test_rlc_analogia_masa_resorte():
    s, p, r = corre("rlc", T=2.0, h=0.001)
    x = lin.forzada(p["L"], p["R"], 1 / p["C"], p["V0"], p["W"], p["q0"], p["i0"], r["t"])["x"]
    assert np.allclose(r["Za"][:, 0], x)


def test_rlc_kirchhoff_y_unidades():
    s, p, r = corre("rlc", T=1.0, h=0.0005)
    ser = s._series(p, r["t"], r["Z"])
    assert np.allclose(ser["vL"] + ser["vR"] + ser["vC"], ser["V"])
    assert math.isclose(1 / math.sqrt(p["L"] * p["C"]), 1 / math.sqrt(0.5 * 0.01))   # ω0 en rad/s


# ---------- validaciones y servicio ----------
def test_validaciones_rechazan_valores_invalidos():
    s = por_clave("simple")
    base = s.defectos()
    for cambio in ({"m": 0.0}, {"m": -1.0}, {"k": 0.0}, {"x0": float("nan")}, {"v0": float("inf")}):
        assert s.validar({**base, **cambio}), cambio
    assert por_clave("pendulo").validar({**por_clave("pendulo").defectos(), "L": 0.0})
    assert por_clave("pendulo").validar({**por_clave("pendulo").defectos(), "g": -9.8})
    assert por_clave("pendulo").validar({**por_clave("pendulo").defectos(), "th0": 270.0})
    assert por_clave("rlc").validar({**por_clave("rlc").defectos(), "L": 0.0})
    assert por_clave("rlc").validar({**por_clave("rlc").defectos(), "C": -1.0})
    assert por_clave("amortiguado").validar({**por_clave("amortiguado").defectos(), "c": 0.0})
    assert por_clave("forzado").validar({**por_clave("forzado").defectos(), "c": -0.1})


def test_config_invalida():
    s = por_clave("simple")
    for cfg in ({"T": 0.0}, {"T": -1.0}, {"h": 0.0}, {"h": 100.0}, {"h": 1e-9}, {"T": float("nan")}):
        assert lanza(lambda: ejecutar(s, s.defectos(), {**config_defecto(s), **cfg})), cfg


def test_todos_los_sistemas_sin_nan_ni_inf_y_con_contenido():
    for s in listar():
        r = ejecutar(s, s.defectos(), config_defecto(s))
        assert np.all(np.isfinite(r["Z"])) and len(r["figuras"]) >= 5
        assert r["desarrollo"] and r["interpretacion"] and r["derivadas"]
        assert len(tabla_resultados(r)) > 1
        for f in r["figuras"] + r["figuras_comparacion"]:
            for se in f["series"]:
                assert len(se["x"]) == len(se["y"])


def test_resultados_dependen_de_los_parametros():
    _, _, a = corre("simple", {"k": 4.0})
    _, _, b = corre("simple", {"k": 9.0})
    assert not np.allclose(a["Z"][:, 0], b["Z"][:, 0])
    assert "9" in "".join(t for _, t in b["desarrollo"]) and "4" in "".join(t for _, t in a["desarrollo"])


def test_metodos_seleccionables():
    for m in ig.METODOS:
        _, _, r = corre("simple", metodo=m, T=5.0, h=0.01)
        assert r["cfg"]["metodo"] == m and np.all(np.isfinite(r["Z"]))
