"""Péndulo simple no lineal: θ'' + (g/L) sin θ = 0."""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ellipk

from ..solucionadores.integradores import referencia
from ..utilidades.formato import H, L, T, fm, sg
from ..visualizaciones.figuras import figura, figuras_oscilador, serie
from .base import Campo, Sistema

NOMBRES = {"x": ("Ángulo θ", "θ (rad)"), "v": ("Velocidad angular", "ω (rad/s)"),
           "a": ("Aceleración angular", "α (rad/s²)"), "E": "Energía por unidad de masa (J/kg)"}


def periodo_exacto(L_, g, th0, w0=0.0):
    """Período exacto con la integral elíptica K. None si el péndulo gira (no oscila)."""
    cmax = math.cos(th0) - w0 * w0 * L_ / (2.0 * g)
    if cmax <= -1.0:
        return None
    thmax = math.acos(max(min(cmax, 1.0), -1.0))
    if thmax == 0:
        return 2 * math.pi * math.sqrt(L_ / g)
    return 4.0 * math.sqrt(L_ / g) * ellipk(math.sin(thmax / 2.0) ** 2)


def solucion_lineal(L_, g, th0, w0, t):
    w = math.sqrt(g / L_)
    return th0 * np.cos(w * t) + (w0 / w) * np.sin(w * t), -th0 * w * np.sin(w * t) + w0 * np.cos(w * t)


def barrido_aproximacion(L_, g, angulos_deg, n_periodos=3.0):
    """Máx |θ_nl − θ_lin| durante n períodos lineales, para cada ángulo inicial (ω₀=0)."""
    T0 = 2 * math.pi * math.sqrt(L_ / g)
    t = np.linspace(0.0, n_periodos * T0, 1500)
    out = []
    for a in angulos_deg:
        th0 = math.radians(a)
        f = lambda tt, z: np.array([z[1], -(g / L_) * math.sin(z[0])])
        nl = referencia(f, [th0, 0.0], t)[:, 0]
        lineal, _ = solucion_lineal(L_, g, th0, 0.0, t)
        out.append(float(np.max(np.abs(nl - lineal))))
    return np.array(out)


class PenduloNoLineal(Sistema):
    clave = "pendulo"
    nombre = "Péndulo simple no lineal"
    descripcion = ("Péndulo ideal sin fricción con la ecuación **no lineal** completa "
                   "$\\ddot\\theta+\\frac{g}{L}\\sin\\theta=0$, comparada con la aproximación "
                   "lineal de ángulos pequeños $\\sin\\theta\\approx\\theta$.")
    campos = (Campo("L", "Longitud L", "m", 1.0, "pos"),
              Campo("g", "Gravedad g", "m/s²", 9.81, "pos"),
              Campo("th0", "Ángulo inicial θ₀", "grados", 60.0, "any", "entre −180° y 180°"),
              Campo("w0", "Velocidad angular inicial ω₀", "rad/s", 0.0, "any"))
    estados = ("theta", "omega")
    tiempo_defecto, paso_defecto = 10.0, 0.01

    def validar_extra(self, p):
        return [] if abs(p["th0"]) <= 180 else ["Ángulo inicial θ₀: debe estar entre −180° y 180°."]

    def rhs(self, p):
        r = p["g"] / p["L"]
        return lambda t, z: np.array([z[1], -r * math.sin(z[0])])

    def z0(self, p):
        return np.array([math.radians(p["th0"]), p["w0"]])

    def _E(self, p, th, w):
        return 0.5 * p["L"] ** 2 * w ** 2 + p["g"] * p["L"] * (1 - np.cos(th))

    def derivadas(self, p):
        L_, g, th0, w0 = p["L"], p["g"], math.radians(p["th0"]), p["w0"]
        wl = math.sqrt(g / L_)
        T0 = 2 * math.pi / wl
        T2 = T0 * (1 + th0 ** 2 / 16 + 11 * th0 ** 4 / 3072)
        Tex = periodo_exacto(L_, g, th0, w0)
        out = [("Frecuencia angular lineal √(g/L)", wl, "rad/s"),
               ("Período lineal T₀ = 2π√(L/g)", T0, "s"),
               ("Período con serie (2 términos)", T2, "s")]
        if Tex is None:
            out.append(("Período exacto", "no definido: el péndulo da vueltas completas", ""))
        else:
            out += [("Período exacto 4√(L/g)·K(k)", Tex, "s"),
                    ("Error relativo del período lineal", abs(T0 - Tex) / Tex * 100, "%")]
        out.append(("Energía mecánica específica E/m", float(self._E(p, th0, w0)), "J/kg"))
        return out

    def figuras(self, p, t, Z, Za):
        L_, g = p["L"], p["g"]
        thl, wl = solucion_lineal(L_, g, math.radians(p["th0"]), p["w0"], t)
        a = -(g / L_) * np.sin(Z[:, 0])
        E = self._E(p, Z[:, 0], Z[:, 1])
        El = 0.5 * L_ ** 2 * wl ** 2 + 0.5 * g * L_ * thl ** 2
        figs = figuras_oscilador(t, Z[:, 0], Z[:, 1], a,
                                 {"Energía exacta (no lineal)": E, "Energía del modelo lineal": El},
                                 NOMBRES)
        figs.insert(0, figura("Ángulo: no lineal vs lineal", "t (s)", "θ (rad)",
                              [serie("No lineal (numérica)", t, Z[:, 0]),
                               serie("Lineal (sin θ ≈ θ)", t, thl, "dash")]))
        figs.insert(1, figura("Diferencia |θ_no lineal − θ_lineal|", "t (s)", "rad",
                              [serie("|Δθ|", t, np.abs(Z[:, 0] - thl))]))
        figs.append(figura("Retrato de fase: no lineal vs lineal", "θ (rad)", "ω (rad/s)",
                           [serie("No lineal", Z[:, 0], Z[:, 1]), serie("Lineal", thl, wl, "dash")]))
        return figs

    def figuras_comparacion(self, p, t, Zs, ref):
        ang = np.linspace(2.0, 170.0, 28)
        d = barrido_aproximacion(p["L"], p["g"], ang)
        return [figura("Error de la aproximación lineal vs ángulo inicial (ω₀=0, 3 períodos)",
                       "θ₀ (grados)", "máx |θ_nl − θ_lin| (rad)", [serie("Diferencia máxima", ang, d)])]

    def desarrollo(self, p, cfg):
        L_, g, th0d, w0 = p["L"], p["g"], p["th0"], p["w0"]
        th0 = math.radians(th0d)
        wl = math.sqrt(g / L_)
        T0 = 2 * math.pi / wl
        Tex = periodo_exacto(L_, g, th0, w0)
        b = [H("1. Ecuación diferencial"), L(r"\ddot{\theta}+\frac{g}{L}\sin\theta=0"),
             T("Se obtiene de $mL^2\\ddot\\theta=-mgL\\sin\\theta$. **No tiene solución elemental cerrada** "
               "(se expresa con funciones elípticas de Jacobi), por eso se resuelve numéricamente."),
             H("2. Sustitución de parámetros"),
             L(rf"\ddot\theta+\frac{{{fm(g)}}}{{{fm(L_)}}}\sin\theta=\ddot\theta+{fm(g/L_)}\sin\theta=0"),
             L(rf"\theta(0)={fm(th0d)}^\circ={fm(th0)}\ \text{{rad}},\qquad \dot\theta(0)={fm(w0)}\ \text{{rad/s}}"),
             H("3. Aproximación de ángulo pequeño"),
             L(r"\sin\theta\approx\theta-\frac{\theta^3}{6}+\cdots\;\Rightarrow\;\ddot\theta+\frac{g}{L}\theta=0"),
             L(rf"\theta_{{lin}}(t)={fm(th0)}\cos({fm(wl)}\,t) {sg(w0/wl)}\sin({fm(wl)}\,t),\qquad T_0=2\pi\sqrt{{L/g}}={fm(T0)}\ \text{{s}}"),
             T(f"El error relativo de $\\sin\\theta\\approx\\theta$ en $\\theta_0$ es ≈ $\\theta_0^2/6$ = "
               f"{fm(th0**2/6*100)} %. Para $\\theta_0$ grande el período real **aumenta** y el modelo lineal pierde validez."),
             H("4. Período"),
             L(r"T=4\sqrt{\frac{L}{g}}\,K(k),\quad k=\sin\frac{\theta_{max}}{2},\quad T\approx T_0\left(1+\frac{\theta_0^2}{16}+\frac{11\theta_0^4}{3072}\right)")]
        if Tex is not None:
            b.append(L(rf"T_{{exacto}}={fm(Tex)}\ \text{{s}}"))
        else:
            b.append(T("Con estas condiciones el péndulo **gira** (no oscila): el período no está definido."))
        b += [H("5. Energía"), L(r"\frac{E}{m}=\tfrac12L^2\dot\theta^2+gL(1-\cos\theta)"),
              T("Constante en el modelo no lineal. En el lineal se usa $\\tfrac12 gL\\theta^2$ en lugar de $gL(1-\\cos\\theta)$.")]
        return b

    def interpretacion(self, p, t, Z):
        th0 = math.radians(p["th0"])
        thl, _ = solucion_lineal(p["L"], p["g"], th0, p["w0"], t)
        dmax = float(np.max(np.abs(Z[:, 0] - thl)))
        return [f"La diferencia máxima entre el modelo no lineal y el lineal en la ventana simulada es {fm(dmax)} rad.",
                "Para ángulos pequeños (< ~10°) ambos coinciden; al crecer θ₀ el péndulo real es más lento "
                "(período mayor) y la solución lineal se desfasa progresivamente.",
                "Ver la pestaña *Comparación* para el barrido del error frente al ángulo inicial."]
