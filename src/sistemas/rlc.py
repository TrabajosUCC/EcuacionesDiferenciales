"""Circuito RLC serie:  L q'' + R q' + q/C = V0 cos(ω t)."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ..analitica_osc import lineal as lin
from ..utilidades.formato import H, L, T, fm
from ..visualizaciones.figuras import figura, serie
from .base import Campo, Sistema


class CircuitoRLC(Sistema):
    clave = "rlc"
    nombre = "Circuito RLC"
    descripcion = ("Circuito RLC **en serie** alimentado por una fuente alterna $V(t)=V_0\\cos(\\omega t)$. "
                   "Modelo: $L\\ddot q+R\\dot q+q/C=V_0\\cos\\omega t$, con $i=\\dot q$.")
    campos = (Campo("L", "Inductancia L", "H", 0.5), Campo("R", "Resistencia R", "Ω", 2.0, "nonneg"),
              Campo("C", "Capacitancia C", "F", 0.01),
              Campo("V0", "Amplitud del voltaje V₀", "V", 10.0, "any"),
              Campo("W", "Frecuencia de excitación ω", "rad/s", 10.0, "pos"),
              Campo("q0", "Carga inicial q₀", "C", 0.0, "any"),
              Campo("i0", "Corriente inicial i₀", "A", 0.0, "any"))
    estados = ("q", "i")
    tiempo_defecto, paso_defecto = 3.0, 0.001

    def _mck(self, p):
        return p["L"], p["R"], 1.0 / p["C"]

    def rhs(self, p):
        L_, R, C, V0, W = p["L"], p["R"], p["C"], p["V0"], p["W"]
        return lambda t, z: np.array([z[1], (V0 * math.cos(W * t) - R * z[1] - z[0] / C) / L_])

    def z0(self, p):
        return np.array([p["q0"], p["i0"]], float)

    def analitica(self, p, t):
        m, c, k = self._mck(p)
        r = lin.forzada(m, c, k, p["V0"], p["W"], p["q0"], p["i0"], t)
        return np.column_stack([r["x"], r["v"]])

    def _series(self, p, t, Z):
        L_, R, C, V0, W = p["L"], p["R"], p["C"], p["V0"], p["W"]
        q, i = Z[:, 0], Z[:, 1]
        V = V0 * np.cos(W * t)
        vR, vC = R * i, q / C
        vL = V - vR - vC                      # ley de voltajes de Kirchhoff: vL = L di/dt
        Ealm = 0.5 * L_ * i ** 2 + 0.5 * q ** 2 / C
        Edis = cumulative_trapezoid(R * i ** 2, t, initial=0.0)
        Esum = cumulative_trapezoid(V * i, t, initial=0.0)
        return dict(V=V, vR=vR, vC=vC, vL=vL, Ealm=Ealm, Edis=Edis, Esum=Esum)

    def derivadas(self, p):
        L_, R, C, V0, W = p["L"], p["R"], p["C"], p["V0"], p["W"]
        w0 = 1 / math.sqrt(L_ * C)
        z = lin.zeta(L_, R, 1 / C)
        X = W * L_ - 1 / (W * C)
        Zmod = math.hypot(R, X)
        out = [("Frecuencia natural ω₀ = 1/√(LC)", w0, "rad/s"),
               ("Frecuencia natural f₀", w0 / (2 * math.pi), "Hz"),
               ("Coeficiente de amortiguamiento α = R/2L", R / (2 * L_), "1/s"),
               ("Amortiguamiento relativo ζ", z, ""),
               ("Régimen", lin.regimen(L_, R, 1 / C), ""),
               ("R crítica = 2√(L/C)", 2 * math.sqrt(L_ / C), "Ω"),
               ("Reactancia neta X = ωL − 1/(ωC)", X, "Ω"),
               ("Impedancia |Z|", Zmod, "Ω")]
        if R > 0:
            out.append(("Factor de calidad Q = ω₀L/R", w0 * L_ / R, ""))
        if not lin.es_resonancia_pura(L_, R, 1 / C, W) and Zmod > 0:
            out += [("Amplitud de corriente estacionaria I₀ = V₀/|Z|", abs(V0) / Zmod, "A"),
                    ("Amplitud de carga estacionaria", lin.amplitud(L_, R, 1 / C, abs(V0), W), "C"),
                    ("Desfase de la carga φ", lin.fase(L_, R, 1 / C, W), "rad")]
        if abs(W - w0) / w0 < 0.05:
            out.append(("⚠ Aviso", "ω cerca de ω₀: resonancia eléctrica", ""))
        return out

    def figuras(self, p, t, Z, Za):
        s = self._series(p, t, Z)
        q = [serie("Numérica", t, Z[:, 0])]
        i = [serie("Numérica", t, Z[:, 1])]
        if Za is not None:
            q.append(serie("Analítica", t, Za[:, 0], "dash"))
            i.append(serie("Analítica", t, Za[:, 1], "dash"))
        bal = s["Esum"] - (s["Ealm"] - s["Ealm"][0]) - s["Edis"]
        return [figura("Carga del capacitor", "t (s)", "q (C)", q),
                figura("Corriente", "t (s)", "i (A)", i),
                figura("Voltaje de entrada", "t (s)", "V (V)", [serie("V(t)", t, s["V"])]),
                figura("Voltajes en los componentes", "t (s)", "V (V)",
                       [serie("Resistencia v_R=R·i", t, s["vR"]), serie("Inductor v_L=L·di/dt", t, s["vL"]),
                        serie("Capacitor v_C=q/C", t, s["vC"]), serie("Fuente", t, s["V"], "dot")]),
                figura("Energía almacenada y disipada", "t (s)", "E (J)",
                       [serie("Almacenada (½Li²+q²/2C)", t, s["Ealm"]),
                        serie("Disipada en R (∫R i² dt)", t, s["Edis"]),
                        serie("Entregada por la fuente (∫V i dt)", t, s["Esum"])]),
                figura("Balance de energía: entregada − almacenada − disipada", "t (s)", "J",
                       [serie("Residuo (≈0)", t, bal)]),
                figura("Retrato de fase", "q (C)", "i (A)", [serie("q–i", Z[:, 0], Z[:, 1])])]

    def desarrollo(self, p, cfg):
        L_, R, C, V0, W, q0, i0 = (p[k] for k in ("L", "R", "C", "V0", "W", "q0", "i0"))
        w0 = 1 / math.sqrt(L_ * C)
        X = W * L_ - 1 / (W * C)
        Zm = math.hypot(R, X)
        return [
            H("1. Ecuación diferencial (ley de voltajes de Kirchhoff)"),
            L(r"v_L+v_R+v_C=V(t)\;\Rightarrow\; L\ddot q+R\dot q+\frac{q}{C}=V_0\cos\omega t,\qquad i=\dot q"),
            H("2. Sustitución"),
            L(rf"{fm(L_)}\ddot q+{fm(R)}\dot q+\frac{{q}}{{{fm(C)}}}={fm(V0)}\cos({fm(W)}\,t)"),
            L(rf"q(0)={fm(q0)}\ \text{{C}},\qquad i(0)={fm(i0)}\ \text{{A}}"),
            H("3. Analogía con el sistema masa-resorte"),
            T("| Eléctrico | Mecánico |\n|---|---|\n| carga $q$ (C) | posición $x$ (m) |\n| corriente $i$ (A) | velocidad $v$ (m/s) |\n"
              "| inductancia $L$ (H) | masa $m$ (kg) |\n| resistencia $R$ (Ω) | amortiguamiento $c$ (N·s/m) |\n"
              "| $1/C$ (1/F) | constante elástica $k$ (N/m) |\n| voltaje $V_0$ (V) | fuerza $F_0$ (N) |\n"
              "| energía magnética $\\tfrac12Li^2$ | energía cinética |\n| energía eléctrica $q^2/2C$ | energía potencial elástica |"),
            H("4. Frecuencia natural y amortiguamiento"),
            L(rf"\omega_0=\frac{{1}}{{\sqrt{{LC}}}}=\frac{{1}}{{\sqrt{{{fm(L_)}\cdot{fm(C)}}}}}={fm(w0)}\ \text{{rad/s}},\qquad "
              rf"\alpha=\frac{{R}}{{2L}}={fm(R/(2*L_))}\ \text{{s}}^{{-1}},\qquad \zeta={fm(lin.zeta(L_, R, 1/C))}"),
            T(f"Régimen del transitorio: **{lin.regimen(L_, R, 1/C)}**. Resistencia crítica $2\\sqrt{{L/C}}={fm(2*math.sqrt(L_/C))}$ Ω."),
            H("5. Solución estacionaria (analítica)"),
            L(r"q_p(t)=Q\cos(\omega t-\varphi),\quad Q=\frac{V_0}{\sqrt{(1/C-L\omega^2)^2+R^2\omega^2}},\quad I_0=\omega Q=\frac{V_0}{|Z|},\quad |Z|=\sqrt{R^2+\left(\omega L-\frac1{\omega C}\right)^2}"),
            L(rf"X={fm(X)}\ \Omega,\quad |Z|={fm(Zm)}\ \Omega,\quad I_0={fm(abs(V0)/Zm) if Zm > 0 else '∞'}\ \text{{A}}"),
            T("La solución completa suma la homogénea (transitorio, mismo método que masa-resorte con $m\\to L$, $c\\to R$, $k\\to1/C$) "
              "ajustada a $q(0)$ e $i(0)$."),
            H("6. Voltajes y energía"),
            L(r"v_R=R\,i,\qquad v_C=\frac{q}{C},\qquad v_L=L\frac{di}{dt}=V-v_R-v_C"),
            L(r"\underbrace{\int_0^t V i\,dt'}_{\text{entregada}}=\underbrace{\Delta\!\left(\tfrac12Li^2+\tfrac{q^2}{2C}\right)}_{\text{almacenada}}+\underbrace{\int_0^t R\,i^2dt'}_{\text{disipada}}"),
        ]

    def interpretacion(self, p, t, Z):
        L_, R, C, W = p["L"], p["R"], p["C"], p["W"]
        w0 = 1 / math.sqrt(L_ * C)
        out = [f"La frecuencia natural es {fm(w0)} rad/s y la de la fuente {fm(W)} rad/s "
               f"({'inductivo (ωL > 1/ωC)' if W * L_ > 1 / (W * C) else 'capacitivo (1/ωC > ωL)'})."
               if abs(W * L_ - 1 / (W * C)) > 1e-12 else
               "El circuito está en **resonancia eléctrica** (ωL = 1/ωC): la impedancia es puramente R."]
        if R > 0:
            out.append("La resistencia disipa energía en forma de calor (efecto Joule); el transitorio decae "
                       f"como e^(−{fm(R/(2*L_))}·t) y queda la oscilación forzada a la frecuencia de la fuente.")
        else:
            out.append("Con R = 0 no hay disipación: el transitorio no decae.")
        return out
