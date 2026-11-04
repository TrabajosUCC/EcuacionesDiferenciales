"""Sistemas masa-resorte: simple, amortiguado y forzado."""
from __future__ import annotations

import math

import numpy as np

from ..analitica_osc import lineal as lin
from ..utilidades.formato import H, L, T, fm, fp, sg
from ..visualizaciones.figuras import figura, figuras_oscilador, serie
from .base import Campo, Sistema

NOMBRES = {"x": ("Posición", "x (m)"), "v": ("Velocidad", "v (m/s)"),
           "a": ("Aceleración", "a (m/s²)"), "E": "Energía (J)"}

C_M = Campo("m", "Masa m", "kg", 1.0, "pos")
C_K = Campo("k", "Constante elástica k", "N/m", 4.0, "pos")
C_C = Campo("c", "Coeficiente de amortiguamiento c", "N·s/m", 0.8, "pos")
C_X0 = Campo("x0", "Posición inicial x₀", "m", 0.5, "any")
C_V0 = Campo("v0", "Velocidad inicial v₀", "m/s", 0.0, "any")


def _latex_hom(d: dict) -> str:
    """x_h(t) con las constantes numéricas, según el régimen."""
    if d["tipo"] == "sub":
        a, wd = d["alpha"], d["wd"]
        pre = rf"e^{{-{fm(a)}t}}" if a > 0 else ""
        return (rf"x_h(t)={pre}\left[{fm(d['C1'])}\cos({fm(wd)}\,t)"
                rf" {sg(d['C2'])}\sin({fm(wd)}\,t)\right]")
    if d["tipo"] == "crit":
        return rf"x_h(t)=e^{{-{fm(d['alpha'])}t}}\left({fm(d['C1'])} {sg(d['C2'])}\,t\right)"
    return (rf"x_h(t)={fm(d['C1'])}\,e^{{{fm(d['r1'])}t}} {sg(d['C2'])}\,e^{{{fm(d['r2'])}t}}")


def _pasos_homogenea(m, c, k, x0, v0, etiqueta_ic=""):
    """Bloques: ecuación característica, raíces, solución general y constantes."""
    d = lin.constantes(m, c, k, x0, v0)
    b = [L(rf"m r^2 + c r + k = 0 \;\Rightarrow\; {fm(m)}\,r^2 + {fm(c)}\,r + {fm(k)} = 0"),
         L(rf"\Delta=c^2-4mk={fm(c)}^2-4({fm(m)})({fm(k)})={fm(c*c-4*m*k)}")]
    a = d["alpha"]
    if d["tipo"] == "sub":
        b += [T("**Raíces complejas conjugadas** (subamortiguado)."),
              L(rf"r=-\alpha\pm i\omega_d,\quad \alpha=\frac{{c}}{{2m}}={fm(a)},\quad "
                rf"\omega_d=\omega_0\sqrt{{1-\zeta^2}}={fm(d['wd'])}"),
              L(r"x_h(t)=e^{-\alpha t}\left[C_1\cos(\omega_d t)+C_2\sin(\omega_d t)\right]"),
              L(rf"C_1={fm(x0)},\qquad C_2=\frac{{v_0+\alpha x_0}}{{\omega_d}}={fm(d['C2'])}")]
    elif d["tipo"] == "crit":
        b += [T("**Raíz real doble** (críticamente amortiguado)."),
              L(rf"r=-\alpha=-{fm(a)}"),
              L(r"x_h(t)=(C_1+C_2t)\,e^{-\alpha t}"),
              L(rf"C_1={fm(x0)},\qquad C_2=v_0+\alpha x_0={fm(d['C2'])}")]
    else:
        b += [T("**Dos raíces reales distintas** (sobreamortiguado)."),
              L(rf"r_{{1,2}}=-\alpha\pm\omega_0\sqrt{{\zeta^2-1}}\;\Rightarrow\; "
                rf"r_1={fm(d['r1'])},\; r_2={fm(d['r2'])}"),
              L(r"x_h(t)=C_1e^{r_1t}+C_2e^{r_2t}"),
              L(rf"C_2=\frac{{v_0-r_1x_0}}{{r_2-r_1}}={fm(d['C2'])},\qquad C_1=x_0-C_2={fm(d['C1'])}")]
    b.append(L(_latex_hom(d)))
    return b


class MasaResorte(Sistema):
    estados = ("x", "v")
    amortiguado = False
    forzado = False

    # ---- helpers -------------------------------------------------------
    def _q(self, p):
        return (p["m"], p.get("c", 0.0) if self.amortiguado or self.forzado else 0.0, p["k"])

    def _F(self, p):
        return (p["F0"], p["W"]) if self.forzado else (0.0, 0.0)

    def rhs(self, p):
        m, c, k = self._q(p)
        F0, W = self._F(p)

        def f(t, z):
            return np.array([z[1], (F0 * math.cos(W * t) - c * z[1] - k * z[0]) / m])
        return f

    def z0(self, p):
        return np.array([p["x0"], p["v0"]], float)

    def analitica(self, p, t):
        m, c, k = self._q(p)
        if self.forzado:
            r = lin.forzada(m, c, k, p["F0"], p["W"], p["x0"], p["v0"], t)
            return np.column_stack([r["x"], r["v"]])
        x, v = lin.homogenea(m, c, k, p["x0"], p["v0"], t)
        return np.column_stack([x, v])

    def _energia(self, p, Z):
        return 0.5 * p["m"] * Z[:, 1] ** 2 + 0.5 * p["k"] * Z[:, 0] ** 2

    def _acel(self, p, t, Z):
        m, c, k = self._q(p)
        F0, W = self._F(p)
        return (F0 * np.cos(W * t) - c * Z[:, 1] - k * Z[:, 0]) / m

    def figuras(self, p, t, Z, Za):
        a = self._acel(p, t, Z)
        xa = Za[:, 0] if Za is not None else None
        va = Za[:, 1] if Za is not None else None
        E = self._energia(p, Z)
        en = {"Energía mecánica numérica": E}
        if Za is not None:
            en["Energía mecánica analítica"] = self._energia(p, Za)
        figs = figuras_oscilador(t, Z[:, 0], Z[:, 1], a, en, NOMBRES, xa, va)
        # La energía en el sistema forzado no se conserva (la fuerza hace trabajo).
        return figs


class MasaResorteSimple(MasaResorte):
    clave = "simple"
    nombre = "Masa-resorte simple"
    descripcion = ("Oscilador armónico simple: una masa unida a un resorte ideal, sin "
                   "fricción ni fuerzas externas. Modelo: $m\\ddot{x}+kx=0$.")
    campos = (C_M, C_K, C_X0, C_V0)
    tiempo_defecto, paso_defecto = 10.0, 0.01

    def derivadas(self, p):
        m, k, x0, v0 = p["m"], p["k"], p["x0"], p["v0"]
        w0 = math.sqrt(k / m)
        A = math.hypot(x0, v0 / w0)
        return [("Frecuencia angular natural ω₀", w0, "rad/s"),
                ("Frecuencia natural f₀", w0 / (2 * math.pi), "Hz"),
                ("Período T", 2 * math.pi / w0, "s"),
                ("Amplitud A", A, "m"),
                ("Energía mecánica E", 0.5 * k * A * A, "J")]

    def desarrollo(self, p, cfg):
        m, k, x0, v0 = p["m"], p["k"], p["x0"], p["v0"]
        w0 = math.sqrt(k / m)
        A, phi = math.hypot(x0, v0 / w0), math.atan2(v0 / w0, x0)
        return [
            H("1. Ecuación diferencial original"),
            L(r"m\,\ddot{x}+k\,x=0"),
            T("Ley de Hooke + segunda ley de Newton: $x$ es el desplazamiento desde el equilibrio, "
              "$m$ la masa y $k$ la constante elástica."),
            H("2. Sustitución de parámetros"),
            L(rf"{fm(m)}\,\ddot{{x}}+{fm(k)}\,x=0"),
            H("3. Ecuación característica"),
            L(rf"x=e^{{rt}}\;\Rightarrow\; m r^2+k=0\;\Rightarrow\; {fm(m)}\,r^2+{fm(k)}=0"),
            H("4. Raíces características"),
            L(rf"r=\pm i\sqrt{{\tfrac{{k}}{{m}}}}=\pm i\sqrt{{\tfrac{{{fm(k)}}}{{{fm(m)}}}}}=\pm {fm(w0)}\,i"),
            H("5. Solución general"),
            L(r"x(t)=C_1\cos(\omega_0 t)+C_2\sin(\omega_0 t),\qquad \omega_0=\sqrt{k/m}"),
            H("6. Aplicación de las condiciones iniciales"),
            L(rf"x(0)=C_1={fm(x0)},\qquad \dot{{x}}(0)=C_2\,\omega_0={fm(v0)}"
              rf"\;\Rightarrow\; C_2=\frac{{{fm(v0)}}}{{{fm(w0)}}}={fm(v0/w0)}"),
            H("7. Solución particular"),
            L(rf"x(t)={fm(x0)}\cos({fm(w0)}\,t) {sg(v0/w0)}\sin({fm(w0)}\,t)"
              rf"={fm(A)}\cos({fm(w0)}\,t-{fm(phi)})"),
            L(rf"v(t)=\dot{{x}}(t)=-{fm(A*w0)}\sin({fm(w0)}\,t-{fm(phi)})"),
            H("8. Frecuencia y período"),
            L(rf"\omega_0={fm(w0)}\ \text{{rad/s}},\quad f_0=\frac{{\omega_0}}{{2\pi}}={fm(w0/(2*math.pi))}\ \text{{Hz}},"
              rf"\quad T=\frac{{2\pi}}{{\omega_0}}={fm(2*math.pi/w0)}\ \text{{s}}"),
            H("9. Energía"),
            L(rf"E=\tfrac12 m v^2+\tfrac12 k x^2=\tfrac12 k A^2={fm(0.5*k*A*A)}\ \text{{J}}\ (\text{{constante}})"),
        ]

    def interpretacion(self, p, t, Z):
        w0 = math.sqrt(p["k"] / p["m"])
        E = self._energia(p, Z)
        deriva = (E.max() - E.min()) / E[0] if E[0] > 0 else 0.0
        return [f"El sistema oscila indefinidamente con período {fm(2*math.pi/w0)} s: no hay "
                "disipación, así que la amplitud no decae.",
                "Una masa mayor o un resorte más blando **alargan** el período "
                "($T=2\\pi\\sqrt{m/k}$); no depende de las condiciones iniciales.",
                f"La energía mecánica oscila numéricamente un {deriva:.2e} (relativo) alrededor "
                "de un valor constante: la variación es error del método (Euler la hace crecer, "
                "RK4 casi no la altera)."]


class MasaResorteAmortiguado(MasaResorte):
    clave = "amortiguado"
    nombre = "Masa-resorte amortiguado"
    descripcion = ("Oscilador con fricción viscosa lineal: $m\\ddot{x}+c\\dot{x}+kx=0$. Según "
                   "$\\zeta$ el movimiento es subamortiguado, crítico o sobreamortiguado.")
    campos = (C_M, C_K, C_C, C_X0, C_V0)
    amortiguado = True
    tiempo_defecto, paso_defecto = 15.0, 0.01

    def derivadas(self, p):
        m, c, k = p["m"], p["c"], p["k"]
        w0, z = lin.omega0(m, k), lin.zeta(m, c, k)
        d = lin.constantes(m, c, k, p["x0"], p["v0"])
        out = [("Régimen", lin.regimen(m, c, k), ""),
               ("Frecuencia natural ω₀", w0, "rad/s"),
               ("Amortiguamiento relativo ζ", z, ""),
               ("Amortiguamiento crítico c_c = 2√(km)", 2 * math.sqrt(k * m), "N·s/m"),
               ("Parámetro α = c/2m", d["alpha"], "1/s")]
        if d["tipo"] == "sub":
            out += [("Frecuencia amortiguada ω_d", d["wd"], "rad/s"),
                    ("Período amortiguado T_d", 2 * math.pi / d["wd"], "s"),
                    ("Decremento logarítmico δ", 2 * math.pi * z / math.sqrt(1 - z * z), ""),
                    ("Raíces r₁,₂", f"{-d['alpha']:.6g} ± {d['wd']:.6g} i", "1/s")]
        elif d["tipo"] == "crit":
            out.append(("Raíz doble r", -d["alpha"], "1/s"))
        else:
            out += [("Raíz r₁", d["r1"], "1/s"), ("Raíz r₂", d["r2"], "1/s")]
        out.append(("Tiempo de asentamiento aprox. 4/α", 4 / d["alpha"], "s"))
        return out

    def desarrollo(self, p, cfg):
        m, c, k, x0, v0 = p["m"], p["c"], p["k"], p["x0"], p["v0"]
        w0, z = lin.omega0(m, k), lin.zeta(m, c, k)
        return ([H("1. Ecuación diferencial original"), L(r"m\,\ddot{x}+c\,\dot{x}+k\,x=0"),
                 H("2. Sustitución de parámetros"),
                 L(rf"{fm(m)}\,\ddot{{x}}+{fm(c)}\,\dot{{x}}+{fm(k)}\,x=0"),
                 L(rf"\omega_0=\sqrt{{k/m}}={fm(w0)},\qquad "
                   rf"\zeta=\frac{{c}}{{2\sqrt{{km}}}}=\frac{{{fm(c)}}}{{2\sqrt{{{fm(k)}\cdot{fm(m)}}}}}={fm(z)}"),
                 T(f"Como $\\zeta={fm(z)}$, el sistema es **{lin.regimen(m, c, k)}**."),
                 H("3-5. Ecuación característica, raíces y solución general")]
                + _pasos_homogenea(m, c, k, x0, v0)
                + [H("6. Condiciones iniciales"), L(rf"x(0)={fm(x0)},\qquad \dot{{x}}(0)={fm(v0)}"),
                   H("7. Diferencias entre regímenes"),
                   T("- **Subamortiguado** ($\\zeta<1$): raíces complejas; oscila con amplitud que decae como $e^{-\\alpha t}$.\n"
                     "- **Crítico** ($\\zeta=1$): raíz doble; regresa al equilibrio lo más rápido posible sin oscilar.\n"
                     "- **Sobreamortiguado** ($\\zeta>1$): dos raíces reales negativas; no oscila y el retorno es más lento que en el crítico.")])

    def interpretacion(self, p, t, Z):
        m, c, k = p["m"], p["c"], p["k"]
        E = self._energia(p, Z)
        reg = lin.regimen(m, c, k)
        txt = {"subamortiguado": "La masa oscila alrededor del equilibrio con amplitud decreciente.",
               "críticamente amortiguado": "La masa vuelve al equilibrio sin oscilar y con el menor tiempo posible.",
               "sobreamortiguado": "La masa vuelve lentamente al equilibrio sin oscilar."}[reg]
        return [f"Régimen **{reg}** (ζ = {fm(lin.zeta(m, c, k))}). {txt}",
                f"La energía mecánica pasa de {fm(E[0])} J a {fm(E[-1])} J: la fricción la disipa "
                "($\\dot E=-c\\,v^2\\le 0$)."]


class MasaResorteForzado(MasaResorte):
    clave = "forzado"
    nombre = "Masa-resorte forzado"
    descripcion = ("Oscilador amortiguado con fuerza armónica externa: "
                   "$m\\ddot{x}+c\\dot{x}+kx=F_0\\cos(\\omega t)$. Muestra transitorio, estado "
                   "estacionario y resonancia.")
    campos = (C_M, C_K, Campo("c", "Coeficiente de amortiguamiento c", "N·s/m", 0.4, "nonneg",
                              "0 = sin amortiguamiento"),
              Campo("F0", "Amplitud de la fuerza F₀", "N", 1.0, "any"),
              Campo("W", "Frecuencia de excitación ω", "rad/s", 1.5, "pos"),
              C_X0, Campo("v0", "Velocidad inicial v₀", "m/s", 0.0, "any"))
    amortiguado = True
    forzado = True
    tiempo_defecto, paso_defecto = 40.0, 0.01

    def derivadas(self, p):
        m, c, k, F0, W = p["m"], p["c"], p["k"], p["F0"], p["W"]
        w0 = lin.omega0(m, k)
        pura = lin.es_resonancia_pura(m, c, k, W)
        out = [("Frecuencia natural ω₀", w0, "rad/s"),
               ("Frecuencia de excitación ω", W, "rad/s"),
               ("Frecuencia de excitación f", W / (2 * math.pi), "Hz"),
               ("Cociente ω/ω₀", W / w0, ""),
               ("Amortiguamiento relativo ζ", lin.zeta(m, c, k), ""),
               ("Régimen del transitorio", lin.regimen(m, c, k), "")]
        if pura:
            out.append(("Estado estacionario", "no acotado (resonancia pura): A(t)=F₀t/(2mω₀)", ""))
        else:
            A, ph = lin.amplitud(m, c, k, F0, W), lin.fase(m, c, k, W)
            out += [("Amplitud estacionaria A", A, "m"),
                    ("Desfase φ", ph, "rad"), ("Desfase φ", math.degrees(ph), "°"),
                    ("Factor de amplificación A/(F₀/k)", A * k / F0 if F0 else 0.0, "")]
        Wr, Mr = lin.frecuencia_pico(m, c, k)
        if Wr is not None and c > 0:
            out += [("Frecuencia de resonancia de amplitud ω_r", Wr, "rad/s"),
                    ("Amplitud máxima A(ω_r)", abs(F0) * Mr, "m")]
        elif c > 0:
            out.append(("Resonancia de amplitud", "no hay pico (c² ≥ 2km)", ""))
        if abs(W - w0) / w0 < 0.05:
            out.append(("⚠ Aviso", "ω está a menos del 5 % de ω₀: zona de resonancia", ""))
        return out

    def figuras(self, p, t, Z, Za):
        m, c, k, F0, W = p["m"], p["c"], p["k"], p["F0"], p["W"]
        r = lin.forzada(m, c, k, F0, W, p["x0"], p["v0"], t)
        figs = [figura("Fuerza externa", "t (s)", "F (N)", [serie("F₀cos(ωt)", t, F0 * np.cos(W * t))]),
                figura("Posición frente al tiempo", "t (s)", "x (m)",
                       [serie("Numérica", t, Z[:, 0]), serie("Analítica", t, r["x"], "dash")]),
                figura("Velocidad frente al tiempo", "t (s)", "v (m/s)",
                       [serie("Numérica", t, Z[:, 1]), serie("Analítica", t, r["v"], "dash")]),
                figura("Respuesta transitoria y estacionaria", "t (s)", "x (m)",
                       [serie("Transitoria", t, r["x_tr"]), serie("Estacionaria", t, r["x_ss"]),
                        serie("Total", t, r["x"], "dash")])]
        w0 = lin.omega0(m, k)
        Ws = np.linspace(0.01 * w0, max(2.5 * w0, 1.2 * W), 600)
        den = np.sqrt((k - m * Ws ** 2) ** 2 + (c * Ws) ** 2)
        with np.errstate(divide="ignore"):
            A = np.where(den > 0, abs(F0) / den, np.nan)
        Aw = abs(F0) / np.sqrt((k - m * W * W) ** 2 + (c * W) ** 2) if not lin.es_resonancia_pura(m, c, k, W) else np.nan
        s = [serie("A(ω)", Ws, A), serie("ω excitación", [W], [Aw], modo="markers"),
             serie("ω₀ (frecuencia natural)", [w0], [0.0], modo="markers")]
        figs.append(figura("Amplitud estacionaria vs frecuencia de excitación", "ω (rad/s)", "A (m)", s))
        figs.append(figura("Energía mecánica del oscilador", "t (s)", "E (J)",
                           [serie("E", t, self._energia(p, Z))]))
        return figs

    def desarrollo(self, p, cfg):
        m, c, k, F0, W, x0, v0 = (p[q] for q in ("m", "c", "k", "F0", "W", "x0", "v0"))
        w0 = lin.omega0(m, k)
        b = [H("1. Ecuación diferencial original"), L(r"m\ddot{x}+c\dot{x}+kx=F_0\cos(\omega t)"),
             H("2. Sustitución de parámetros"),
             L(rf"{fm(m)}\ddot{{x}}+{fm(c)}\dot{{x}}+{fm(k)}x={fm(F0)}\cos({fm(W)}\,t)"),
             L(rf"\omega_0={fm(w0)}\ \text{{rad/s}},\quad \omega={fm(W)}\ \text{{rad/s}},\quad "
               rf"\omega/\omega_0={fm(W/w0)}"),
             T("La solución es $x=x_h+x_p$: la **homogénea** (transitorio, decae si $c>0$) más una "
               "solución **particular** (estado estacionario, a la frecuencia de la fuerza).")]
        if lin.es_resonancia_pura(m, c, k, W):
            b += [H("3. Resonancia pura ($c=0$, $\\omega=\\omega_0$)"),
                  T("El método de coeficientes indeterminados con $x_p=At\\sin\\omega_0 t$ da:"),
                  L(rf"x_p(t)=\frac{{F_0}}{{2m\omega_0}}\,t\sin(\omega_0t)=\frac{{{fm(F0)}}}{{2({fm(m)})({fm(w0)})}}\,t\sin({fm(w0)}\,t)"),
                  T("La amplitud crece linealmente sin cota: sin amortiguamiento la energía "
                    "aportada por la fuerza se acumula indefinidamente.")]
            xh0, vh0 = x0, v0
        else:
            A, ph = lin.amplitud(m, c, k, F0, W), lin.fase(m, c, k, W)
            b += [H("3. Solución particular (estado estacionario)"),
                  L(r"x_p(t)=A\cos(\omega t-\varphi),\quad A=\frac{F_0}{\sqrt{(k-m\omega^2)^2+(c\omega)^2}},"
                    r"\quad \varphi=\operatorname{atan2}(c\omega,\;k-m\omega^2)"),
                  L(rf"A=\frac{{{fm(F0)}}}{{\sqrt{{({fm(k-m*W*W)})^2+({fm(c*W)})^2}}}}={fm(A)}\ \text{{m}},"
                    rf"\qquad \varphi={fm(ph)}\ \text{{rad}}={fm(math.degrees(ph))}^\circ"),
                  L(rf"x_p(t)={fm(A)}\cos({fm(W)}\,t-{fm(ph)})")]
            xh0, vh0 = x0 - A * math.cos(ph), v0 - A * W * math.sin(ph)
        b += [H("4. Solución homogénea (transitorio)")]
        if c == 0:
            b += [T("Sin amortiguamiento el 'transitorio' no decae: oscila a $\\omega_0$ (pulsaciones si $\\omega\\approx\\omega_0$).")]
        b += _pasos_homogenea(m, c, k, xh0, vh0)
        b += [T("Las constantes se calculan con $x_h(0)=x_0-x_p(0)$ y $\\dot x_h(0)=v_0-\\dot x_p(0)$."),
              H("5. Solución completa"), L(r"x(t)=x_h(t)+x_p(t)"),
              H("6. Resonancia e influencia del amortiguamiento"),
              L(r"A(\omega)=\frac{F_0}{\sqrt{(k-m\omega^2)^2+c^2\omega^2}}"),
              T("El denominador es mínimo (amplitud máxima) cerca de $\\omega_r=\\sqrt{k/m-c^2/(2m^2)}$, "
                "con $A_{max}=F_0/(c\\,\\omega_d)$. Al aumentar $c$ el pico **baja y se desplaza** hacia "
                "frecuencias menores; si $c^2\\ge 2km$ desaparece el pico.")]
        return b

    def interpretacion(self, p, t, Z):
        m, c, k, F0, W = p["m"], p["c"], p["k"], p["F0"], p["W"]
        w0 = lin.omega0(m, k)
        out = [f"La frecuencia de excitación es {fm(W/w0)}·ω₀."]
        if lin.es_resonancia_pura(m, c, k, W):
            out.append("**Resonancia pura**: la amplitud crece sin límite (linealmente en el tiempo).")
        else:
            A = lin.amplitud(m, c, k, F0, W)
            out.append(f"En régimen estacionario la masa oscila con amplitud {fm(A)} m, "
                       f"{fm(A*k/abs(F0)) if F0 else 0}× la deformación estática F₀/k.")
            out.append("Con ω < ω₀ la masa va en fase con la fuerza (φ<90°); con ω > ω₀ va en contrafase (φ>90°).")
        if c > 0:
            out.append(f"El transitorio decae con e^(−{fm(c/(2*m))}·t); tras ≈{fm(4*2*m/c)} s domina el estacionario.")
        return out
