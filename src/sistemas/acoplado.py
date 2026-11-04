"""Dos masas entre paredes unidas por tres resortes: pared–k₁–m₁–k₂–m₂–k₃–pared."""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh

from ..utilidades.formato import H, L, T, fm
from ..visualizaciones.figuras import figura, serie
from .base import Campo, Sistema


def matrices(p):
    M = np.diag([p["m1"], p["m2"]])
    K = np.array([[p["k1"] + p["k2"], -p["k2"]], [-p["k2"], p["k2"] + p["k3"]]])
    return M, K


def modos(p):
    """Frecuencias (rad/s) y modos Φ con Φᵀ M Φ = I (resuelve K φ = ω² M φ)."""
    M, K = matrices(p)
    lam, Phi = eigh(K, M)
    return np.sqrt(lam), Phi


class DosMasasAcopladas(Sistema):
    clave = "acoplado"
    nombre = "Dos masas acopladas"
    descripcion = ("Dos masas entre dos paredes fijas, unidas por **tres resortes** "
                   "(pared–$k_1$–$m_1$–$k_2$–$m_2$–$k_3$–pared), movimiento horizontal sin fricción. "
                   "$x_1, x_2$ son desplazamientos desde el equilibrio.")
    campos = (Campo("m1", "Masa 1  m₁", "kg", 1.0), Campo("m2", "Masa 2  m₂", "kg", 1.0),
              Campo("k1", "Resorte 1 (pared–m₁)  k₁", "N/m", 4.0),
              Campo("k2", "Resorte 2 (m₁–m₂)  k₂", "N/m", 2.0),
              Campo("k3", "Resorte 3 (m₂–pared)  k₃", "N/m", 4.0),
              Campo("x1", "Posición inicial m₁  x₁(0)", "m", 1.0, "any"),
              Campo("x2", "Posición inicial m₂  x₂(0)", "m", 0.0, "any"),
              Campo("v1", "Velocidad inicial m₁  v₁(0)", "m/s", 0.0, "any"),
              Campo("v2", "Velocidad inicial m₂  v₂(0)", "m/s", 0.0, "any"))
    estados = ("x1", "x2", "v1", "v2")
    tiempo_defecto, paso_defecto = 20.0, 0.01

    def rhs(self, p):
        M, K = matrices(p)
        Mi = np.linalg.inv(M)
        A = -Mi @ K

        def f(t, z):
            return np.concatenate([z[2:], A @ z[:2]])
        return f

    def z0(self, p):
        return np.array([p["x1"], p["x2"], p["v1"], p["v2"]], float)

    def analitica(self, p, t):
        M, _ = matrices(p)
        w, Phi = modos(p)
        x0, v0 = np.array([p["x1"], p["x2"]]), np.array([p["v1"], p["v2"]])
        q0, qd0 = Phi.T @ M @ x0, Phi.T @ M @ v0
        t = np.asarray(t, float)[:, None]
        q = q0 * np.cos(w * t) + (qd0 / w) * np.sin(w * t)
        qd = -q0 * w * np.sin(w * t) + qd0 * np.cos(w * t)
        return np.hstack([q @ Phi.T, qd @ Phi.T])

    def _energias(self, p, Z):
        T_ = 0.5 * p["m1"] * Z[:, 2] ** 2 + 0.5 * p["m2"] * Z[:, 3] ** 2
        U = (0.5 * p["k1"] * Z[:, 0] ** 2 + 0.5 * p["k2"] * (Z[:, 1] - Z[:, 0]) ** 2
             + 0.5 * p["k3"] * Z[:, 1] ** 2)
        return T_, U

    def derivadas(self, p):
        w, Phi = modos(p)
        out = []
        for i in range(2):
            out += [(f"Frecuencia natural ω{i+1}", w[i], "rad/s"),
                    (f"Frecuencia f{i+1}", w[i] / (2 * math.pi), "Hz"),
                    (f"Período T{i+1}", 2 * math.pi / w[i], "s"),
                    (f"Modo {i+1} (normalizado Φᵀ M Φ = I)", f"[{Phi[0,i]:.5g}, {Phi[1,i]:.5g}]", "")]
        return out

    def figuras(self, p, t, Z, Za):
        Ta, U = self._energias(p, Z)
        figs = [figura("Posición de ambas masas", "t (s)", "x (m)",
                       [serie("x₁ numérica", t, Z[:, 0]), serie("x₂ numérica", t, Z[:, 1])]
                       + ([serie("x₁ analítica", t, Za[:, 0], "dash"), serie("x₂ analítica", t, Za[:, 1], "dash")]
                          if Za is not None else [])),
                figura("Velocidad de ambas masas", "t (s)", "v (m/s)",
                       [serie("v₁", t, Z[:, 2]), serie("v₂", t, Z[:, 3])])]
        M, _ = matrices(p)
        w, Phi = modos(p)
        q = (Phi.T @ M @ Z[:, :2].T).T
        figs.append(figura("Coordenadas modales qᵢ(t) (cada modo oscila a su frecuencia)", "t (s)", "q (m·√kg)",
                           [serie(f"Modo {i+1} (ω={w[i]:.4g} rad/s)", t, q[:, i]) for i in range(2)]))
        figs.append(figura("Forma de los modos normales", "masa (1 o 2)", "amplitud relativa",
                           [serie(f"Modo {i+1} (ω={w[i]:.4g} rad/s)", [1, 2], Phi[:, i], modo="lines+markers")
                            for i in range(2)]))
        figs.append(figura("Retrato de fase", "x (m)", "v (m/s)",
                           [serie("Masa 1", Z[:, 0], Z[:, 2]), serie("Masa 2", Z[:, 1], Z[:, 3])]))
        figs.append(figura("Energía", "t (s)", "E (J)",
                           [serie("Cinética", t, Ta), serie("Potencial", t, U), serie("Total", t, Ta + U)]))
        return figs

    def desarrollo(self, p, cfg):
        m1, m2, k1, k2, k3 = (p[q] for q in ("m1", "m2", "k1", "k2", "k3"))
        M, K = matrices(p)
        w, Phi = modos(p)
        mat = lambda A: r"\begin{pmatrix}" + r"\\".join("&".join(fm(v) for v in r) for r in A) + r"\end{pmatrix}"
        a, d, c_ = k1 + k2, k2 + k3, k2
        return [
            H("1. Ecuaciones diferenciales (segunda ley de Newton)"),
            L(r"m_1\ddot x_1=-(k_1+k_2)x_1+k_2x_2,\qquad m_2\ddot x_2=k_2x_1-(k_2+k_3)x_2"),
            L(rf"{fm(m1)}\ddot x_1=-{fm(a)}x_1+{fm(k2)}x_2,\qquad {fm(m2)}\ddot x_2={fm(k2)}x_1-{fm(d)}x_2"),
            H("2. Forma matricial  M ẍ + K x = 0"),
            L(rf"M={mat(M)},\qquad K={mat(K)}"),
            H("3. Frecuencias naturales"),
            T("Se busca $\\mathbf{x}=\\boldsymbol\\phi\\cos(\\omega t-\\delta)$; resulta el problema de valores propios generalizado "
              "$(K-\\omega^2M)\\boldsymbol\\phi=0$, que tiene solución no trivial si"),
            L(r"\det(K-\omega^2M)=0"),
            L(rf"({fm(a)}-{fm(m1)}\omega^2)({fm(d)}-{fm(m2)}\omega^2)-{fm(c_*c_)}=0"),
            L(rf"\omega_1={fm(w[0])}\ \text{{rad/s}},\qquad \omega_2={fm(w[1])}\ \text{{rad/s}}"),
            H("4. Modos normales"),
            L(rf"\boldsymbol\phi_1={mat(Phi[:, [0]])},\qquad \boldsymbol\phi_2={mat(Phi[:, [1]])}"),
            T("Normalizados con $\\Phi^TM\\Phi=I$. En el modo de menor frecuencia las masas suelen moverse "
              "en fase; en el de mayor frecuencia, en oposición de fase (el resorte central se deforma más)."),
            H("5. Solución"),
            L(r"\mathbf{x}(t)=\sum_{i=1}^{2}\boldsymbol\phi_i\left[q_i(0)\cos\omega_it+\frac{\dot q_i(0)}{\omega_i}\sin\omega_it\right],\quad q(0)=\Phi^TM\mathbf{x}(0)"),
            H("6. Energía total"),
            L(r"E=\tfrac12m_1\dot x_1^2+\tfrac12m_2\dot x_2^2+\tfrac12k_1x_1^2+\tfrac12k_2(x_2-x_1)^2+\tfrac12k_3x_2^2\ \text{(constante)}"),
        ]

    def interpretacion(self, p, t, Z):
        w, Phi = modos(p)
        Ta, U = self._energias(p, Z)
        E = Ta + U
        return [f"Hay dos frecuencias propias: ω₁ = {fm(w[0])} y ω₂ = {fm(w[1])} rad/s. El movimiento general es la "
                "superposición de ambos modos, por lo que cada masa muestra pulsaciones e intercambia energía con la otra.",
                "Si las condiciones iniciales coinciden con un solo modo, el sistema oscila armónicamente con una sola frecuencia.",
                f"Energía total: {fm(E[0])} J al inicio y {fm(E[-1])} J al final (conservativo salvo error numérico)."]
