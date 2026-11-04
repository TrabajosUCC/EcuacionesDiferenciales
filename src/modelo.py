"""
Modelo matemático del autoescalado de un servicio en la nube.

El sistema se describe con dos variables de desviación respecto al punto de
equilibrio:

    x(t) : peticiones pendientes en la cola                  [peticiones]
    y(t) : capacidad adicional (réplicas activas)             [réplicas]
    d(t) : perturbación de carga entrante                     [peticiones/s]

Las ecuaciones del modelo son

    x' = -mu*y + d(t)
    tau*y' = -y + K*x
    x(0) = y(0) = 0

donde

    mu  : capacidad de una réplica             [peticiones/(s*réplica)]
    K   : ganancia del autoescalador          [réplicas/petición]
    tau : retardo de aprovisionamiento        [s]

Este módulo centraliza los parámetros y las operaciones básicas del
modelo (entradas, campo vectorial, valores propios y chequeo dimensional).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Parámetros del modelo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Parametros:
    """Parámetros físicos y numéricos del caso de estudio.

    Se usa una dataclass inmutable para evitar que un módulo modifique por
    accidente los parámetros que usan los demás. Todas las unidades están
    indicadas en los comentarios.

    Atributos
    ---------
    mu : float
        Capacidad de una réplica [peticiones/(s*réplica)].
    K : float
        Ganancia del autoescalador [réplicas/petición].
    tau : float
        Retardo de aprovisionamiento (arranque de contenedores) [s].
    d0 : float
        Amplitud de la perturbación de carga [peticiones/s].
    T : float
        Horizonte de simulación [s].
    x0, y0 : float
        Condiciones iniciales [peticiones] y [réplicas].
    """

    mu: float = 10.0       # [peticiones/(s*réplica)]
    K: float = 0.005       # [réplicas/petición]
    tau: float = 20.0      # [s]
    d0: float = 20.0       # [peticiones/s]
    T: float = 600.0       # [s]
    x0: float = 0.0        # [peticiones]
    y0: float = 0.0        # [réplicas]

    def __post_init__(self) -> None:
        """Valida que los parámetros tengan sentido físico."""
        if self.mu <= 0 or self.K <= 0 or self.tau <= 0:
            raise ValueError("mu, K y tau deben ser estrictamente positivos.")
        if self.d0 < 0:
            raise ValueError("d0 no puede ser negativo en este modelo lineal.")
        if self.T <= 0:
            raise ValueError("El horizonte T debe ser positivo.")

    # ------------------------------------------------------------------
    # Cantidades derivadas del modelo
    # ------------------------------------------------------------------
    @property
    def wn(self) -> float:
        """Frecuencia natural no amortiguada [rad/s].

        Se obtiene de la EDO de segundo orden:
            x'' + (1/tau)x' + (mu*K/tau)x = ...
        Por lo tanto wn = sqrt(mu*K/tau).
        """
        return float(np.sqrt(self.mu * self.K / self.tau))

    @property
    def zeta(self) -> float:
        """Factor de amortiguamiento (adimensional).

        Para la EDO x'' + 2*zeta*wn*x' + wn^2*x = ...
        se cumple 2*zeta*wn = 1/tau, de donde
        zeta = 1 / (2*sqrt(mu*K*tau)).
        """
        return float(1.0 / (2.0 * np.sqrt(self.mu * self.K * self.tau)))

    @property
    def discriminante(self) -> float:
        """Discriminante normalizado de la ecuación característica.

        tau*s^2 + s + mu*K = 0 tiene discriminante
        Delta = 1 - 4*mu*K*tau.  El signo de Delta define el régimen:
        negativo -> subamortiguado, cero -> crítico, positivo -> sobreamortiguado.
        """
        return float(1.0 - 4.0 * self.mu * self.K * self.tau)

    @property
    def regimen(self) -> str:
        """Régimen de amortiguamiento según zeta."""
        # Se usa una tolerancia numérica para el caso crítico exacto.
        if abs(self.zeta - 1.0) < 1.0e-9:
            return "crítico"
        if self.zeta < 1.0:
            return "subamortiguado"
        return "sobreamortiguado"

    @property
    def x_ss(self) -> float:
        """Cola estacionaria ante un escalón d0 [peticiones]."""
        return float(self.d0 / (self.mu * self.K))

    @property
    def y_ss(self) -> float:
        """Capacidad adicional estacionaria ante un escalón d0 [réplicas]."""
        return float(self.d0 / self.mu)

    @property
    def matriz_A(self) -> np.ndarray:
        """Matriz del sistema lineal en variables de estado [x, y].

        A = [[0, -mu], [K/tau, -1/tau]]
        """
        return np.array(
            [[0.0, -self.mu], [self.K / self.tau, -1.0 / self.tau]],
            dtype=float,
        )

    @property
    def valores_propios(self) -> np.ndarray:
        """Valores propios de A, es decir, raíces de tau*s^2+s+mu*K=0."""
        return np.linalg.eigvals(self.matriz_A)

    def raices_caracteristicas(self) -> Tuple[complex, complex]:
        """Raíces ordenadas de la ecuación característica.

        Se ordenan de menor a mayor parte real para que las fórmulas del
        régimen sobreamortiguado sean reproducibles.
        """
        raices = np.roots([self.tau, 1.0, self.mu * self.K])
        raices = np.sort_complex(raices)
        return complex(raices[0]), complex(raices[1])

    def con_parametros(self, **cambios: float) -> "Parametros":
        """Devuelve una copia con algunos parámetros modificados.

        Es útil para los barridos sin mutar el caso base.
        """
        datos = {
            "mu": self.mu,
            "K": self.K,
            "tau": self.tau,
            "d0": self.d0,
            "T": self.T,
            "x0": self.x0,
            "y0": self.y0,
        }
        datos.update(cambios)
        return Parametros(**datos)


# Parámetro base exigido por el enunciado. Se deja como constante para que
# cualquier módulo pueda importarlo sin reconstruirlo.
BASE = Parametros(mu=10.0, K=0.005, tau=20.0, d0=20.0, T=600.0)


# ---------------------------------------------------------------------------
# 2. Entradas de carga
# ---------------------------------------------------------------------------

def entrada_escalon(t: np.ndarray | float, d0: float) -> np.ndarray | float:
    """Carga tipo escalón: d(t) = d0 para t >= 0.

    Se implementa con np.where para aceptar tanto escalares como arreglos.
    """
    return np.where(np.asarray(t) >= 0.0, d0, 0.0)


def entrada_sinusoidal(
    t: np.ndarray | float, d0: float, w: float
) -> np.ndarray | float:
    """Carga sinusoidal: d(t) = d0*sin(w*t) [peticiones/s]."""
    return d0 * np.sin(w * np.asarray(t))


def entrada_derivada_sinusoidal(
    t: np.ndarray | float, d0: float, w: float
) -> np.ndarray | float:
    """Derivada de la carga sinusoidal: d'(t) = d0*w*cos(w*t).

    Se usa en la EDO de segundo orden y en la verificación simbólica.
    """
    return d0 * w * np.cos(w * np.asarray(t))


# ---------------------------------------------------------------------------
# 3. Campo vectorial del sistema de primer orden
# ---------------------------------------------------------------------------

def campo_vectorial(
    t: float,
    estado: np.ndarray,
    p: Parametros,
    entrada: Callable[[float], float],
) -> np.ndarray:
    """Calcula f(t, [x, y]) para el sistema x'=-mu*y+d(t), tau*y'=-y+K*x.

    Parámetros
    ----------
    t : float
        Tiempo actual [s].
    estado : array_like
        Vector [x, y] con x en peticiones e y en réplicas.
    p : Parametros
        Parámetros del modelo.
    entrada : callable
        Función d(t) que devuelve la carga en peticiones/s.

    Devuelve
    --------
    numpy.ndarray
        Derivadas [x', y'] en peticiones/s y réplicas/s.
    """
    x, y = estado
    d = entrada(t)
    dx = -p.mu * y + d
    dy = (-y + p.K * x) / p.tau
    return np.array([dx, dy], dtype=float)


# ---------------------------------------------------------------------------
# 4. Chequeo de consistencia dimensional
# ---------------------------------------------------------------------------

def verificar_dimensiones(p: Parametros = BASE) -> Dict[str, object]:
    """Comprueba las combinaciones adimensionales del modelo.

    No se usa una librería de unidades para no añadir dependencias. En su
    lugar se verifican numéricamente las relaciones que deben ser
    adimensionales y se documentan las unidades de cada término:

      * x' = -mu*y + d:
          [peticiones/s] = [peticiones/(s*réplica)]*[réplica]
                         = [peticiones/s].
      * tau*y' = -y + K*x:
          [s]*[réplicas/s] = [réplicas] = [réplicas/petición]*[peticiones].
      * mu*K*tau:
          [peticiones/(s*réplica)]*[réplicas/petición]*[s] = 1.
      * wn^2 = mu*K/tau:
          [1/s^2] = [peticiones/(s*réplica)]*[réplicas/petición]/[s].
      * zeta = 1/(2*sqrt(mu*K*tau)): adimensional.

    Devuelve un diccionario con los valores y con el resultado booleano.
    """
    grupo_adimensional = p.mu * p.K * p.tau
    zeta_desde_grupo = 1.0 / (2.0 * np.sqrt(grupo_adimensional))
    wn_desde_grupo = np.sqrt(p.mu * p.K / p.tau)

    chequeos = {
        "mu_K_tau_adimensional": float(grupo_adimensional),
        "zeta_consistente": bool(np.isclose(zeta_desde_grupo, p.zeta)),
        "wn_consistente": bool(np.isclose(wn_desde_grupo, p.wn)),
        # Si el modelo es estable, ambos valores propios deben tener parte
        # real negativa (criterio de estabilidad asintótica).
        "estable": bool(np.all(np.real(p.valores_propios) < 0.0)),
    }
    return chequeos


if __name__ == "__main__":
    # Pequeña autocomprobación al ejecutar el módulo directamente.
    print("Caso base:", BASE)
    print("wn =", BASE.wn, "rad/s")
    print("zeta =", BASE.zeta)
    print("régimen =", BASE.regimen)
    print("x_ss =", BASE.x_ss, "peticiones")
    print("y_ss =", BASE.y_ss, "réplicas")
    print("valores propios =", BASE.valores_propios)
    print("chequeo dimensional =", verificar_dimensiones(BASE))
