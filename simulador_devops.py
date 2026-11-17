"""
simulador_devops.py
===================

Trabajo final de Ecuaciones Diferenciales
-----------------------------------------
Simulador de escritorio para la mitigación de ataques DDoS en dos capas de
defensa: Firewall Perimetral (Edge) y WAF Interno.

El proyecto está organizado internamente en las cuatro capas solicitadas:

    Capa 1 - Modelado:
        Sistema lineal de 4 estados x' = A x. La matriz A de 4x4 se construye
        a partir de los parámetros m (inercia/saturación del buffer),
        c (amortiguamiento/cooldown del IPS), k (rigidez/agresividad de
        bloqueo) y k12 (acoplamiento/tráfico hacia el WAF).

    Capa 2 - Analítica:
        La solución exacta se calcula con scipy.linalg.expm:
            x(t) = exp(A t) x(0)
        Los valores propios se obtienen con scipy.linalg.eigvals y se muestra
        si el sistema es estable o inestable.

    Capa 3 - Numérica:
        Se implementa manualmente Runge-Kutta de cuarto orden (RK4) con un
        bucle for explícito. Se calcula el error absoluto medio entre la
        solución analítica y la numérica.

    Capa 4 - Interactividad:
        Interfaz nativa de escritorio con customtkinter. Cuatro sliders
        recalculan la solución analítica, la numérica y la gráfica en vivo.

Modelo del escenario DDoS
-------------------------
Cada capa de defensa (Firewall Perimetral Edge y WAF Interno) se representa
mediante un lazo de segundo orden:

    o1' = r1
    r1' = [ -(k + k12) o1 - c r1 + k12 o2 ] / m

    o2' = r2
    r2' = [ -(k + k12) o2 - c r2 + k12 o1 ] / m

donde:
    o1, o2 : anomalía de tráfico malicioso normalizada en cada capa.
    r1, r2 : velocidad de respuesta (derivada de la anomalía).
    m      : inercia / saturación del buffer.
    c      : amortiguamiento / cooldown del IPS.
    k      : rigidez / agresividad de bloqueo.
    k12    : acoplamiento / tráfico hacia el WAF.

El vector de estado es x = [o1, r1, o2, r2]^T y, por tanto:

        [   0          1        0          0    ]
    A = [ -(k+k12)/m  -c/m     k12/m       0    ]
        [   0          0        0          1    ]
        [  k12/m        0    -(k+k12)/m   -c/m   ]

Con m > 0, c > 0 y k > 0 el sistema es asintóticamente estable. Si c = 0,
el sistema queda en el límite de estabilidad (autovalores sobre el eje
imaginario): la respuesta no decae y se clasifica como INESTABLE/MARGINAL.

Ejecución
---------
    python simulador_devops.py

Dependencias:
    pip install customtkinter numpy scipy matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigvals, expm

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =============================================================================
# 1. CONFIGURACIÓN GENERAL
# =============================================================================

# Ventana temporal y resolución de la simulación.
T_FINAL: float = 30.0          # segundos / unidades de tiempo
N_PUNTOS: int = 401            # puntos de la malla (dt = 30 / 400 = 0.075)

# Condición inicial: el Firewall Perimetral (Edge) arranca con anomalía 1.0
# y el WAF Interno en 0. x = [o1, r1, o2, r2]
X0: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

# Tolerancia para decidir estabilidad numérica.
TOL_ESTABILIDAD: float = 1e-7

# Paleta visual "Dark Analytics": gris pizarra + azul eléctrico.
COLOR_FONDO = "#18181b"
COLOR_SIDEBAR = "#18181b"
COLOR_TARJETA = "#27272a"
COLOR_BORDE = "#3f3f46"
COLOR_TEXTO = "#f4f4f5"
COLOR_TEXTO_SUAVE = "#a1a1aa"
COLOR_ACENTO = "#3b82f6"
COLOR_ACENTO_2 = "#f59e0b"
COLOR_VERDE = "#22c55e"
COLOR_VERDE_CLARO = "#86efac"
COLOR_ROJO = "#ef4444"
COLOR_ROJO_CLARO = "#fca5a5"
COLOR_AMARILLO = "#f59e0b"

# Parámetros que se muestran en el panel izquierdo.
# (clave, etiqueta, mínimo, máximo, valor por defecto, número de pasos)
CONFIG_SLIDERS = (
    ("m", "Inercia (Saturación del Buffer) [m]", 0.2, 5.0, 1.0, 240),
    ("c", "Amortiguamiento (Cooldown del IPS) [c]", 0.0, 5.0, 0.8, 500),
    ("k", "Rigidez (Agresividad de bloqueo) [k]", 0.5, 12.0, 3.0, 460),
    ("k12", "Acoplamiento (Tráfico hacia el WAF) [k12]", 0.0, 6.0, 1.2, 300),
)

VALORES_DEFECTO = {clave: defecto for clave, _, _, _, defecto, _ in CONFIG_SLIDERS}

# Escenarios didácticos de la guía interactiva.
# (nombre, m, c, k, k12, descripción)
ESCENARIOS_GUIA: tuple[tuple[str, float, float, float, float, str], ...] = (
    (
        "1. Operación Normal (Defecto)",
        1.0,
        0.8,
        3.0,
        1.2,
        "Configuración de referencia: el IPS aplica cooldown y el sistema "
        "vuelve al equilibrio de forma asintótica.",
    ),
    (
        "2. Sin Cooldown IPS (Marginal)",
        1.0,
        0.0,
        3.0,
        1.2,
        "Sin amortiguamiento (c = 0) la energía no se disipa: la anomalía "
        "oscila de forma sostenida y el sistema queda en el límite de estabilidad.",
    ),
    (
        "3. Rigidez Extrema (Límite RK4)",
        0.2,
        0.0,
        12.0,
        6.0,
        "Masa mínima y rigidez máxima: aparecen modos muy rápidos y el paso "
        "fijo de RK4 pierde precisión frente a la solución exacta.",
    ),
)


# =============================================================================
# 2. CAPA 1: MODELADO MATEMÁTICO
# =============================================================================

@dataclass(frozen=True)
class Parametros:
    """Parámetros del lazo de mitigación de ataques DDoS."""

    m: float
    c: float
    k: float
    k12: float


def construir_matriz_A(p: Parametros) -> np.ndarray:
    """
    Construye la matriz de estado A de 4x4.

    El estado es x = [o1, r1, o2, r2]^T, con:
        o_i : anomalía de tráfico malicioso de la capa i.
        r_i : derivada temporal de la anomalía.

    Las ecuaciones acopladas son:
        o1' = r1
        r1' = [-(k + k12) o1 - c r1 + k12 o2] / m
        o2' = r2
        r2' = [-(k + k12) o2 - c r2 + k12 o1] / m
    """

    if p.m <= 0.0:
        # El slider nunca llega aquí, pero se protege la división por cero.
        raise ValueError(
            "La inercia (saturación del buffer) m debe ser estrictamente positiva."
        )

    return np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-(p.k + p.k12) / p.m, -p.c / p.m, p.k12 / p.m, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [p.k12 / p.m, 0.0, -(p.k + p.k12) / p.m, -p.c / p.m],
        ],
        dtype=float,
    )


# =============================================================================
# 3. CAPA 2: SOLUCIÓN ANALÍTICA EXACTA
# =============================================================================

def solucion_analitica(A: np.ndarray, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Calcula la solución exacta del sistema lineal x' = A x:

        x(t) = exp(A t) x(0)

    Se usa scipy.linalg.expm para cada instante de la malla. La exponencial
    matricial es la solución exacta del sistema lineal autónomo.
    """

    # Cada fila de la lista es el vector de estado x(t) en un instante.
    return np.array([expm(A * ti) @ x0 for ti in t], dtype=float)


def calcular_autovalores(A: np.ndarray) -> np.ndarray:
    """Obtiene los valores propios de A con scipy.linalg.eigvals."""

    return eigvals(A)


# =============================================================================
# 4. CAPA 3: SOLUCIÓN NUMÉRICA CON RK4 MANUAL
# =============================================================================

def rk4_lineal(A: np.ndarray, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Resuelve x' = A x con Runge-Kutta de cuarto orden implementado a mano.

    No se utiliza scipy.integrate: el bucle for y las cuatro etapas k1, k2,
    k3 y k4 se escriben explícitamente, tal como exige el proyecto.

    Parámetros
    ----------
    A : matriz 4x4 del sistema.
    x0 : condición inicial de dimensión 4.
    t : malla temporal uniforme.

    Retorna
    -------
    x : matriz (len(t), 4) con la aproximación numérica en cada instante.
    """

    n = len(t)
    dt = float(t[1] - t[0])

    x = np.zeros((n, x0.size), dtype=float)
    x[0] = x0

    # Iteración temporal explícita.
    for i in range(n - 1):
        xi = x[i]

        # Etapas clásicas de RK4 para una EDO lineal x' = A x.
        k1 = A @ xi
        k2 = A @ (xi + 0.5 * dt * k1)
        k3 = A @ (xi + 0.5 * dt * k2)
        k4 = A @ (xi + dt * k3)

        # Combinación ponderada de las cuatro pendientes.
        x[i + 1] = xi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return x


# =============================================================================
# 5. RESULTADO COMPLETO DE LA SIMULACIÓN
# =============================================================================

@dataclass
class ResultadoSimulacion:
    """Empaqueta todo lo que necesita la interfaz para actualizarse."""

    t: np.ndarray
    x_analitica: np.ndarray
    x_numerica: np.ndarray
    autovalores: np.ndarray
    error_medio_total: float
    error_medio_cpu: float
    abscisa_espectral: float


def simular(p: Parametros, t_final: float = T_FINAL, n_puntos: int = N_PUNTOS) -> ResultadoSimulacion:
    """Ejecuta las capas 1, 2 y 3 y devuelve todos los resultados."""

    # Capa 1: matriz de estado.
    A = construir_matriz_A(p)

    # Malla temporal común para comparar analítica y numérica punto a punto.
    t = np.linspace(0.0, t_final, n_puntos)

    # Capa 2: solución exacta y autovalores.
    x_analitica = solucion_analitica(A, X0, t)
    autovalores = calcular_autovalores(A)

    # Capa 3: solución numérica con RK4 manual.
    x_numerica = rk4_lineal(A, X0, t)

    # Error absoluto medio sobre los 4 estados.
    error_medio_total = float(np.mean(np.abs(x_numerica - x_analitica)))

    # Error absoluto medio sobre las dos señales de anomalía de tráfico (o1, o2).
    # Es el error que se ve directamente en la gráfica.
    error_medio_cpu = float(
        np.mean(np.abs(x_numerica[:, [0, 2]] - x_analitica[:, [0, 2]]))
    )

    # La abscisa espectral es la mayor parte real de los autovalores.
    # Si es negativa, todas las perturbaciones decaen.
    abscisa_espectral = float(np.max(autovalores.real))

    return ResultadoSimulacion(
        t=t,
        x_analitica=x_analitica,
        x_numerica=x_numerica,
        autovalores=autovalores,
        error_medio_total=error_medio_total,
        error_medio_cpu=error_medio_cpu,
        abscisa_espectral=abscisa_espectral,
    )


def clasificar_estabilidad(abscisa: float) -> tuple[bool, str]:
    """
    Clasifica el sistema a partir de la abscisa espectral.

    Retorna (es_estable, etiqueta).
    - Estable asintóticamente: Re(lambda_max) < 0.
    - Marginal / inestable: Re(lambda_max) >= 0.
    """

    if abscisa < -TOL_ESTABILIDAD:
        return True, "estable"
    if abs(abscisa) <= TOL_ESTABILIDAD:
        return False, "marginal"
    return False, "inestable"


def formatear_complejo(z: complex, precision: int = 5) -> str:
    """Devuelve un autovalor en formato legible, real o complejo."""

    real = float(np.real(z))
    imag = float(np.imag(z))

    if abs(imag) < 1e-10:
        return f"{real:+.{precision}f}"

    signo = "+" if imag >= 0.0 else "-"
    return f"{real:+.{precision}f} {signo} {abs(imag):.{precision}f}j"


# =============================================================================
# 6. CAPA 4: INTERFAZ GRÁFICA DE ESCRITORIO
# =============================================================================

class SimuladorDevOps(ctk.CTk):
    """Ventana principal del simulador."""

    def __init__(self) -> None:
        # Tema oscuro nativo de customtkinter.
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        super().__init__()

        self.title("Simulador · Mitigación de Ataques DDoS en dos capas")
        self.geometry("1280x800")
        self.minsize(1180, 800)
        self.configure(fg_color=COLOR_FONDO)

        # Estructura general: panel izquierdo fijo + panel derecho expansible.
        self.grid_columnconfigure(0, weight=0, minsize=390)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Referencias a widgets que se actualizan en vivo.
        self.sliders: dict[str, ctk.CTkSlider] = {}
        self.valores_labels: dict[str, ctk.CTkLabel] = {}
        self._job_actualizacion: str | None = None

        # Último resultado simulado y estado de la ventana de ayuda.
        self._ultimo_resultado: ResultadoSimulacion | None = None
        self._ventana_guia: ctk.CTkToplevel | None = None
        self._guia_live_labels: dict[str, ctk.CTkLabel] = {}
        self._guia_escenario_label: ctk.CTkLabel | None = None

        # Construcción de la interfaz.
        self._crear_sidebar()
        self._crear_panel_principal()

        # Primera simulación al abrir la ventana.
        self.after(80, self._actualizar_todo)

    # -------------------------------------------------------------------------
    # 6.1 PANEL IZQUIERDO: CONTROLES
    # -------------------------------------------------------------------------

    def _crear_sidebar(self) -> None:
        """Crea el panel izquierdo con los cuatro sliders y la matriz A."""

        self.sidebar = ctk.CTkFrame(
            self,
            fg_color=COLOR_SIDEBAR,
            corner_radius=0,
            width=390,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # Encabezado de marca.
        ctk.CTkLabel(
            self.sidebar,
            text="EDO · CIBERSEGURIDAD",
            font=("Segoe UI", 11, "bold"),
            text_color=COLOR_ACENTO,
            anchor="w",
        ).pack(fill="x", padx=28, pady=(20, 4))

        ctk.CTkLabel(
            self.sidebar,
            text="Mitigación de\nAtaques DDoS",
            font=("Segoe UI", 24, "bold"),
            text_color=COLOR_TEXTO,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=28)

        ctk.CTkLabel(
            self.sidebar,
            text=(
                "Sistema lineal de 4 estados con acoplamiento. "
                "Ajusta la defensa y observa la respuesta en vivo."
            ),
            font=("Segoe UI", 11),
            text_color=COLOR_TEXTO_SUAVE,
            justify="left",
            anchor="w",
            wraplength=334,
        ).pack(fill="x", padx=28, pady=(8, 0))

        # Contenedor de sliders: sin separadores ni bordes duros.
        sliders_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        sliders_frame.pack(fill="x", padx=28, pady=(16, 0))

        for clave, etiqueta, minimo, maximo, defecto, pasos in CONFIG_SLIDERS:
            self._crear_tarjeta_slider(
                parent=sliders_frame,
                clave=clave,
                etiqueta=etiqueta,
                minimo=minimo,
                maximo=maximo,
                defecto=defecto,
                pasos=pasos,
            )

        # Botón para volver a los parámetros iniciales.
        self.boton_reset = ctk.CTkButton(
            self.sidebar,
            text="Restablecer parámetros",
            command=self._restablecer,
            height=42,
            corner_radius=8,
            fg_color=COLOR_ACENTO,
            hover_color="#2563eb",
            font=("Segoe UI", 12, "bold"),
        )
        self.boton_reset.pack(fill="x", padx=28, pady=(8, 6))

        # Botón secundario y discreto de ayuda interactiva.
        self.boton_guia = ctk.CTkButton(
            self.sidebar,
            text="Guía e Interpretación en Vivo",
            command=self._abrir_guia_interactiva,
            height=38,
            corner_radius=8,
            fg_color=COLOR_TARJETA,
            border_width=1,
            border_color=COLOR_BORDE,
            hover_color=COLOR_BORDE,
            text_color=COLOR_TEXTO,
            font=("Segoe UI", 12),
        )
        self.boton_guia.pack(fill="x", padx=28, pady=(0, 10))

        # Nota sobre el caso marginal.
        ctk.CTkLabel(
            self.sidebar,
            text="c = 0 → sin cooldown del IPS: oscilación sostenida (marginal).",
            font=("Segoe UI", 10),
            text_color=COLOR_TEXTO_SUAVE,
            justify="left",
            anchor="w",
            wraplength=334,
        ).pack(fill="x", padx=28)

        # Tarjeta con la matriz A actualizada.
        self.tarjeta_matriz = ctk.CTkFrame(
            self.sidebar,
            fg_color=COLOR_TARJETA,
            corner_radius=8,
        )
        self.tarjeta_matriz.pack(fill="x", padx=28, pady=(14, 16))

        ctk.CTkLabel(
            self.tarjeta_matriz,
            text="Matriz de estado A (4×4)",
            font=("Segoe UI", 11, "bold"),
            text_color=COLOR_TEXTO,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(10, 4))

        ctk.CTkLabel(
            self.tarjeta_matriz,
            text="x = [o1, r1, o2, r2]ᵀ   ·   x' = A x",
            font=("Segoe UI", 9),
            text_color=COLOR_TEXTO_SUAVE,
            anchor="w",
        ).pack(fill="x", padx=16)

        self.matriz_label = ctk.CTkLabel(
            self.tarjeta_matriz,
            text="",
            font=("Consolas", 10),
            text_color=COLOR_TEXTO_SUAVE,
            justify="left",
            anchor="w",
        )
        self.matriz_label.pack(fill="x", padx=16, pady=(6, 4))

        ctk.CTkLabel(
            self.tarjeta_matriz,
            text="x(0) = [1, 0, 0, 0]ᵀ",
            font=("Segoe UI", 9),
            text_color=COLOR_TEXTO_SUAVE,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(0, 10))

    def _crear_tarjeta_slider(
        self,
        parent: ctk.CTkFrame,
        clave: str,
        etiqueta: str,
        minimo: float,
        maximo: float,
        defecto: float,
        pasos: int,
    ) -> None:
        """Crea una tarjeta con un slider y su valor numérico."""

        tarjeta = ctk.CTkFrame(
            parent,
            fg_color=COLOR_TARJETA,
            corner_radius=8,
        )
        tarjeta.pack(fill="x", pady=(0, 8))

        fila = ctk.CTkFrame(tarjeta, fg_color="transparent")
        fila.pack(fill="x", padx=16, pady=(10, 0))

        ctk.CTkLabel(
            fila,
            text=etiqueta,
            font=("Segoe UI", 11),
            text_color=COLOR_TEXTO,
            anchor="w",
        ).pack(side="left")

        valor_label = ctk.CTkLabel(
            fila,
            text=f"{defecto:.2f}",
            font=("Consolas", 12, "bold"),
            text_color=COLOR_TEXTO,
        )
        valor_label.pack(side="right")

        slider = ctk.CTkSlider(
            tarjeta,
            from_=minimo,
            to=maximo,
            number_of_steps=pasos,
            command=lambda v, c=clave: self._al_mover_slider(c, v),
            height=16,
            fg_color=COLOR_BORDE,
            progress_color=COLOR_ACENTO,
            button_color=COLOR_ACENTO,
            button_hover_color="#60a5fa",
        )
        slider.set(defecto)
        slider.pack(fill="x", padx=16, pady=(2, 10))

        self.sliders[clave] = slider
        self.valores_labels[clave] = valor_label

    # -------------------------------------------------------------------------
    # 6.2 PANEL DERECHO: MÉTRICAS, GRÁFICA Y AUTOVALORES
    # -------------------------------------------------------------------------

    def _crear_panel_principal(self) -> None:
        """Crea el encabezado, las métricas, la gráfica y los autovalores."""

        self.main = ctk.CTkFrame(self, fg_color="transparent")
        self.main.grid(row=0, column=1, sticky="nsew", padx=(26, 26), pady=24)
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(2, weight=1)

        # ------------------------------------------------------------------
        # Encabezado limpio, sin semáforo lateral.
        # ------------------------------------------------------------------
        header = ctk.CTkFrame(self.main, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Mitigación de Ataques DDoS",
            font=("Segoe UI", 26, "bold"),
            text_color=COLOR_TEXTO,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text="Firewall Perimetral (Edge) vs. WAF Interno · analítica exacta vs. Runge-Kutta 4",
            font=("Segoe UI", 11),
            text_color=COLOR_TEXTO_SUAVE,
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        # ------------------------------------------------------------------
        # Fila superior de KPI: estado, error, abscisa y malla.
        # ------------------------------------------------------------------
        kpis = ctk.CTkFrame(self.main, fg_color="transparent")
        kpis.grid(row=1, column=0, sticky="ew", pady=(20, 0))
        for columna in range(4):
            kpis.grid_columnconfigure(columna, weight=1, uniform="kpi")

        self._crear_kpi_estado(kpis, columna=0)

        self.error_value, self.error_sub = self._crear_metrica(
            kpis,
            columna=1,
            titulo="ERROR NUMÉRICO",
            valor="—",
            subtitulo="MAE · 4 estados",
        )

        self.abscisa_value, self.abscisa_sub = self._crear_metrica(
            kpis,
            columna=2,
            titulo="ABSCISA ESPECTRAL",
            valor="—",
            subtitulo="máx Re(λ)",
        )

        self.malla_value, self.malla_sub = self._crear_metrica(
            kpis,
            columna=3,
            titulo="MALLA RK4",
            valor=str(N_PUNTOS),
            subtitulo=f"pasos · Δt = {T_FINAL / (N_PUNTOS - 1):.3f}",
        )

        # ------------------------------------------------------------------
        # Tarjeta de la gráfica incrustada.
        # ------------------------------------------------------------------
        self.tarjeta_grafica = ctk.CTkFrame(
            self.main,
            fg_color=COLOR_TARJETA,
            corner_radius=8,
        )
        self.tarjeta_grafica.grid(row=2, column=0, sticky="nsew", pady=(20, 20))
        self.tarjeta_grafica.grid_columnconfigure(0, weight=1)
        self.tarjeta_grafica.grid_rowconfigure(1, weight=1)

        cabecera_grafica = ctk.CTkFrame(self.tarjeta_grafica, fg_color="transparent")
        cabecera_grafica.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 0))

        ctk.CTkLabel(
            cabecera_grafica,
            text="Tiempo vs. Anomalía de Tráfico Malicioso",
            font=("Segoe UI", 13, "bold"),
            text_color=COLOR_TEXTO,
            anchor="w",
        ).pack(side="left")

        ctk.CTkLabel(
            cabecera_grafica,
            text="— analítica   ○ RK4",
            font=("Segoe UI", 10),
            text_color=COLOR_TEXTO_SUAVE,
            anchor="e",
        ).pack(side="right")

        self.cuerpo_grafica = ctk.CTkFrame(
            self.tarjeta_grafica,
            fg_color=COLOR_TARJETA,
            corner_radius=0,
        )
        self.cuerpo_grafica.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        self._crear_figura()

        # ------------------------------------------------------------------
        # Tarjeta de autovalores.
        # ------------------------------------------------------------------
        self.tarjeta_eigen = ctk.CTkFrame(
            self.main,
            fg_color=COLOR_TARJETA,
            corner_radius=8,
        )
        self.tarjeta_eigen.grid(row=3, column=0, sticky="ew")
        self.tarjeta_eigen.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.tarjeta_eigen,
            text="Autovalores de A · scipy.linalg.eigvals",
            font=("Segoe UI", 12, "bold"),
            text_color=COLOR_TEXTO,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(14, 4))

        eigen_grid = ctk.CTkFrame(self.tarjeta_eigen, fg_color="transparent")
        eigen_grid.grid(row=1, column=0, sticky="ew", padx=20, pady=(4, 14))
        eigen_grid.grid_columnconfigure(0, weight=1)
        eigen_grid.grid_columnconfigure(1, weight=1)

        self.eigen_labels: list[ctk.CTkLabel] = []
        for i in range(4):
            label = ctk.CTkLabel(
                eigen_grid,
                text=f"λ{i + 1} = —",
                font=("Consolas", 12),
                text_color=COLOR_TEXTO_SUAVE,
                anchor="w",
            )
            label.grid(
                row=i // 2,
                column=i % 2,
                sticky="w",
                padx=(0, 18),
                pady=2,
            )
            self.eigen_labels.append(label)

    def _crear_kpi_estado(self, parent: ctk.CTkFrame, columna: int) -> None:
        """Crea la tarjeta KPI de estabilidad del sistema."""

        tarjeta = ctk.CTkFrame(
            parent,
            fg_color=COLOR_TARJETA,
            corner_radius=8,
        )
        tarjeta.grid(
            row=0,
            column=columna,
            sticky="ew",
            padx=(0, 8),
        )

        ctk.CTkLabel(
            tarjeta,
            text="ESTADO DEL SISTEMA",
            font=("Segoe UI", 10, "bold"),
            text_color=COLOR_TEXTO_SUAVE,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(16, 0))

        self.status_badge = ctk.CTkLabel(
            tarjeta,
            text="CALCULANDO",
            font=("Segoe UI", 13, "bold"),
            text_color="#ffffff",
            fg_color=COLOR_BORDE,
            corner_radius=8,
            padx=14,
            pady=5,
        )
        self.status_badge.pack(anchor="w", padx=18, pady=(4, 0))

        self.status_detail = ctk.CTkLabel(
            tarjeta,
            text="Esperando la primera simulación…",
            font=("Segoe UI", 9),
            text_color=COLOR_TEXTO_SUAVE,
            justify="left",
            anchor="w",
            wraplength=180,
        )
        self.status_detail.pack(fill="x", padx=18, pady=(6, 16))

    def _crear_metrica(
        self,
        parent: ctk.CTkFrame,
        columna: int,
        titulo: str,
        valor: str,
        subtitulo: str,
    ) -> tuple[ctk.CTkLabel, ctk.CTkLabel]:
        """Crea una tarjeta KPI y devuelve sus etiquetas actualizables."""

        tarjeta = ctk.CTkFrame(
            parent,
            fg_color=COLOR_TARJETA,
            corner_radius=8,
        )
        tarjeta.grid(
            row=0,
            column=columna,
            sticky="ew",
            padx=(0 if columna == 0 else 8, 0 if columna == 3 else 8),
        )

        ctk.CTkLabel(
            tarjeta,
            text=titulo,
            font=("Segoe UI", 10, "bold"),
            text_color=COLOR_TEXTO_SUAVE,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(16, 0))

        valor_label = ctk.CTkLabel(
            tarjeta,
            text=valor,
            font=("Consolas", 22, "bold"),
            text_color=COLOR_TEXTO,
            anchor="w",
        )
        valor_label.pack(fill="x", padx=18, pady=(2, 0))

        sub_label = ctk.CTkLabel(
            tarjeta,
            text=subtitulo,
            font=("Segoe UI", 9),
            text_color=COLOR_TEXTO_SUAVE,
            anchor="w",
        )
        sub_label.pack(fill="x", padx=18, pady=(0, 16))

        return valor_label, sub_label

    def _crear_figura(self) -> None:
        """Incrusta una figura de matplotlib dentro de la ventana de Tk."""

        self.figura = Figure(
            figsize=(8.2, 4.8),
            dpi=100,
            facecolor=COLOR_TARJETA,
        )
        self.figura.patch.set_facecolor(COLOR_TARJETA)
        self.ejes = self.figura.add_subplot(111)
        self.ejes.set_facecolor(COLOR_TARJETA)
        self.figura.subplots_adjust(
            left=0.085,
            right=0.985,
            top=0.96,
            bottom=0.14,
        )

        self.canvas = FigureCanvasTkAgg(self.figura, master=self.cuerpo_grafica)
        widget_canvas = self.canvas.get_tk_widget()
        widget_canvas.configure(bg=COLOR_TARJETA, highlightthickness=0, bd=0)
        widget_canvas.pack(fill="both", expand=True)

    # -------------------------------------------------------------------------
    # 6.3 LÓGICA DE ACTUALIZACIÓN EN TIEMPO REAL
    # -------------------------------------------------------------------------

    def _al_mover_slider(self, clave: str, valor: float) -> None:
        """
        Actualiza la etiqueta del slider y programa un recálculo.

        Se usa un pequeño debounce de 25 ms para no lanzar decenas de
        simulaciones completas mientras el usuario arrastra el control.
        """

        self.valores_labels[clave].configure(text=f"{float(valor):.2f}")

        if self._job_actualizacion is not None:
            try:
                self.after_cancel(self._job_actualizacion)
            except Exception:
                pass

        self._job_actualizacion = self.after(25, self._actualizar_todo)

    def _leer_parametros(self) -> Parametros:
        """Lee los valores actuales de los cuatro sliders."""

        return Parametros(
            m=float(self.sliders["m"].get()),
            c=float(self.sliders["c"].get()),
            k=float(self.sliders["k"].get()),
            k12=float(self.sliders["k12"].get()),
        )

    def _actualizar_todo(self) -> None:
        """Recalcula Capa 2 y Capa 3 y actualiza toda la interfaz."""

        self._job_actualizacion = None

        try:
            p = self._leer_parametros()
            A = construir_matriz_A(p)
            resultado = simular(p)

            # Se conserva el último resultado para la guía interactiva.
            self._ultimo_resultado = resultado

            self._actualizar_matriz(A)
            self._actualizar_textos(resultado)
            self._actualizar_grafica(resultado)
            self._refrescar_guia()

        except Exception as exc:
            # La interfaz no debe cerrarse si ocurre un error inesperado.
            self.status_badge.configure(
                text="ERROR",
                fg_color="#7f1d1d",
                text_color="#fecaca",
            )
            self.status_detail.configure(
                text=f"No se pudo simular:\n{exc}",
                text_color=COLOR_ROJO_CLARO,
            )

    def _actualizar_matriz(self, A: np.ndarray) -> None:
        """Muestra la matriz A de 4x4 con los parámetros actuales."""

        filas = ["A ="]
        for fila in A:
            filas.append("[ " + "  ".join(f"{valor:7.3f}" for valor in fila) + " ]")
        self.matriz_label.configure(text="\n".join(filas))

    def _actualizar_textos(self, resultado: ResultadoSimulacion) -> None:
        """Actualiza métricas, autovalores y semáforo de estabilidad."""

        # Error absoluto medio total y error específico de las señales de anomalía.
        self.error_value.configure(text=f"{resultado.error_medio_total:.3e}")
        self.error_sub.configure(
            text=f"MAE · Anomalía: {resultado.error_medio_cpu:.3e}"
        )

        # Abscisa espectral.
        abscisa = resultado.abscisa_espectral
        self.abscisa_value.configure(text=f"{abscisa:+.4f}")

        # Autovalores ordenados de mayor a menor parte real.
        orden = np.argsort(resultado.autovalores.real)[::-1]
        autovalores_ordenados = resultado.autovalores[orden]
        for i, autovalor in enumerate(autovalores_ordenados):
            self.eigen_labels[i].configure(
                text=f"λ{i + 1} = {formatear_complejo(autovalor)}"
            )

        # Clasificación de estabilidad.
        estable, etiqueta = clasificar_estabilidad(abscisa)

        if estable:
            self.status_badge.configure(
                text="ESTABLE",
                fg_color=COLOR_VERDE,
                text_color="#ffffff",
            )
            self.status_detail.configure(
                text=(
                    f"Re(λmáx) = {abscisa:+.4f}\n"
                    "Decaimiento estable"
                ),
                text_color=COLOR_VERDE_CLARO,
            )
            self.abscisa_value.configure(text_color=COLOR_VERDE_CLARO)
            self.abscisa_sub.configure(
                text="máx Re(λ) < 0 · estable",
                text_color=COLOR_TEXTO_SUAVE,
            )

        elif etiqueta == "marginal":
            self.status_badge.configure(
                text="INESTABLE",
                fg_color=COLOR_AMARILLO,
                text_color="#1c1917",
            )
            self.status_detail.configure(
                text=(
                    f"Re(λmáx) ≈ 0\n"
                    "Oscilación sostenida"
                ),
                text_color=COLOR_AMARILLO,
            )
            self.abscisa_value.configure(text_color=COLOR_AMARILLO)
            self.abscisa_sub.configure(
                text="máx Re(λ) ≈ 0 · marginal",
                text_color=COLOR_AMARILLO,
            )

        else:
            self.status_badge.configure(
                text="INESTABLE",
                fg_color=COLOR_ROJO,
                text_color="#ffffff",
            )
            self.status_detail.configure(
                text=(
                    f"Re(λmáx) = {abscisa:+.4f}\n"
                    "Crecimiento inestable"
                ),
                text_color=COLOR_ROJO_CLARO,
            )
            self.abscisa_value.configure(text_color=COLOR_ROJO_CLARO)
            self.abscisa_sub.configure(
                text="máx Re(λ) > 0 · inestable",
                text_color=COLOR_ROJO_CLARO,
            )

    def _actualizar_grafica(self, resultado: ResultadoSimulacion) -> None:
        """Redibuja la comparación entre solución analítica y RK4."""

        ax = self.ejes
        ax.clear()

        # Estilo "Dark Analytics": fondo camuflado con la tarjeta de Tkinter.
        ax.set_facecolor(COLOR_TARJETA)
        ax.tick_params(
            axis="both",
            colors=COLOR_TEXTO_SUAVE,
            labelsize=9,
            length=0,
        )

        # Solo quedan los ejes inferior e izquierdo, en gris suave.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLOR_TEXTO_SUAVE)
        ax.spines["bottom"].set_color(COLOR_TEXTO_SUAVE)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        # Rejilla muy sutil, sin competir con las series.
        ax.grid(
            True,
            which="major",
            color=COLOR_BORDE,
            linewidth=0.6,
            alpha=0.35,
        )
        ax.set_axisbelow(True)

        t = resultado.t
        xa = resultado.x_analitica
        xn = resultado.x_numerica

        # Solución analítica: líneas continuas.
        ax.plot(
            t,
            xa[:, 0],
            color=COLOR_ACENTO,
            linewidth=2.2,
            label="Analítica · Firewall Perimetral (Edge)",
            zorder=3,
        )
        ax.plot(
            t,
            xa[:, 2],
            color=COLOR_ACENTO_2,
            linewidth=2.0,
            label="Analítica · WAF Interno",
            zorder=3,
        )

        # Solución numérica: puntos discretos (sin línea), para comparar.
        ax.plot(
            t,
            xn[:, 0],
            linestyle="none",
            marker="o",
            markersize=3.6,
            markerfacecolor=COLOR_TARJETA,
            markeredgecolor=COLOR_ACENTO,
            markeredgewidth=1.0,
            markevery=3,
            label="RK4 · Firewall Perimetral (Edge)",
            zorder=4,
        )
        ax.plot(
            t,
            xn[:, 2],
            linestyle="none",
            marker="s",
            markersize=3.3,
            markerfacecolor=COLOR_TARJETA,
            markeredgecolor=COLOR_ACENTO_2,
            markeredgewidth=1.0,
            markevery=3,
            label="RK4 · WAF Interno",
            zorder=4,
        )

        # Línea de referencia: anomalía nula (tráfico normal).
        ax.axhline(0.0, color=COLOR_BORDE, linewidth=0.8, alpha=0.8, zorder=1)

        # Límites dinámicos con un margen del 10 %.
        datos = np.concatenate([xa[:, 0], xa[:, 2], xn[:, 0], xn[:, 2]])
        minimo = float(np.min(datos))
        maximo = float(np.max(datos))
        if maximo - minimo < 1e-9:
            maximo = minimo + 1.0
        margen = 0.10 * (maximo - minimo)
        ax.set_xlim(0.0, T_FINAL)
        ax.set_ylim(minimo - margen, maximo + margen)

        ax.set_xlabel("Tiempo (u.t.)", color=COLOR_TEXTO_SUAVE, fontsize=10)
        ax.set_ylabel(
            "Anomalía de Tráfico Malicioso",
            color=COLOR_TEXTO_SUAVE,
            fontsize=10,
        )

        ax.legend(
            loc="upper right",
            ncol=2,
            fontsize=8,
            frameon=False,
            labelcolor=COLOR_TEXTO_SUAVE,
            borderpad=0.4,
            columnspacing=1.4,
            handlelength=2.2,
        )

        self.canvas.draw_idle()

    # -------------------------------------------------------------------------
    # 6.4 ACCIONES DE LA INTERFAZ
    # -------------------------------------------------------------------------

    def _abrir_guia_interactiva(self) -> None:
        """Abre la guía interactiva o enfoca la ventana si ya está abierta."""

        if self._guia_abierta():
            assert self._ventana_guia is not None
            self._ventana_guia.focus()
            self._ventana_guia.lift()
            return

        ventana = ctk.CTkToplevel(self)
        self._ventana_guia = ventana
        ventana.title("Guía e Interpretación en Vivo")
        ventana.configure(fg_color=COLOR_FONDO)
        ventana.transient(self)
        ventana.protocol("WM_DELETE_WINDOW", self._cerrar_guia)

        ventana.grid_columnconfigure(0, weight=1)
        ventana.grid_rowconfigure(1, weight=1)

        # Encabezado de la ventana emergente.
        cabecera = ctk.CTkFrame(ventana, fg_color="transparent")
        cabecera.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 14))

        ctk.CTkLabel(
            cabecera,
            text="Guía e Interpretación en Vivo",
            font=("Segoe UI", 20, "bold"),
            text_color=COLOR_TEXTO,
            anchor="w",
        ).pack(fill="x")

        ctk.CTkLabel(
            cabecera,
            text=(
                "Explicación dinámica del modelo, del método numérico "
                "y de los escenarios de prueba."
            ),
            font=("Segoe UI", 11),
            text_color=COLOR_TEXTO_SUAVE,
            anchor="w",
            justify="left",
            wraplength=740,
        ).pack(fill="x", pady=(4, 0))

        # Pestañas de la guía.
        pestanas = ctk.CTkTabview(
            ventana,
            fg_color=COLOR_TARJETA,
            segmented_button_fg_color=COLOR_FONDO,
            segmented_button_selected_color=COLOR_ACENTO,
            segmented_button_selected_hover_color="#2563eb",
            segmented_button_unselected_color=COLOR_FONDO,
            segmented_button_unselected_hover_color=COLOR_BORDE,
            text_color=COLOR_TEXTO,
            corner_radius=8,
        )
        pestanas.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

        pestana_vivo = pestanas.add("Análisis Actual (En Vivo)")
        pestana_capas = pestanas.add("Las 4 Capas del Modelo")
        pestana_escenarios = pestanas.add("Escenarios de Prueba")

        self._guia_live_labels.clear()
        self._crear_pestana_analisis(pestana_vivo)
        self._crear_pestana_capas(pestana_capas)
        self._crear_pestana_escenarios(pestana_escenarios)

        # Centrado sobre la ventana principal.
        ventana.update_idletasks()
        ancho, alto = 880, 660
        x = self.winfo_rootx() + max((self.winfo_width() - ancho) // 2, 0)
        y = self.winfo_rooty() + max((self.winfo_height() - alto) // 2, 0)
        ventana.geometry(f"{ancho}x{alto}+{x}+{y}")
        ventana.minsize(820, 560)

        self._refrescar_guia()
        ventana.lift()
        ventana.focus_force()

    def _cerrar_guia(self) -> None:
        """Cierra la guía interactiva y libera las referencias a sus widgets."""

        ventana = self._ventana_guia
        if ventana is not None:
            try:
                if ventana.winfo_exists():
                    ventana.destroy()
            except Exception:
                pass

        self._ventana_guia = None
        self._guia_live_labels.clear()
        self._guia_escenario_label = None

    def _guia_abierta(self) -> bool:
        """Indica si la ventana de la guía sigue existiendo."""

        if self._ventana_guia is None:
            return False

        try:
            return bool(self._ventana_guia.winfo_exists())
        except Exception:
            return False

    def _crear_tarjeta_guia(
        self,
        parent: ctk.CTkScrollableFrame,
        titulo: str,
    ) -> ctk.CTkLabel:
        """Crea una tarjeta de la guía y devuelve la etiqueta de su cuerpo."""

        tarjeta = ctk.CTkFrame(
            parent,
            fg_color=COLOR_FONDO,
            corner_radius=8,
        )
        tarjeta.pack(fill="x", padx=4, pady=(0, 12))

        ctk.CTkLabel(
            tarjeta,
            text=titulo,
            font=("Segoe UI", 11, "bold"),
            text_color=COLOR_ACENTO,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(14, 4))

        cuerpo = ctk.CTkLabel(
            tarjeta,
            text="",
            font=("Segoe UI", 11),
            text_color=COLOR_TEXTO_SUAVE,
            justify="left",
            anchor="w",
            wraplength=700,
        )
        cuerpo.pack(fill="x", padx=16, pady=(0, 14))

        return cuerpo

    def _crear_pestana_analisis(self, pestana: ctk.CTkFrame) -> None:
        """Construye la pestaña con el análisis dinámico del estado actual."""

        pestana.grid_columnconfigure(0, weight=1)
        pestana.grid_rowconfigure(0, weight=1)

        marco = ctk.CTkScrollableFrame(pestana, fg_color="transparent")
        marco.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self._guia_live_labels = {
            "estado": self._crear_tarjeta_guia(
                marco,
                "Estado actual y acoplamiento Edge-WAF",
            ),
            "decaimiento": self._crear_tarjeta_guia(
                marco,
                "Abscisa espectral y tiempo de estabilización",
            ),
            "error": self._crear_tarjeta_guia(
                marco,
                "Solución analítica vs. precisión de RK4",
            ),
        }

    def _crear_pestana_capas(self, pestana: ctk.CTkFrame) -> None:
        """Construye la pestaña que explica las cuatro capas del código."""

        pestana.grid_columnconfigure(0, weight=1)
        pestana.grid_rowconfigure(0, weight=1)

        marco = ctk.CTkScrollableFrame(pestana, fg_color="transparent")
        marco.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        capas: tuple[tuple[str, str], ...] = (
            (
                "Capa 1 · Matriz A 4×4 y acoplamiento Edge-WAF",
                "construir_matriz_A ensambla la matriz de estado a partir de m, c, k "
                "y k12. Cada capa de defensa aporta un lazo de segundo orden "
                "(anomalía y velocidad de respuesta) y el término k12/m acopla el "
                "Firewall Perimetral con el WAF Interno. Si k12 = 0, ambas capas "
                "evolucionan de forma independiente.",
            ),
            (
                "Capa 2 · Solución exacta y autovalores",
                "solucion_analitica evalúa x(t) = expm(A·t)·x(0) sobre toda la malla "
                "temporal, por lo que la curva analítica es la referencia exacta. "
                "calcular_autovalores usa scipy.linalg.eigvals y la abscisa espectral "
                "Re(λmáx) decide la clasificación: negativa ⇒ estable, nula ⇒ "
                "marginal, positiva ⇒ inestable.",
            ),
            (
                "Capa 3 · Runge-Kutta 4 manual con bucle explícito",
                "rk4_lineal implementa las cuatro etapas clásicas (k1, k2, k3, k4) "
                "dentro de un bucle for explícito, sin integradores automáticos. "
                "Con Δt = 0.075 u.t. el método es de cuarto orden: al refinar la "
                "malla el error cae aproximadamente como Δt⁴, pero con rigidez "
                "extrema el paso fijo puede dejar de resolver los modos rápidos.",
            ),
            (
                "Capa 4 · Simulación reactiva",
                "SimuladorDevOps conecta los cuatro sliders con _actualizar_todo, "
                "que reconstruye A, recalcula la solución analítica y la numérica, "
                "actualiza KPIs, autovalores y gráfica. Un debounce de 25 ms evita "
                "lanzar decenas de simulaciones mientras se arrastra un control. "
                "Esta guía se refresca con cada recálculo.",
            ),
        )

        for titulo, texto in capas:
            cuerpo = self._crear_tarjeta_guia(marco, titulo)
            cuerpo.configure(text=texto)

    def _crear_pestana_escenarios(self, pestana: ctk.CTkFrame) -> None:
        """Construye la pestaña con los escenarios de prueba interactivos."""

        pestana.grid_columnconfigure(0, weight=1)
        pestana.grid_rowconfigure(0, weight=1)

        marco = ctk.CTkScrollableFrame(pestana, fg_color="transparent")
        marco.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        ctk.CTkLabel(
            marco,
            text=(
                "Cada botón aplica los parámetros al panel principal, "
                "recalcula la simulación y actualiza esta guía."
            ),
            font=("Segoe UI", 11),
            text_color=COLOR_TEXTO_SUAVE,
            anchor="w",
            justify="left",
            wraplength=700,
        ).pack(fill="x", padx=4, pady=(4, 12))

        for nombre, m, c, k, k12, descripcion in ESCENARIOS_GUIA:
            ctk.CTkButton(
                marco,
                text=nombre,
                command=lambda m=m, c=c, k=k, k12=k12, d=descripcion: (
                    self._aplicar_escenario(m, c, k, k12, d)
                ),
                height=42,
                corner_radius=8,
                fg_color=COLOR_TARJETA,
                border_width=1,
                border_color=COLOR_BORDE,
                hover_color=COLOR_BORDE,
                text_color=COLOR_TEXTO,
                font=("Segoe UI", 12, "bold"),
                anchor="w",
            ).pack(fill="x", padx=4, pady=(0, 10))

        tarjeta = ctk.CTkFrame(
            marco,
            fg_color=COLOR_FONDO,
            corner_radius=8,
        )
        tarjeta.pack(fill="x", padx=4, pady=(8, 4))

        ctk.CTkLabel(
            tarjeta,
            text="Resultado del escenario",
            font=("Segoe UI", 11, "bold"),
            text_color=COLOR_ACENTO,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(14, 4))

        self._guia_escenario_label = ctk.CTkLabel(
            tarjeta,
            text="Aún no se ha aplicado ningún escenario.",
            font=("Segoe UI", 11),
            text_color=COLOR_TEXTO_SUAVE,
            justify="left",
            anchor="w",
            wraplength=700,
        )
        self._guia_escenario_label.pack(fill="x", padx=16, pady=(0, 14))

    def _textos_analisis_vivo(self) -> dict[str, str]:
        """Genera las explicaciones dinámicas de la pestaña en vivo."""

        p = self._leer_parametros()
        resultado = self._ultimo_resultado

        if resultado is None:
            sin_datos = "Todavía no hay una simulación calculada."
            return {
                "estado": sin_datos,
                "decaimiento": sin_datos,
                "error": sin_datos,
            }

        re_max = resultado.abscisa_espectral

        # --- Estado y acoplamiento -------------------------------------------
        if abs(p.c) <= TOL_ESTABILIDAD:
            clasificacion = "MARGINAL (sin cooldown del IPS)"
            explicacion = (
                "Con c = 0 no existe disipación de energía en el lazo: la "
                "anomalía oscila de forma sostenida y no vuelve al equilibrio. "
                "La respuesta queda en el límite de estabilidad."
            )
        elif re_max < -TOL_ESTABILIDAD:
            clasificacion = "ESTABLE"
            explicacion = (
                "El amortiguamiento del IPS domina y todas las perturbaciones "
                "decaen exponencialmente: la anomalía de tráfico desaparece y el "
                "sistema vuelve a su línea base."
            )
        elif abs(re_max) <= TOL_ESTABILIDAD:
            clasificacion = "MARGINAL"
            explicacion = (
                "La parte real máxima es prácticamente nula: el sistema está en "
                "el límite de estabilidad y la oscilación no decae."
            )
        else:
            clasificacion = "INESTABLE"
            explicacion = (
                "La parte real máxima es positiva: la anomalía crece sin límite "
                "y la defensa no logra contener el ataque."
            )

        if abs(p.k12) <= TOL_ESTABILIDAD:
            acoplamiento = (
                "Las capas están DESACOPLADAS (k12 = 0): el Firewall Perimetral "
                "y el WAF Interno evolucionan de forma independiente."
            )
        else:
            acoplamiento = (
                f"Las capas están ACOPLADAS (k12 = {p.k12:.2f}): parte de la "
                "anomalía del Edge se transfiere al WAF y ambas respuestas se "
                "influyen mutuamente."
            )

        texto_estado = (
            f"Clasificación: {clasificacion}\n\n"
            f"{explicacion}\n\n{acoplamiento}"
        )

        # --- Decaimiento exponencial -----------------------------------------
        if re_max < -TOL_ESTABILIDAD:
            tau = 1.0 / abs(re_max)
            t_est = 3.0 * tau
            texto_decaimiento = (
                f"Re(λmáx) = {re_max:+.4f}\n\n"
                "La envolvente de la respuesta decae como "
                f"exp({re_max:+.4f}·t).\n"
                f"Constante de tiempo: τ = 1/|Re(λmáx)| ≈ {tau:.3f} u.t.\n"
                f"Tiempo aproximado de estabilización: 3·τ ≈ {t_est:.3f} u.t. "
                "(la amplitud conserva ~5 % de su valor inicial)."
            )
        elif abs(re_max) <= TOL_ESTABILIDAD:
            texto_decaimiento = (
                f"Re(λmáx) ≈ {re_max:+.2e}\n\n"
                "La envolvente se comporta como exp(Re(λmáx)·t) ≈ 1: no hay "
                "decaimiento apreciable. La constante de tiempo τ → ∞ y no "
                "existe un tiempo finito de estabilización; la oscilación se "
                "mantiene."
            )
        else:
            texto_decaimiento = (
                f"Re(λmáx) = {re_max:+.4f} > 0\n\n"
                f"La envolvente crece como exp({re_max:+.4f}·t): la anomalía se "
                "amplifica con el tiempo. No existe tiempo de estabilización "
                "porque el sistema es inestable."
            )

        # --- Analítica vs. RK4 -----------------------------------------------
        error = resultado.error_medio_total
        error_anomalia = resultado.error_medio_cpu

        if error < 1e-9:
            calidad = "excelente (prácticamente exacta)"
        elif error < 1e-6:
            calidad = "muy alta"
        elif error < 1e-3:
            calidad = "buena"
        else:
            calidad = "limitada: conviene revisar la rigidez del sistema"

        texto_error = (
            f"MAE sobre los 4 estados: {error:.3e}\n"
            f"MAE sobre las anomalías o₁ y o₂: {error_anomalia:.3e}\n\n"
            "La curva analítica se obtiene con expm(A·t) y es la referencia "
            f"exacta; RK4 manual con Δt = {T_FINAL / (N_PUNTOS - 1):.3f} u.t. "
            f"la reproduce con calidad {calidad}."
        )

        return {
            "estado": texto_estado,
            "decaimiento": texto_decaimiento,
            "error": texto_error,
        }

    def _refrescar_guia(self) -> None:
        """Actualiza los textos dinámicos si la guía está abierta."""

        if not self._guia_abierta() or not self._guia_live_labels:
            return

        textos = self._textos_analisis_vivo()
        for clave, etiqueta in self._guia_live_labels.items():
            try:
                etiqueta.configure(text=textos[clave])
            except Exception:
                # La ventana pudo cerrarse entre la comprobación y la escritura.
                return

    def _aplicar_escenario(
        self,
        m: float,
        c: float,
        k: float,
        k12: float,
        descripcion: str,
    ) -> None:
        """Aplica un escenario predefinido a los sliders y recalcula todo."""

        if self._job_actualizacion is not None:
            try:
                self.after_cancel(self._job_actualizacion)
            except Exception:
                pass
            self._job_actualizacion = None

        valores = {"m": m, "c": c, "k": k, "k12": k12}
        for clave, valor in valores.items():
            self.sliders[clave].set(valor)
            self.valores_labels[clave].configure(text=f"{valor:.2f}")

        self._actualizar_todo()

        resultado = self._ultimo_resultado
        etiqueta = self._guia_escenario_label
        if etiqueta is None or resultado is None:
            return

        try:
            etiqueta.configure(
                text=(
                    f"Escenario aplicado: m = {m:.2f} · c = {c:.2f} · "
                    f"k = {k:.2f} · k12 = {k12:.2f}\n"
                    f"{descripcion}\n\n"
                    f"Estado: {self._estado_corto(resultado)} · "
                    f"Re(λmáx) = {resultado.abscisa_espectral:+.4f} · "
                    f"MAE = {resultado.error_medio_total:.3e}"
                )
            )
        except Exception:
            pass

    @staticmethod
    def _estado_corto(resultado: ResultadoSimulacion) -> str:
        """Devuelve una etiqueta breve del estado a partir del resultado."""

        if resultado.abscisa_espectral < -TOL_ESTABILIDAD:
            return "ESTABLE"
        if abs(resultado.abscisa_espectral) <= TOL_ESTABILIDAD:
            return "MARGINAL"
        return "INESTABLE"

    def _restablecer(self) -> None:
        """Devuelve los cuatro sliders a sus valores por defecto."""

        if self._job_actualizacion is not None:
            try:
                self.after_cancel(self._job_actualizacion)
            except Exception:
                pass
            self._job_actualizacion = None

        for clave, slider in self.sliders.items():
            valor = VALORES_DEFECTO[clave]
            slider.set(valor)
            self.valores_labels[clave].configure(text=f"{valor:.2f}")

        self._actualizar_todo()


# =============================================================================
# 7. PUNTO DE ENTRADA
# =============================================================================

def main() -> None:
    """Inicia la aplicación de escritorio."""

    app = SimuladorDevOps()
    app.mainloop()


if __name__ == "__main__":
    main()
