# EcuacionesDiferenciales

## Simulador de Mitigación de Ataques DDoS

Aplicación de escritorio desarrollada en Python para el trabajo final de **Ecuaciones Diferenciales**. El simulador modela la dinámica de la anomalía de tráfico malicioso en dos capas de defensa acopladas:

- **Firewall Perimetral (Edge)**
- **WAF Interno**

La interfaz permite modificar cuatro parámetros del sistema y recalcular en vivo la solución analítica exacta, la aproximación numérica mediante RK4 y la gráfica comparativa.

El proyecto completo está contenido en un único archivo: **`simulador_devops.py`**.

---

## Características

- Modelo lineal en espacio de estados con matriz `A` de `4 × 4`.
- Solución analítica exacta mediante `scipy.linalg.expm`.
- Cálculo de autovalores con `scipy.linalg.eigvals`.
- Clasificación automática del sistema como **ESTABLE** o **INESTABLE**.
- Implementación manual de **Runge-Kutta de cuarto orden (RK4)** con un bucle `for` explícito.
- Cálculo del error absoluto medio entre la solución analítica y la numérica.
- Cuatro sliders interactivos con recálculo en tiempo real.
- Gráfica de Matplotlib incrustada en la ventana de Tkinter.
- Matriz `A` y autovalores actualizados en vivo.
- Diseño oscuro tipo **Dark Analytics** con paleta gris pizarra y azul eléctrico.
- Aplicación de escritorio nativa; no utiliza Streamlit, Flask ni tecnologías web.

---

## Requisitos

- **Python 3.10 o superior** (probado con Python 3.12).
- Dependencias:
  - `customtkinter`
  - `numpy`
  - `scipy`
  - `matplotlib`

Versiones utilizadas durante el desarrollo:

| Dependencia | Versión |
|---|---:|
| Python | 3.12.10 |
| customtkinter | 6.0.0 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Matplotlib | 3.10.3 |

---

## Instalación

```bash
git clone https://github.com/TrabajosUCC/EcuacionesDiferenciales.git
cd EcuacionesDiferenciales
```

Crear y activar un entorno virtual (opcional pero recomendado):

```bash
python -m venv .venv
```

En Windows:

```powershell
.venv\Scripts\activate
```

En Linux/macOS:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install customtkinter numpy scipy matplotlib
```

---

## Ejecución

```bash
python simulador_devops.py
```

La ventana se abre en `1280 × 800` píxeles y tiene un tamaño mínimo de `1180 × 800` para evitar que se recorte el panel de la matriz.

---

## Modelo matemático

### Vector de estado

El sistema utiliza cuatro variables de estado:

$$
x(t)=
\begin{bmatrix}
o_1(t)\\
r_1(t)\\
o_2(t)\\
r_2(t)
\end{bmatrix}
$$

donde:

- `o₁`: anomalía de tráfico malicioso normalizada en el **Firewall Perimetral (Edge)**.
- `o₂`: anomalía de tráfico malicioso normalizada en el **WAF Interno**.
- `r₁`, `r₂`: velocidad de respuesta, es decir, la derivada temporal de cada anomalía.

### Ecuaciones diferenciales

$$
\begin{aligned}
o_1' &= r_1 \\
r_1' &= \frac{-(k+k_{12})o_1-cr_1+k_{12}o_2}{m} \\
o_2' &= r_2 \\
r_2' &= \frac{-(k+k_{12})o_2-cr_2+k_{12}o_1}{m}
\end{aligned}
$$

### Matriz de estado

El sistema se escribe en forma matricial como:

$$
x' = Ax
$$

con:

$$
A=
\begin{bmatrix}
0 & 1 & 0 & 0 \\
-\dfrac{k+k_{12}}{m} & -\dfrac{c}{m} & \dfrac{k_{12}}{m} & 0 \\
0 & 0 & 0 & 1 \\
\dfrac{k_{12}}{m} & 0 & -\dfrac{k+k_{12}}{m} & -\dfrac{c}{m}
\end{bmatrix}
$$

### Parámetros interactivos

| Símbolo | Significado en el escenario DDoS | Rango | Valor por defecto |
|---|---|---:|---:|
| `m` | Inercia (Saturación del Buffer) | `0.2 – 5.0` | `1.0` |
| `c` | Amortiguamiento (Cooldown del IPS) | `0.0 – 5.0` | `0.8` |
| `k` | Rigidez (Agresividad de bloqueo) | `0.5 – 12.0` | `3.0` |
| `k12` | Acoplamiento (Tráfico hacia el WAF) | `0.0 – 6.0` | `1.2` |

### Condiciones de simulación

| Parámetro | Valor |
|---|---:|
| Tiempo final | `30.0 u.t.` |
| Número de puntos de malla | `401` |
| Paso de integración RK4 | `0.075 u.t.` |
| Condición inicial | `x(0) = [1, 0, 0, 0]ᵀ` |

La condición inicial representa al Firewall Perimetral con una anomalía inicial de `1.0` y al WAF Interno en `0.0`.

---

## Solución analítica exacta

Para un sistema lineal autónomo `x' = Ax`, la solución exacta es:

$$
x(t)=e^{At}x(0)
$$

El código calcula la exponencial matricial con:

```python
from scipy.linalg import expm

x_analitica = np.array([expm(A * ti) @ X0 for ti in t])
```

---

## Análisis de estabilidad

Los autovalores de `A` se calculan con:

```python
from scipy.linalg import eigvals

autovalores = eigvals(A)
```

La estabilidad se decide a partir de la **abscisa espectral**:

$$
\alpha=\max_i \operatorname{Re}(\lambda_i)
$$

| Condición | Clasificación |
|---|---|
| `α < -1e-7` | **ESTABLE**: todos los autovalores tienen parte real negativa. |
| `abs(α) ≤ 1e-7` | **INESTABLE / MARGINAL**: autovalores sobre el eje imaginario. |
| `α > 1e-7` | **INESTABLE**: al menos un autovalor tiene parte real positiva. |

Con `m > 0`, `c > 0`, `k > 0` y `k12 ≥ 0`, el sistema es asintóticamente estable. El caso `c = 0` produce una oscilación sostenida y se muestra como **INESTABLE / marginal**.

Los autovalores se ordenan de mayor a menor parte real y se muestran en la tarjeta inferior de la interfaz.

---

## Solución numérica con RK4

El método de Runge-Kutta de cuarto orden se implementa manualmente en la función `rk4_lineal`. Para el sistema `x' = Ax`, las etapas son:

$$
\begin{aligned}
k_1 &= A x_n \\
k_2 &= A\left(x_n+\frac{h}{2}k_1\right) \\
k_3 &= A\left(x_n+\frac{h}{2}k_2\right) \\
k_4 &= A\left(x_n+h k_3\right) \\
x_{n+1} &= x_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4)
\end{aligned}
$$

donde `h = 0.075`.

---

## Error numérico

El error absoluto medio total se calcula sobre los cuatro estados:

$$
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}\left|\mathbf{x}_{\text{num}}(t_i)-\mathbf{x}_{\text{ana}}(t_i)\right|
$$

También se calcula un error específico sobre las dos señales de anomalía de tráfico, `o₁` y `o₂`, que es el que se muestra como complemento en la tarjeta KPI:

$$
\mathrm{MAE}_{\text{anomalía}}=
\frac{1}{2N}\sum_{i=1}^{N}
\left(
|o_{1,\text{num}}(t_i)-o_{1,\text{ana}}(t_i)|
+
|o_{2,\text{num}}(t_i)-o_{2,\text{ana}}(t_i)|
\right)
$$

La interfaz muestra:

- **ERROR NUMÉRICO**: MAE sobre los cuatro estados.
- **MAE · Anomalía**: error sobre las señales `o₁` y `o₂`.

---

## Interfaz gráfica

### Panel izquierdo

- Cuatro sliders:
  - Inercia (Saturación del Buffer) `[m]`
  - Amortiguamiento (Cooldown del IPS) `[c]`
  - Rigidez (Agresividad de bloqueo) `[k]`
  - Acoplamiento (Tráfico hacia el WAF) `[k12]`
- Botón **Restablecer parámetros**.
- Matriz de estado `A` actualizada en vivo.
- Nota sobre el caso marginal `c = 0`.

### Panel derecho

- **Tarjetas KPI superiores**:
  - Estado del sistema: `ESTABLE` / `INESTABLE`.
  - Error numérico.
  - Abscisa espectral.
  - Tamaño de la malla RK4.
- **Gráfica incrustada**:
  - Eje X: tiempo.
  - Eje Y: anomalía de tráfico malicioso.
  - Solución analítica: línea continua.
  - Solución RK4: puntos discretos.
  - Firewall Perimetral: azul `#3b82f6`.
  - WAF Interno: ámbar `#f59e0b`.
- **Autovalores de A** ordenados y formateados.

### Paleta Dark Analytics

| Elemento | Color |
|---|---|
| Fondo principal | `#18181b` |
| Tarjetas y paneles | `#27272a` |
| Texto principal | `#f4f4f5` |
| Texto secundario | `#a1a1aa` |
| Acento / botones / sliders | `#3b82f6` |
| Firewall Perimetral | `#3b82f6` |
| WAF Interno | `#f59e0b` |

La figura y los ejes de Matplotlib utilizan `#27272a` para integrarse con el panel de Tkinter. Los spines superior y derecho de la gráfica están ocultos.

---

## Estructura del código

El archivo `simulador_devops.py` está organizado en secciones:

| Sección | Contenido |
|---|---|
| Configuración general | Tiempo, malla, condición inicial, colores y parámetros de sliders. |
| Capa 1: modelado | `Parametros`, `construir_matriz_A`. |
| Capa 2: analítica | `solucion_analitica`, `calcular_autovalores`, `clasificar_estabilidad`. |
| Capa 3: numérica | `rk4_lineal`, `simular`, cálculo de errores. |
| Capa 4: interfaz | `SimuladorDevOps`, sliders, KPIs, gráfica y actualización en vivo. |

---

## Funcionamiento de la interactividad

Cada vez que se mueve un slider:

1. Se actualiza el valor mostrado.
2. Se programa un recálculo con un pequeño debounce de `25 ms`.
3. Se reconstruye la matriz `A`.
4. Se calcula la solución analítica con `expm`.
5. Se calculan los autovalores con `eigvals`.
6. Se integra numéricamente con RK4 manual.
7. Se actualizan los KPIs, la matriz, los autovalores y la gráfica.

---

## Notas y limitaciones

- El modelo es lineal; la anomalía normalizada puede tomar valores negativos, interpretados como tráfico por debajo de la línea base.
- La malla de integración es fija (`401` puntos, `Δt = 0.075`). En configuraciones de rigidez extrema, el error de RK4 puede aumentar.
- El caso `c = 0` se clasifica como inestable/marginal porque la respuesta no decae asintóticamente.
- La aplicación está pensada como herramienta educativa para visualizar el comportamiento cualitativo del sistema, no como un sistema real de mitigación DDoS.

---

## Repositorio

**GitHub:** [https://github.com/TrabajosUCC/EcuacionesDiferenciales](https://github.com/TrabajosUCC/EcuacionesDiferenciales)

**Trabajo final — Ecuaciones Diferenciales**
