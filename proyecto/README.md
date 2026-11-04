# Simulador de autoescalado: oscilaciones en sistemas de control distribuido

Trabajo final de **Ecuaciones Diferenciales para Ingenieros** (Ingeniería de
Software). El proyecto modela, resuelve, simula y analiza el comportamiento de
un servicio en la nube cuyo autoescalador tarda en aprovisionar réplicas.

El entregable es completamente reproducible: un solo comando regenera las
figuras y las tablas a partir de los parámetros definidos en el código.

---

## 1. Qué hace el proyecto

1. **Modela** la cola de peticiones y la capacidad adicional con un sistema de
   dos EDOs de primer orden (equivalente a una EDO de segundo orden).
2. **Resuelve analíticamente** el sistema con transformada de Laplace para los
   regímenes subamortiguado, crítico y sobreamortiguado, y para una carga
   sinusoidal.
3. **Simula numéricamente** con Euler explícito y RK4 implementados a mano, y
   mide el error frente a la solución analítica.
4. **Analiza parámetros**: barridos de `tau` y `K`, rejilla 3×3, mapa de `zeta`,
   respuesta en frecuencia, resonancia y ubicación de polos.
5. Incluye una **extensión opcional** con retardo puro de comunicación, que es
   el mecanismo que sí puede producir inestabilidad en este tipo de lazo.

---

## 2. Requisitos e instalación

- Python **3.10 o superior**.
- Dependencias fijadas en `requirements.txt` (todas compatibles con Python
  3.10+).

```bash
# Desde la carpeta proyecto/
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

En Windows, si `python` no está en el PATH, usar `py -3` en lugar de `python`.

Para ejecutar las pruebas:

```bash
python -m pytest -q
```

`main.py` usa el backend `Agg` de Matplotlib, por lo que **no abre ventanas** y
guarda todas las figuras a 200 dpi en `resultados/figuras/`.

---

## 3. Estructura

```
proyecto/
├── README.md
├── requirements.txt
├── requirements-app.txt       # solo demo Streamlit (opcional)
├── main.py
├── app_streamlit.py           # demo interactiva opcional
├── src/
│   ├── __init__.py
│   ├── modelo.py              # parámetros, campo vectorial, unidades
│   ├── analitica.py           # Laplace, formas cerradas, resonancia, métricas
│   ├── numerico.py            # Euler y RK4 a mano + solve_ivp de referencia
│   ├── convergencia.py        # tabla y figura de convergencia
│   ├── parametros.py          # barridos, rejilla, mapas, polos, frecuencia
│   └── extension_retardo.py   # retardo puro, Padé 1 y simulación (opcional)
├── tests/
│   ├── test_basico.py
│   └── test_app_streamlit.py  # humo de la demo (se salta sin streamlit)
└── resultados/
    ├── figuras/
    ├── tabla_convergencia.csv
    ├── tabla_parametros.csv
    └── tabla_retardo.csv      # tabla extra de la extensión opcional
```

---

## 4. Modelo matemático

Variables de desviación respecto al equilibrio:

| Símbolo | Significado | Unidades |
|---|---|---|
| `x(t)` | peticiones pendientes en la cola | peticiones |
| `y(t)` | capacidad adicional (réplicas activas) | réplicas |
| `d(t)` | perturbación de carga entrante | peticiones/s |

Parámetros:

| Símbolo | Significado | Unidades |
|---|---|---|
| `mu` | capacidad de una réplica | peticiones/(s·réplica) |
| `K` | ganancia del autoescalador | réplicas/petición |
| `tau` | retardo de aprovisionamiento | s |
| `d0` | amplitud de la carga | peticiones/s |
| `T` | horizonte de simulación | s |

Sistema:

```text
x' = -mu*y + d(t)
tau*y' = -y + K*x
x(0) = 0,  y(0) = 0
```

Eliminando `y`:

```text
x'' + (1/tau)x' + (mu*K/tau)x = d'(t) + d(t)/tau
```

Cantidades derivadas:

```text
wn   = sqrt(mu*K/tau)              [rad/s]
zeta = 1 / (2*sqrt(mu*K*tau))      [adimensional]
A    = [[0, -mu], [K/tau, -1/tau]]
```

Los valores propios de `A` son las raíces de `tau*s^2 + s + mu*K = 0`.

### Consistencia dimensional

El módulo `src/modelo.py` incluye `verificar_dimensiones()`, que comprueba las
relaciones adimensionales y la estabilidad. Las comprobaciones manuales son:

- `x' = -mu*y + d`: `[peticiones/s] = [peticiones/(s·réplica)]·[réplica]`.
- `tau*y' = -y + K*x`: `[s]·[réplicas/s] = [réplicas] = [réplicas/petición]·[peticiones]`.
- `mu*K*tau`: `[peticiones/(s·réplica)]·[réplicas/petición]·[s] = 1`.
- `wn² = mu*K/tau`: `[1/s²]`.
- `zeta` y `mu*K*tau` son adimensionales.

### Estabilidad

Con `mu, K, tau > 0`, todos los coeficientes de
`tau*s^2 + s + mu*K` son positivos. Por el criterio de Routh-Hurwitz, el sistema
lineal es **siempre asintóticamente estable**. Lo que cambia con los parámetros
es el **amortiguamiento** (cuánto oscila la respuesta), no la estabilidad. La
inestabilidad real aparece solo al añadir retardo puro (extensión opcional).

### Caso base

```text
mu = 10, K = 0.005, tau = 20, d0 = 20, T = 600 s
mu*K*tau = 1  ->  zeta = 0.5,  wn = 0.05 rad/s
x_ss = d0/(mu*K) = 400 peticiones
y_ss = d0/mu = 2 réplicas
```

---

## 5. Solución analítica

La función de transferencia de carga a cola es

```text
G(s) = X(s)/D(s) = (tau*s + 1) / (tau*s^2 + s + mu*K)
```

y la de carga a capacidad es

```text
G_y(s) = K / (tau*s^2 + s + mu*K).
```

### Escalón

```text
X(s) = d0*(tau*s + 1) / (s*(tau*s^2 + s + mu*K))
Y(s) = d0*K / (s*(tau*s^2 + s + mu*K))
```

`src/analitica.py` implementa las formas cerradas para los tres regímenes y
elige automáticamente según `zeta`:

- **Subamortiguado** (`zeta < 1`): exponencial por seno/coseno con
  `wd = wn*sqrt(1-zeta^2)`.
- **Crítico** (`zeta = 1`): términos `exp(-wn*t)` y `t*exp(-wn*t)`.
- **Sobreamortiguado** (`zeta > 1`): dos exponenciales reales con las raíces
  `r1`, `r2` de la característica.

Además, SymPy deriva la solución con `inverse_laplace_transform` y el residuo
de la EDO se evalúa numéricamente en los tres regímenes. El residuo máximo
obtenido es 0.0 (o del orden de `1e-17`), y la comparación con
`solve_ivp` (DOP853, `rtol=1e-10`, `atol=1e-12`) da errores del orden de
`1e-7` en `x` y `1e-9` en `y` para el caso base.

### Estado estacionario

```text
x_ss = d0/(mu*K) = 400 peticiones
y_ss = d0/mu     = 2 réplicas
```

### Carga sinusoidal

Para `d(t) = d0*sin(w*t)`, el estado estacionario es

```text
x_p(t) = d0*|G(jw)|*sin(w*t + fase_x)
y_p(t) = d0*|G_y(jw)|*sin(w*t + fase_y)
```

La solución completa suma la particular y la homogénea, eligiendo los
coeficientes para que `x(0)=y(0)=0`.

El pico de `|G(jw)|` se obtiene derivando respecto de `u=w²`:

```text
u_res = [sqrt(c*(c+2)) - 1] / tau²,   c = mu*K*tau
```

Con `c = 1/(4*zeta²)`, existe un pico interior si
`zeta < sqrt((sqrt(2)+1)/4) ≈ 0.7769`. El umbral clásico `1/sqrt(2)` es para un
sistema de segundo orden **sin cero**; aquí el cero `s=-1/tau` desplaza el
umbral. Esta diferencia se documenta en el código y en el README.

### Métricas de respuesta

`metricas_escalon()` calcula, a partir de la solución analítica densa y con
refinamiento numérico:

- sobrepaso máximo respecto a `x_ss` (%),
- tiempo del pico (s),
- tiempo de asentamiento con banda del 2 % (s),
- periodo de oscilación amortiguada `2*pi/wd` (s) cuando `zeta<1`.

Como `G(s)` tiene un cero en `s=-1/tau`, las fórmulas clásicas de un sistema
de segundo orden sin cero (`Mp=exp(-pi*zeta/sqrt(1-zeta^2))`,
`tp=pi/wd`) **no son exactas** para este modelo. Por eso las métricas se
obtienen de la solución completa, no de esas expresiones aproximadas. Para el
caso base el sobrepaso es 29.84 %, mayor que el 16.30 % que predeciría la
fórmula clásica.

---

## 6. Simulación numérica

`src/numerico.py` implementa **a mano**:

- Euler explícito: `z_{n+1} = z_n + h*f(t_n,z_n)`.
- RK4 clásico: cuatro etapas con la entrada `d(t)` evaluada en
  `t_n`, `t_n+h/2`, `t_n+h/2` y `t_n+h`.

`solve_ivp` se usa solo como referencia independiente de validación, nunca como
método principal.

### Convergencia

La tabla `resultados/tabla_convergencia.csv` usa los pasos
`h = {2, 1, 0.5, 0.25, 0.1, 0.05, 0.01} s` y el error en norma infinita sobre el
estado `[x, y]`. Los órdenes observados son:

| h [s] | error Euler | error RK4 | orden Euler | orden RK4 |
|---:|---:|---:|---:|---:|
| 2.00 | 1.7967e+01 | 2.9485e-04 | — | — |
| 1.00 | 8.7150e+00 | 1.8035e-05 | 1.0437 | 4.0311 |
| 0.50 | 4.2952e+00 | 1.1148e-06 | 1.0208 | 4.0159 |
| 0.25 | 2.1321e+00 | 6.9292e-08 | 1.0105 | 4.0080 |
| 0.10 | 8.4916e-01 | 1.7681e-09 | 1.0047 | 4.0036 |
| 0.05 | 4.2397e-01 | 1.1033e-10 | 1.0021 | 4.0022 |
| 0.01 | 8.4697e-02 | 2.5011e-12 | 1.0007 | 2.3529 |

Euler muestra orden ≈ 1 y RK4 orden ≈ 4. El último punto de RK4 se acerca al
límite de redondeo de doble precisión (`~1e-12`), por eso su orden aparente
baja; no es un error del integrador. La figura `03_convergencia.png` incluye
rectas de referencia de pendiente 1 y 4.

> Nota: la norma infinita mezcla unidades de `x` (peticiones) e `y` (réplicas).
> En la práctica el error de `x` domina la norma. Se eligió así porque el
> enunciado pide un único error máximo del estado; en el README se deja
> explícita esta decisión.

---

## 7. Análisis de parámetros

Se mantiene `mu = 10` fijo.

- `tau` en `{5, 20, 60} s` con `K = 0.005`.
- `K` en `{0.0025, 0.005, 0.02}` con `tau = 20`.
- Rejilla 3×3 completa: 9 pares.

La tabla `resultados/tabla_parametros.csv` contiene `tau`, `K`, `wn`, `zeta`,
polos, régimen, sobrepaso, tiempo de pico, tiempo de asentamiento y periodo.

Casos destacados:

- `tau=5, K=0.005`: `zeta=1`, régimen crítico (sin oscilación).
- `tau=5, K=0.0025`: `zeta=1.414`, único caso sobreamortiguado de la rejilla.
- `tau=60, K=0.02`: `zeta=0.144`, respuesta muy oscilatoria (sobrepaso 269.7 %).

### Figuras generadas

La figura de respuesta en frecuencia usa los tres valores de `zeta` que
aparecen en el barrido de `K` (`0.707`, `0.500`, `0.250`), porque los tres
tienen un pico de resonancia interior con este modelo.

| Archivo | Contenido |
|---|---|
| `01_diagrama_bloques.png` | diagrama de bloques del lazo |
| `02_comparacion_analitica_numerica.png` | analítica vs Euler vs RK4 (escalón) |
| `03_convergencia.png` | error vs `h` en log-log con pendientes 1 y 4 |
| `04_retrato_fase.png` | retrato de fase `x` vs `y` |
| `05a_barrido_tau.png` | barrido de `tau` |
| `05b_barrido_K.png` | barrido de `K` |
| `06_rejilla_3x3.png` | 3×3 pares `(tau, K)` |
| `07_mapa_zeta.png` | mapa de `zeta` y zonas de amortiguamiento |
| `08_respuesta_frecuencia.png` | respuesta en frecuencia, resonancia y simulación sinusoidal |
| `09_polos.png` | polos en el plano complejo |
| `10_retardo_critico.png` | extensión: `theta_c` exacto vs Padé |
| `11_retardo_simulacion.png` | extensión: caso estable e inestable |

---

## 8. Extensión opcional: retardo puro

Se reemplaza la realimentación instantánea por

```text
tau*y' = -y + K*x(t - theta)
```

La ecuación característica es

```text
tau*s^2 + s + mu*K*exp(-s*theta) = 0.
```

En la frontera `s=j*omega`:

```text
mu*K*cos(omega*theta) = tau*omega^2
mu*K*sin(omega*theta) = omega
```

de donde se obtiene el retardo crítico exacto

```text
omega^2 = [-1 + sqrt(1 + 4*tau^2*(mu*K)^2)] / (2*tau^2)
theta_c = atan(1/(tau*omega)) / omega.
```

Para el caso base, `theta_c ≈ 23.012 s`.

La aproximación de Padé de primer orden,

```text
exp(-s*theta) ≈ (1 - theta*s/2)/(1 + theta*s/2),
```

conduce al polinomio cúbico

```text
(tau*theta/2)s^3 + (tau+theta/2)s^2
+ (1-mu*K*theta/2)s + mu*K = 0.
```

Aplicando Routh-Hurwitz se obtiene un `theta_c` aproximado. Para el caso base,
Padé predice `24.721 s` (7.4 % por encima del exacto). La simulación con
método de pasos y RK4 confirma que `theta=11.51 s` es estable y
`theta=34.52 s` es inestable.

La tabla `resultados/tabla_retardo.csv` recoge el retardo crítico exacto y
aproximado para varios valores de `tau` y `K`.

---

## 9. Pruebas

`tests/test_basico.py` verifica:

- condiciones iniciales y límite estacionario,
- residuo simbólico ≈ 0 en los tres regímenes,
- comparación con `solve_ivp` de tolerancia estricta,
- orden de convergencia ≈ 1 (Euler) y ≈ 4 (RK4),
- monotonía de `zeta` al aumentar `tau` o `K`,
- consistencia de la resonancia,
- métricas del caso base,
- retardo crítico y simulación estable/inestable de la extensión.

Resultado esperado con las dependencias obligatorias:

```text
17 passed
```

Si además está instalada la demo opcional (`requirements-app.txt`), se suman
tres pruebas de humo de Streamlit: `20 passed`.

---

## 10. Demo interactiva opcional (Streamlit)

`app_streamlit.py` es una aplicación interactiva pensada para la presentación
oral. **No forma parte de los entregables obligatorios ni de `python main.py`**:
no genera figuras PNG ni sustituye al informe.

Permite:

- mover `mu`, `K`, `tau`, `d0` y `T` con sliders;
- ver en vivo `wn`, `zeta`, el régimen, `x_ss` e `y_ss`;
- comparar la solución analítica con Euler y RK4 en la misma gráfica;
- cambiar entre carga escalón y sinusoidal (incluida la resonancia);
- consultar el plano de polos y las curvas `zeta(tau)` y `zeta(K)`.

Instalación y ejecución (desde `proyecto/`):

```bash
pip install -r requirements-app.txt
streamlit run app_streamlit.py
```

**Justificación de dependencias:** `streamlit` y `plotly` no están en la lista
de librerías permitidas para el núcleo del trabajo. Se usan exclusivamente en
esta demo opcional, viven en un archivo de requisitos separado y no
intervienen en `main.py`, en las figuras ni en las tablas. El código del
modelo, la analítica y los métodos numéricos es exactamente el mismo que usa
`main.py`; la app solo lo llama.

---

## 11. Decisiones de diseño y limitaciones

1. **Linealización:** el modelo es lineal y trabaja con desviaciones. No incluye
   no linealidades de colas (p. ej. M/M/1), saturaciones del autoescalador ni
   límites de réplicas. Es una aproximación de pequeña señal alrededor del
   equilibrio.
2. **Estabilidad:** con parámetros positivos el sistema lineal es siempre
   estable. La extensión con retardo es la que muestra inestabilidad real.
3. **Solución simbólica:** SymPy se usa para derivar y verificar; las gráficas
   usan fórmulas cerradas vectorizadas para evitar evaluar expresiones
   simbólicas en cada punto.
4. **Regímenes:** el caso crítico se detecta con tolerancia `1e-9` en `zeta`.
   `tau=5, K=0.005` cae exactamente en `zeta=1` salvo error de punto flotante.
5. **Error de convergencia:** norma infinita del estado completo, que mezcla
   unidades. Se documenta y se usa un único número porque así lo pide el
   enunciado.
6. **RK4 en `h=0.01`:** el error absoluto llega a `2.5e-12`, cerca del límite de
   redondeo; el orden aparente del último tramo no es representativo.
7. **Resonancia:** por el cero de `G(s)`, el umbral exacto de pico es
   `zeta < 0.7769`, no exactamente `1/sqrt(2)`.
8. **Métricas:** se calculan sobre la solución analítica muestreada y se
   refinan con `minimize_scalar`/`brentq`; son reproducibles y no dependen del
   paso de simulación.
9. **Retardo:** la simulación usa método de pasos con interpolación lineal de
   la historia y exige `h < theta`. Es una implementación docente, no un
   integrador de retardo de propósito general.
10. **Reproducibilidad:** no se usan números aleatorios ni rutas absolutas. Las
    figuras y tablas se sobrescriben en cada ejecución.
11. **Dependencias:** solo `numpy`, `scipy`, `matplotlib`, `sympy`, `pandas` y
    `pytest` (este último para pruebas). No se requieren librerías adicionales.

---

## 12. Referencias de apoyo (libros reales)

Estas referencias son material estándar de control y métodos numéricos; no se
inventaron para este proyecto. El informe escrito puede citarlas y ampliarlas
con las fuentes que el grupo consulte.

1. K. Ogata, *Modern Control Engineering*, 5th ed. Prentice Hall, 2010.
2. N. S. Nise, *Control Systems Engineering*, 6th ed. Wiley, 2011.
3. S. C. Chapra y R. P. Canale, *Numerical Methods for Engineers*, 7th ed.
   McGraw-Hill, 2015.
4. E. Hairer, S. P. Nørsett y G. Wanner, *Solving Ordinary Differential
   Equations I: Nonstiff Problems*, 2nd ed. Springer, 1993.
