# Simulador interactivo de sistemas de oscilación

Aplicación web (Streamlit) para resolver, comparar y visualizar sistemas oscilatorios
descritos por ecuaciones diferenciales ordinarias. Incluye el proyecto original de
**autoescalado en la nube** (`main.py`, `src/modelo.py`, `src/analitica.py`, …), que se conserva intacto.

## Ejecución

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py   # abre el navegador
python -m pytest -q              # pruebas
python main.py                   # (opcional) figuras del proyecto original
```

## Sistemas

| # | Sistema | Ecuación | Solución analítica |
|---|---|---|---|
| 1 | Masa-resorte simple | `m x'' + k x = 0` | sí |
| 2 | Masa-resorte amortiguado | `m x'' + c x' + k x = 0` | sí (3 regímenes) |
| 3 | Masa-resorte forzado | `m x'' + c x' + k x = F₀ cos ωt` | sí (transitorio + estacionario; resonancia pura) |
| 4 | Péndulo simple no lineal | `θ'' + (g/L) sin θ = 0` | no (se compara con la aproximación lineal y el período exacto con integral elíptica) |
| 5 | Dos masas acopladas | `M x'' + K x = 0` (pared–k₁–m₁–k₂–m₂–k₃–pared) | sí (modos normales) |
| 6 | Circuito RLC serie | `L q'' + R q' + q/C = V₀ cos ωt` | sí |

## Métodos numéricos
Euler explícito y RK4 (implementados a mano, genéricos para cualquier número de variables) y
`scipy.integrate.solve_ivp` (RK45, con tolerancias configurables). Cuando no hay solución analítica
se usa como referencia DOP853 con `rtol=1e-11`. El paso temporal nunca se modifica en silencio:
si `h` no divide a `T`, la app avisa; el máximo es 200 000 pasos.

## Estructura
```
app.py                       interfaz Streamlit (5 pestañas)
src/servicio.py              validación, integración con los 3 métodos, comparación
src/sistemas/                interfaz común (base.py) + un módulo por sistema
src/solucionadores/          Euler, RK4, solve_ivp genéricos
src/analitica_osc/lineal.py  soluciones cerradas del oscilador lineal
src/visualizaciones/         especificaciones de figuras + render Plotly
src/utilidades/              formato LaTeX
src/modelo.py, analitica.py, numerico.py, ...   proyecto original (sin cambios)
tests/test_osciladores.py    pruebas de los nuevos sistemas
tests/test_ui.py             prueba de la interfaz (requiere streamlit)
tests/test_basico.py         pruebas del proyecto original
```

## Interfaz
Selector «Seleccionar sistema de oscilación» → pestaña **Configuración** (formulario específico del
sistema, botón **Calcular**) → **Desarrollo matemático** (LaTeX con tus valores sustituidos) →
**Resultados** (magnitudes, interpretación, tabla) → **Gráficas** (Plotly) → **Comparación**
(errores de cada método vs. solución analítica/referencia, tiempos). Los resultados se guardan en
`st.session_state` y solo se recalculan al pulsar Calcular.

## Convenciones
`x`, `x₁`, `x₂` son desplazamientos desde el equilibrio. La energía del péndulo es por unidad de masa (J/kg).
La frecuencia de excitación se introduce en rad/s (la app muestra también Hz).
