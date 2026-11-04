# Diagnóstico Etapa 0 — Transformación en simulador interactivo

Fecha: 2026-11-04 · Directriz: **no reconstruir desde cero**; conservar lo existente.

---

## 1. Estado verificado hoy

| Comprobación | Resultado |
|---|---|
| `python main.py` | `PROCESO COMPLETADO`, 12 figuras + 3 tablas CSV |
| `pytest -q` | `20 passed` (17 núcleo + 3 humo Streamlit) |
| `compileall` | OK |
| Demo `app_streamlit.py` | Ejecutada en vivo (`streamlit run`), sin errores de consola |
| Entorno | `.venv` en `EcuacionesDiferenciales\` con deps pinneadas + streamlit 1.66.1 + plotly 7.1.0 |

## 2. Funcionalidades existentes y su reusabilidad

### Núcleo (entregable universitario — NO tocar)

- `main.py`: pipeline por fases (bloques, comparación, convergencia, barridos,
  rejilla 3×3, mapa ζ, respuesta frecuencia, polos, retardo). Valida entregables
  al final. **Se conserva intacto.**
- `src/modelo.py`: `Parametros` (dataclass inmutable), `con_parametros()`,
  `wn/zeta/regimen/x_ss/y_ss`, `matriz_A`, `valores_propios`,
  `raices_caracteristicas()`, `campo_vectorial(t, estado, p, entrada)`,
  `verificar_dimensiones()`.
  **Reutilizable**: modelo autoescalado completo como sistema #0 del simulador.
- `src/analitica.py`: Laplace simbólico, formas cerradas de escalón (3
  regímenes), solución sinusoidal completa, `respuesta_frecuencia`,
  `frecuencia_resonancia`, `amplitud_resonancia`, `metricas_escalon`
  (sobrepaso/t_pico/t_asent/periodo), `comparar_con_solve_ivp`,
  `residuo_simbolico`. **Reutilizable** tal cual para el sistema autoescalado.
- `src/numerico.py`: `euler_explicito`, `rk4` (input `entrada(t)` por etapa —
  clave para orden teórico), `resolver_solve_ivp` (DOP853 rtol 1e-10/atol 1e-12),
  `error_max_estado`, `_malla_temporal`.
  **Patrón reutilizable pero NO genérico**: están acoplados a `Parametros` del
  autoescalado vía `campo_vectorial`.
- `src/convergencia.py`: tabla y figura de convergencia (orden observado).
  **Patrón trasladable** a cada sistema con solución cerrada.
- `src/parametros.py`: barridos, rejilla, mapa ζ, frecuencia, polos (matplotlib).
  Se conserva; se complementa con versiones Plotly para la app.
- `src/extension_retardo.py`: método de pasos (RK4 con historia interpolada),
  Padé-1 + Routh-Hurwitz, tabla/figuras de retardo. **Se conserva tal cual**
  (DDE específica; no es el靶向 para el simulador genérico).

### Demo actual

- `app_streamlit.py`: demo de UN solo sistema (autoescalado) con sliders,
  tabs, Plotly, `paso_adecuado()` (estabilidad de Euler condicional),
  helpers de formato. **Prueba la stack y será absorbida o sustituida.**

### Pruebas

- `tests/test_basico.py` (17): CI/estado estacionario, residuo simbólico 3
  regímenes, solve_ivp, órdenes 1/4, monotonía ζ, resonancia, métricas, retardo.
- `tests/test_app_streamlit.py` (3): `AppTest` con guardas
  `pytest.importorskip`. Patrón reutilizable para la nueva app.

## 3. Limitaciones técnicas descubiertas (importantes para el diseño)

1. **Acoplamiento de los integradores**: `euler_explicito/rk4` firman
   `(p: Parametros, h, T, entrada)` y call internamente `campo_vectorial` del
   autoescalado. No sirven para otros sistemas sin romperlos.
   → Se crean solucionadores **genéricos** en un paquete nuevo; los actuales
   quedan intactos ll delegando (thin wrappers) a los genéricos.
2. **`_malla_temporal` exige h que divida T** (salvo fallback linspace). El
   usuario introducirá pasos arbitrarios → el generador del simulador debe
   aceptar T/h no divisibles (redondeo + ajuste explícito documentado).
3. **Rendimiento interactivo**: `metricas_escalon` usa malla de 40k+ puntos +
   `minimize_scalar/brentq`. Aceptable en Streamlit (≈ fragmentos de segundo),
   pero el simulador evitará recomputar sin botón “Calcular”.
4. **Consola Windows cp1252**: `print` con griegos rompe la consola (ya se
   normalizó a ASCII en main.py; mantener la convención en código nuevo).
5. **`use_container_width` deprecado** en streamlit 1.66 (default
   `width="stretch"`): no pasarlo explícitamente.
6. Ruido de cachés (`__pycache__/`, `.pytest_cache/`) y un `test_basico.pyc`
   de pytest 9.2.0 (residuo histórico). Cosmético; opcional limpiar + `.gitignore`.

## 4. Arquitectura propuesta (Etapa 1) — sin destruir `src/`

Paquete nuevo paralelo al entregable universitario:

```
proyecto/
├── app.py                       # PUNTO DE ENTRADA (python -m streamlit run app.py)
├── src/                         # sin cambios: entregable universitario
├── simulador/
│   ├── __init__.py
│   ├── interfaces.py            # dataclass-contrato + registro de sistemas
│   ├── solucionadores/
│   │   ├── euler_generico.py, rk4_generico.py, scipy_generico.py
│   ├── analitica/
│   │   ├── masa_resorte.py, amortiguado.py, forzado.py, pendulo.py,
│   │   │   acopladas.py, rlc.py, autoescalado.py (envoltorio de src/)
│   ├── sistemas/
│   │   ├── base.py + un módulo por sistema (parámetros, campo, analítica,
│   │   │   derivados, validaciones, datos de gráficas, interpretación)
│   ├── visualizaciones/
│   │   ├── plotly_comun.py (retrato fase, energía, comparación, tabla de error)
│   └── utilidades/
│       ├── enums.py, formatos.py, validar.py
```

Contrato común por sistema (interfaz única, sin duplicar lógica):

- `nombre`, `descripcion`, `ecuacion_latex`,
- `dataclass_parametros` (por sistema) + `validar()`,
- `campo(t, estado, params) -> derivadas`,
- `solucion_analitica(t, params)` o `None` (con flag “no hay cerrada”),
- `derivados(params, sol) -> métricas`,
- `desarrollo_matematico(params, derivados) -> (latex, texto interpretación)`,
- `condiciones_iniciales`, `estado_inicial`.

Registro: `SISTEMAS = {nombre → spec}`; la app itera el registro, nunca
`if sistema == "masa_resorte": ...` en cascada.

Botón **Calcular** + `st.session_state` para congelar resultados; widgets en
form; validación pre-cálculo con mensajes `st.error` específicos.

## 5. Archivos: veredicto por archivo

| Archivo | Acción |
|---|---|
| `src/*.py`, `main.py` | **Intactos** (entregable) |
| `app_streamlit.py` | Deprecar tras crear `app.py` (o absorberlo); pedir decisión |
| `tests/*` | Se conservan; se añaden `tests/sistemas/test_<sistema>.py` y humo de `app.py` |
| `README.md` | Se amplía (Etapa 12), conservando secciones del informe |
| `requirements.txt` | Decisión: ¿streamlit+plotly están en requirements.txt? (ver §6) |
| NUEVO | `app.py`, paquete `simulador/`, tests por sistema |

## 6. Decisiones que requieren confirmación del usuario (Etapa 1)

1. **Autoescalado como sistema adicional**: recomiendo integrarlo como
   sistema #7 del selector, reutilizando `src.analitica/numerico` al 100 %
   (cumple “reutiliza el existente”). Alternativa: mantenerlo como demo aparte.
2. **Dependencias**: recomiendo añadir `streamlit` y `plotly` a
   `requirements.txt` (la app pasa a ser punto de entrada oficial) y eliminar
   `requirements-app.txt`. Alternativa: mantener separación opcional.
3. **Ordene `app_streamlit.py`**: borrarlo al migrar a `app.py` (recomendado:
   evita dos aplicaciones divergentes) assuming (1) acepta la integración.
4. **Base del “tiempo real” del enunciado**: cálculos solo al pulsar “Calcular”
   (form), según el nuevo texto; el recalcular automático de la demo se abandona.

## 7. Riesgos y mitigaciones

- **No romper el entregable**: `main.py`/`src/` intocables; `pytest` después de
  cada etapa debe dar 20 passed como mínimo.
- **Exactitud > estética**: TODO número mostrado proviene del cálculo
  (prohibido hardcodear); la verificación numérica (analítica vs Euler/RK4/
  solve_ivp + errores) se muestra en pantalla.
- **Energía conservada/monotona decreciente**: test por sistema según el
  criterios precisos.
- **Metadata de etapa actual**: cada etapa termina con resumen + archivo de
  estado (“Estado actual, archivos modificados, próximos pasos”).

## 8. Plan de etapas (por la directriz del usuario)

Sigo exactamente el orden indicado por el usuario:
0 (diagnóstico) → 1 (arquitectura) → 2 (masa-resorte simple) → 3 (amortiguado)
→ 4 (forzado) → 5 (péndulo no lineal) → 6 (masas acopladas) → 7 (RLC serie)
→ 8 (interfaz integrada) → 9 (métodos/comparaciones) → 10 (desarrollo
matemático) → 11 (validaciones) → 12 (documentación/README).
