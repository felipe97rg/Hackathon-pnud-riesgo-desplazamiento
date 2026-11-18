# Riesgo de desplazamiento por desastres en Colombia — 🥉 3er puesto, Hackathon PNUD Colombia 2026

> **English summary.** Monthly, municipality-level model that forecasts, for each of Colombia's 1,102 municipalities,
> the probability of a disaster event affecting people, how many people could be displaced, and an 80 % uncertainty
> range. Built by Juan Felipe Rodríguez (team *AI COLOMBIA*) for UNDP Colombia's *Predictive Models of Internal
> Displacement Risk due to Disasters* hackathon (Bogotá, October 2026), where it won **3rd place**.
> Four LightGBM models, leak-free temporal validation, public cartography and CHIRPS satellite rainfall.

**Mapa interactivo:** [felipe97rg.github.io/projects/undp-displacement-risk](https://felipe97rg.github.io/projects/undp-displacement-risk/)
(también en [`dashboard/index.html`](dashboard/index.html)).

> **Confidencialidad.** Los datos del reto **no** están en este repositorio y no se publican (T&C art. 9 y 13.1).
> Solo se incluyen datos abiertos (cartografía UPRA/DANE y lluvia CHIRPS) y las predicciones agregadas del mapa.

---

## El problema

Inundaciones, crecientes y deslizamientos obligan cada año a miles de personas en Colombia a dejar su casa. Si una
entidad supiera con anticipación **qué municipios y qué meses** tienen más riesgo, podría preparar albergues, ayuda y
alertas a tiempo.

Para cada municipio y cada mes entre octubre de 2022 y diciembre de 2025 (39 meses, 42,978 filas), el modelo entrega:

| Salida | Qué es |
|---|---|
| `prob_evento` | Probabilidad de al menos un evento climático con afectación (personas o viviendas) |
| `personas_desplazadas_estimadas` | Personas afectadas esperadas en el mes (incluye meses sin evento) |
| `personas_desplazadas_q10` / `q90` | Escenario bajo y alto: intervalo de incertidumbre del 80 % |

El entrenamiento usa enero de 2018 a septiembre de 2022. Dos dificultades: los eventos son **poco frecuentes** (solo el
7.9 % de los municipio-mes tiene uno) y los archivos de prueba **no traen el resultado ni rezagos**, así que hay que
predecir 39 meses sin saber qué pasó entre medio.

## Cómo se calificaba

| Componente | Métrica | Peso |
|---|---|---|
| Detección de riesgo | AUC-ROC sobre `prob_evento` | 50 % |
| Magnitud | RMSE en escala log1p | 20 % |
| Incertidumbre | Winkler del intervalo 80 % en log1p | 5 % |
| Replicabilidad | Tabla de variables y equivalentes globales | 5 % |
| Sustentación (finalistas) | Panel PNUD, UNGRD y academia | 20 % |

## Enfoque

1. **Calidad y tratamiento de datos:** diagnóstico de nulos, duplicados, reglas de dominio y outliers; imputación según
   la temporalidad de cada variable, marcando lo imputado.
2. **Historial municipal sin fuga de información:** frecuencia de eventos de cada municipio calculada solo con años
   anteriores a cada corte (y fija en septiembre de 2022 para la prueba), suavizada hacia las tasas del departamento y
   del país para municipios con pocos eventos.
3. **Contexto territorial:** cartografía de la UPRA (derivada del Marco Geoestadístico del DANE): ubicación, área,
   población e historial de los **5 municipios vecinos**. Aportó cerca de +0.7 puntos.
4. **Lluvia satelital CHIRPS:** lluvia del mes, acumulado de 3 meses y **anomalía** frente a lo normal de cada
   municipio, solo para los meses del reto. Fue la mayor mejora (**+3.6 puntos**); un placebo con la lluvia del año
   anterior no aportó nada (−0.07), así que la ganancia viene de la lluvia real del mes.
5. **Modelos:** cuatro LightGBM, uno por salida: clasificador de probabilidad, regresor sobre log1p(personas) y dos
   regresores cuantílicos (0.1 y 0.9). Con tantos ceros, el intervalo se ajusta para minimizar el Winkler.
6. **Validación temporal** con 3 cortes (entrenar con el pasado, evaluar en años no vistos) y una réplica de la nota
   oficial para comparar modelos.

### Selección del modelo

Se probaron **63 variantes**, todas registradas en [`outputs/reportes/registro_experimentos.csv`](outputs/reportes/registro_experimentos.csv).
Regla fijada de antemano: cambiar de modelo solo si mejora al menos **0.5 puntos** sin empeorar el corte más difícil
(ese era el ruido entre corridas casi idénticas). Un ensamble con CatBoost sumaba +0.25 puntos, dentro del ruido y con
más del doble de piezas, así que se entregó el modelo simple: más reproducible y fácil de explicar.

**Modelo entregado — E10S_LL:** 4 LightGBM · 44 variables (historial, territorio, cartografía, lluvia, ONI y calendario)
· 100 árboles, 15 hojas, ≥ 200 filas por hoja, tasa 0.05, 80 % de filas y variables, L2 = 1, semilla 42.
Las variables más importantes: lluvia del mes, población e historial del municipio.

## Resultados

| | Primera entrega (V1) | **E10S_LL (final)** |
|---|---|---|
| Puntaje de validación (réplica de la nota, máx. 75) | 20.06 | **25.56** |
| AUC en la prueba en vivo | 0.759 | **0.780** |
| RMSE log1p en vivo | 1.172 | **1.157** |
| Winkler en vivo | 2.70 | **2.57** |

Evolución en validación: 20.1 (historial) → 22.0 (+ cartografía) → **25.6 (+ lluvia)**. Mejoró en las tres métricas y
en los tres cortes. Resultado del concurso: **tercer puesto**.

## Estructura del repositorio

```
├── README.md
├── DECLARACION_IA.md            ← declaración de uso de IA generativa (T&C 9.1)
├── requirements.txt             ← dependencias del pipeline
├── requirements-externos.txt    ← dependencias para precomputar datos externos
├── src/hackaton/                ← paquete: datos, calidad, tratamiento, climatología, features, modelos, evaluación, entrega
├── scripts/                     ← generar datos sintéticos, explorar, entrenar, geovisor y precómputo de datos externos
├── tests/                       ← pruebas automáticas (pytest)
├── notebooks/
│   ├── notebook_entrega_E10S_LL_AICOLOMBIA.ipynb  ← notebook final entregado (modelo E10S_LL)
│   ├── notebook_experimentos*_AICOLOMBIA.ipynb   ← rondas de experimentos 1 a 5
│   ├── 01_exploracion.ipynb                      ← exploración y tratamiento interactivos
│   └── 02_pipeline_colab.ipynb                   ← pipeline reproducible en Colab
├── data/
│   ├── reto/                    ← CSV oficiales del reto (NO incluidos)
│   └── externos/procesados/     ← cartografía UPRA/DANE y lluvia CHIRPS por municipio
├── dashboard/index.html         ← mapa y tablero interactivo autocontenido (predicciones agregadas)
├── geovisor/                    ← plantilla del geovisor HTML
├── docs/
│   ├── decisiones_tecnicas.md   ← decisiones de ciencia de datos, alternativas y justificación
│   ├── explicacion_notebook_entrega.md
│   └── tabla_replicabilidad.md  ← variables y sus equivalentes abiertos y globales
└── outputs/reportes/            ← registro de experimentos y entregas, árbol del clasificador
```

## Reproducir

El código corre sin los datos del reto usando **datos sintéticos** con el mismo esquema:

```bash
git clone https://github.com/felipe97rg/Hackathon-pnud-riesgo-desplazamiento.git
cd Hackathon-pnud-riesgo-desplazamiento
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python scripts/generar_datos_sinteticos.py
python -m pytest -q
python scripts/entrenar.py --solucion A --datos sinteticos
```

Con los datos oficiales (solo para quienes los tengan autorizados): copiar `entrenamiento.csv`, `prueba_equipos.csv` y
`prueba_oculta.csv` en `data/reto/` o, en los notebooks, reemplazar `PEGUE_AQUI_EL_ENLACE_DE_LOS_DATOS` por el enlace
entregado por los organizadores.

## Limitaciones

- Usa la lluvia **observada**; para una alerta temprana real habría que usar pronósticos de lluvia o el dato
  preliminar de CHIRPS (publicado a los dos días).
- Estimar **cuántas** personas se desplazan sigue siendo difícil; el escenario alto (q90) es el más útil para planear.
- Las ciudades grandes aparecen arriba por exposición: es la probabilidad de que ocurra y se reporte un evento, no su
  gravedad por habitante.
- **Replicabilidad:** cada variable tiene un equivalente abierto y global (CHIRPS, WorldPop, geoBoundaries, DesInventar),
  así que el enfoque se puede llevar a otros países.

## Uso de IA generativa

Se usó Claude Code (Anthropic) como asistente para código y documentación, declarado según las reglas del reto
([`DECLARACION_IA.md`](DECLARACION_IA.md)). Las decisiones de qué probar, cómo validar y qué modelo entregar las tomó
el autor con base en la evidencia.

## Fuentes de datos

- **Reto (no incluido):** UNGRD (emergencias), TerriData/DNP (covariables), Copernicus DEM GLO-90 + MGN/DANE (relieve),
  NOAA CPC (ONI). Entregado por el PNUD; uso restringido al hackathon.
- **Cartografía municipal:** capa "municipios_generalizada" de la UPRA, derivada del Marco Geoestadístico Nacional del
  DANE (datos abiertos del Gobierno de Colombia).
- **Lluvia:** CHIRPS v2.0 mensual (Climate Hazards Center, UC Santa Barbara; Funk et al., 2015, *Scientific Data*
  2:150066), solo 2018-01 → 2025-12.

## Librerías

pandas, NumPy, SciPy, scikit-learn, Shapely (BSD-3-Clause) · LightGBM, Plotly, gdown, pytest (MIT) ·
CatBoost, XGBoost (Apache-2.0, solo experimentos) · rasterio (BSD-3-Clause). El geovisor y el tablero usan HTML, CSS y
JavaScript propios, sin librerías de terceros.

## Derechos

Trabajo finalista del Hackathon PNUD Colombia 2026. Según los términos del reto, el PNUD controla los derechos
patrimoniales y el autor conserva la autoría y puede mostrarlo en su portafolio con fines no comerciales. Por eso este
repositorio **no incluye una licencia de código abierto**: se publica como portafolio.

**Autor:** Juan Felipe Rodríguez Granados — [Portafolio](https://felipe97rg.github.io/) ·
[LinkedIn](https://www.linkedin.com/in/juan-felipe-rodriguez-granados-558673186)
