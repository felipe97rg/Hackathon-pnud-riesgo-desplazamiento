# Explicación del notebook de entrega — E10S_LL · equipo AI COLOMBIA

Notebook: `notebooks/notebook_entrega_E10S_LL_AICOLOMBIA.ipynb` · Integrante: Juan Felipe Rodríguez Granados ·
Hackathón PNUD Colombia, 6 de octubre de 2026.

Este documento explica, sección por sección, qué hace el notebook, por qué se tomó cada decisión y qué significa cada
término técnico (hay un **glosario** al final). Todas las cifras son agregadas: no se reproduce ninguna fila del dataset.

---

## 0. Resumen en un minuto

**Qué predice.** Para cada municipio (1,102) y cada mes de prueba (39 meses, de 2022-10 a 2025-12):

| Columna del CSV | Significado |
|---|---|
| `prob_evento` | Probabilidad de que ocurra un evento con afectación ese mes |
| `personas_desplazadas_estimadas` | Número esperado de personas desplazadas |
| `personas_desplazadas_q10`, `personas_desplazadas_q90` | Intervalo: valores bajo los cuales se espera el 10 % y el 90 % de los casos |

**Cómo.** Cuatro modelos LightGBM sencillos que combinan cuatro tipos de información:

1. **Historial** de eventos del municipio, del departamento y del país (solo del pasado).
2. **Territorio**: población, ingresos, inversión en gestión del riesgo, NBI, relieve (dataset del reto).
3. **Contexto espacial**: ubicación, área, población y el historial de los 5 municipios más cercanos (cartografía UPRA).
4. **Clima del mes**: ONI (El Niño / La Niña) y **lluvia observada** de CHIRPS.

**Resultado (validación temporal, misma fórmula de la parte automática, máx. 75 puntos):**

| Versión | Puntaje de validación | Qué cambió |
|---|---|---|
| V1 (plantilla + historial + LightGBM) | 20.06 | Punto de partida |
| E10S | 21.98 | Modelo más simple + contexto espacial |
| **E10S_LL (entregado)** | **25.56** | + lluvia observada |

En el corte F2, que es el mismo que usa el notebook base de los organizadores: modelo base 18.6 → V1 24.07 → E10S
24.61 → **E10S_LL 27.19**.

---

## 1. El reto en una página

* **Unidad de análisis:** municipio-mes. Entrenamiento 2018-01 → 2022-09 (57 meses × 1,102 = 62,814 filas).
  Prueba en vivo 2022-10 → 2024-04 (20,938 filas, alimenta la tabla de posiciones). Prueba oculta 2024-05 → 2025-12
  (22,040 filas, nota oficial al cierre). Un solo CSV con las 42,978 filas de prueba.
* **Los archivos de prueba no traen el objetivo ni el detalle de eventos.** Por eso el historial se construye desde el
  entrenamiento y se "congela" (ver §4).
* **Calificación (100 puntos):** AUC 50 · RMSE en log 20 · Winkler 5 (estos tres = parte automática, máx. 75) ·
  replicabilidad 5 · panel 20.
* **Tasa de eventos:** 7.9 % de los municipio-mes del entrenamiento; sube de 6.0 % (2018) a 11.5 % (2022), en parte por
  mejor reporte. Cuando hay evento, la mediana es de 36 personas.

---

## 2. Recorrido del notebook, sección por sección

### Celda de configuración
`INCLUIR_LLUVIA = True` es la decisión del equipo (modelo E10S_LL). Con `False`, el mismo notebook entrega **E10S**
(sin lluvia). Así hay un plan B de una sola línea si los organizadores no permitieran la lluvia de los meses de prueba.

### 1. Cargar los datos
Igual que la plantilla oficial: descarga `hackathon_datos.zip` desde el enlace de Drive (`URL_DATOS`) con `gdown` y
lee los tres CSV. Único ajuste: se extrae el identificador del enlace (`/d/<id>`) y se usa
`https://drive.google.com/uc?id=<id>`, que funciona con cualquier versión de `gdown` (la local no aceptaba `fuzzy`).

### 2. Explorar y tratar
* **Explorar:** celda original de la plantilla (diccionario, prevalencia, eventos por año, distribución de personas).
* **Calidad:** compara las filas con las esperadas (62,814 / 20,938 / 22,040 ✅), cuenta municipios (1,102 en los
  tres), llaves repetidas (0) y celdas nulas, y verifica cuánto ONI de prueba supera el máximo del entrenamiento (los
  árboles no extrapolan: asignan el valor del último tramo conocido).
* **Tratamiento** (`preparar`):

| Paso | Qué hace | Por qué | Resultado con los datos reales |
|---|---|---|---|
| Normalizar llaves | `DIVIPOLA` como texto de 5 dígitos; `anio` y `mes` desde `fecha` | Que los cruces no fallen por ceros a la izquierda | — |
| Duplicados | Quita filas idénticas y, si una llave municipio-fecha se repite, conserva la más completa | Una fila por municipio-mes | 0 duplicados |
| Objetivo según la definición oficial | `tiene_evento = 1` si hay personas desplazadas o viviendas destruidas/averiadas; personas negativas → 0 | Coherencia entre las dos variables objetivo | 0 filas corregidas |
| Valores imposibles → nulo | Rangos físicos: porcentajes 0–100, pendiente 0–90°, elevación −50 a 6,000 m, ONI ±3.5, montos ≥ 0 | Un error de captura no debe parecer un dato real | — |
| Imputación por temporalidad | ONI: mediana de la fecha. Estáticas: mediana del municipio. Anuales: mediana del municipio-año, luego el valor más cercano en el tiempo del mismo municipio; si no hay, mediana del departamento(-año) y por último nacional | Usar primero la información más parecida (mismo municipio, años vecinos) | Imputados: densidad 12, ingresos 36, inversión 96, NBI 96 |
| Banderas `_imputado` | Columna 0/1 por cada variable imputada | El modelo puede aprender si "el dato faltaba" significa algo | 4 banderas |
| Prueba sin objetivo | Se quitan de los archivos de prueba las columnas del objetivo y del detalle de eventos | Evitar usar por error algo que no existe en prueba | — |

### 3. Historial de cada municipio (sin fuga de información)
Es el predictor más fuerte. Detalle en §4.

### 3b. Cartografía municipal
* **Fuente:** capa pública "municipios_generalizada" de la UPRA, derivada del Marco Geoestadístico Nacional (MGN) del
  DANE. Se usa la UPRA porque el geoportal del DANE bloquea las consultas automáticas (no se intentó saltar el
  bloqueo). Los organizadores permitieron la cartografía pública si se declara.
* **Carga robusta:** caché local `cartografia_municipios.geojson` → si no existe, descarga del servicio (la
  simplificación de los límites la hace el servidor) → si falla, `geografia.csv`.
* **Cálculos:** centroide del polígono más grande y área en km² con una proyección local (1° de latitud = 110.574 km;
  1° de longitud = 111.320 km × cos(latitud)).
* **Validaciones que deben pasar:** al menos 1,100 municipios; códigos únicos de 5 dígitos; centroides dentro de
  Colombia; áreas positivas; área de Bogotá ≈ 1,636 km² (±5 %). Informativas: población de Bogotá = densidad × área ≈
  7.8 millones ✅; cubre los 1,102 municipios del reto ✅. Si una validación bloqueante falla, el notebook sigue sin
  variables espaciales.

### Vecindad
Para cada municipio, los **5 municipios con el centroide más cercano** (distancia sobre la esfera, fórmula de
haversine). Se promedia su tasa histórica (global y del mes del año), calculada **con la misma historia que la fila**:
si la fila es de 2021, la tasa de los vecinos usa solo 2018–2020. En la ronda 3 se probaron 3 y 10 vecinos, vecinos
ponderados por distancia y vecinos que comparten frontera: ninguno mejoró por encima del ruido.

### 3c. Lluvia mensual por municipio (CHIRPS)
Detalle en §5.

### 4. Variables y modelo
44 variables (§6) y 4 modelos LightGBM (§7).

### 5. Validación temporal
Tres cortes, métricas oficiales y elección del factor de `q90` (§8).

### 6. Entrenar con todo y generar la entrega
* **Filas de entrenamiento del modelo final:** 2019-01 → 2022-09 (45 meses × 1,102), cada año con el historial de los
  años anteriores. 2018 es el año de **calentamiento** (§4).
* **Filas de prueba:** las 42,978, con el historial calculado con **todo** el entrenamiento (congelado en 2022-09) y la
  lluvia observada de cada mes.
* **Paso a columnas de entrega** (`a_entrega`):
  * `prob_evento` = probabilidad del clasificador.
  * `personas_desplazadas_estimadas` = expm1(predicción de la regresión en log).
  * `q10` = expm1(mín(cuantil 0.1, cuantil 0.9 × factor)), `q90` = expm1(máx(…)): garantiza `q10 ≤ q90`.
* **Importancia de variables:** la lluvia suma el 28.6 % de la ganancia del clasificador; `lluvia_log` es la variable
  más importante (23.6 %), seguida de la población, la tasa histórica del municipio y la tasa del mes.

### Revisar el formato y guardar
`validar_entrega` es **la función original de la plantilla**, con las mismas reglas que el calificador: columnas,
llaves completas y sin duplicados, valores numéricos, `prob_evento` entre 0 y 1, personas ≥ 0, `q10 ≤ q90`. Resultado:
✅ 42,978 filas. Se guarda como `entrega_AICOLOMBIA.csv` (fecha en formato `AAAA-MM-DD`).

### 7. Otros entregables
Tabla de replicabilidad, geovisor, declaración de uso de IA y créditos (también en `docs/tabla_replicabilidad.md` y
`DECLARACION_IA.md`).

---

## 3. Por qué este enfoque (y no otro)

**Tabla con relaciones no lineales → gradient boosting.** El riesgo depende de combinaciones: mucha lluvia + municipio
con historial + ladera pesa mucho más que cada factor por separado. Los árboles captan esas combinaciones sin
programarlas, manejan vacíos y valores sesgados, y son rápidos y deterministas.

**Por qué no series de tiempo ni redes recurrentes.** En prueba **no hay rezagos del objetivo**: no se conoce lo que pasó
el mes anterior en el periodo de prueba, así que un modelo autorregresivo no tiene con qué alimentarse. Además hay solo
57 meses, el 92 % de los municipio-mes sin evento y el modelo debe ser explicable. La estructura temporal entra por
otra vía: historial por mes del año, tendencia reciente, ONI y lluvia.

**Por qué el estimado sale de una regresión sobre log1p(personas).** La métrica es el RMSE en escala log1p, cuyo
predictor óptimo es la media de log1p(personas). La regresión de error cuadrático sobre log1p(y) estima exactamente
eso, con los ceros incluidos. Por eso no se usó regresión Tweedie: modela la media de las personas, no la media del
logaritmo.

**Por qué cuantiles 0.1 y 0.9.** El Winkler con α = 0.2 equivale (× 10) a la pérdida pinball de los cuantiles 0.1 y
0.9, así que la regresión por cuantiles es la herramienta correcta. Con tantos ceros, el intervalo óptimo cubre cerca
del 90 % de los casos (no el 80 %), y el factor de `q90` corrige lo que haga falta.

---

## 4. El historial: calentamiento, congelado y suavizado

### Sin fuga de información
**Fuga** (*leakage*) es usar, al entrenar, información que no existiría al momento de predecir. El modelo parecería
excelente en validación y fallaría en la prueba. Reglas aplicadas:

* Las filas del año **Y** solo ven el historial de los años **anteriores a Y** (bloques por año completo).
* Las filas de prueba ven el historial de **todo** el entrenamiento, congelado en 2022-09: el mismo valor para los 39
  meses de prueba (no se actualiza, porque en prueba no conocemos los eventos).
* La tasa de los vecinos usa la misma historia que la fila.

### Calentamiento
**2018 no se usa como filas de entrenamiento**, solo como historia. Sus filas no tienen años anteriores: todas sus
tasas serían el promedio nacional y el modelo aprendería con filas sin información. 2018 "calienta" el historial para
que las filas de 2019 en adelante tengan algo que mirar. (El notebook base hace lo mismo con `anio >= 2019`.)

### Historia por años completos (no mes a mes)
Las filas de 2021, por ejemplo, ven 2018–2020 completos, aunque sean de diciembre. Así, entre el final de la historia y
la fila hay entre 1 y 12 meses, algo parecido a lo que pasa en prueba, donde la historia termina en 2022-09 y se usa
para meses hasta 39 meses después.

### Suavizado jerárquico (encogimiento)
Un municipio con 1 evento en 4 noviembres tiene una tasa "cruda" de 25 % en noviembre, pero con 4 observaciones eso es
casi ruido. El suavizado mezcla la tasa propia con una tasa previa más estable:

```
tasa suavizada = (eventos + K × tasa previa) / (meses observados + K)
```

**K** funciona como un número de "meses ficticios" con la tasa previa. Con pocos datos manda la previa; con muchos,
los datos propios. Ejemplo: 1 evento en 4 meses, previa 10 %, K = 3 → (1 + 0.3) / (4 + 3) = 18.6 % (en lugar de 25 %).

| Variable | Qué es | Tasa previa | K |
|---|---|---|---|
| `hist_tasa_nac_mes` | Tasa nacional en ese mes del año | Tasa nacional | 1,102 |
| `hist_tasa_dep` | Tasa del departamento | Tasa nacional | 120 |
| `hist_tasa_dep_mes` | Departamento en ese mes del año | Tasa del departamento × estacionalidad nacional | 60 |
| `hist_tasa_mun` | Tasa del municipio | Tasa del departamento | 24 |
| `hist_tasa_mes` | Municipio en ese mes del año | Tasa del municipio × estacionalidad del departamento | 3 |
| `hist_mag_mun` | Tamaño típico (log1p de personas) cuando hay evento | Departamento (K = 5 hacia el nacional) | 3 |
| `hist_esperado_log` | `hist_tasa_mes × hist_mag_mun`: el log1p de personas esperado según la historia | — | — |
| `hist_pct_inund`, `hist_pct_masa` | Proporción de inundaciones / movimientos en masa, suavizada hacia 1/3: (n + 1) / (total + 3) | 1/3 | — |
| `hist_reciente_12` | Tasa del municipio en los últimos 12 meses de la historia | Tasa nacional si no hay datos | — |

La "estacionalidad" de la previa es una razón: por ejemplo, tasa nacional de noviembre ÷ tasa nacional. Así, un
municipio hereda la forma estacional de su departamento, pero a su propio nivel.

---

## 5. La lluvia: de satélite a variable del modelo

1. **Fuente.** CHIRPS v2.0 mensual (Climate Hazards Center, UC Santa Barbara; Funk et al., 2015): estima la lluvia con
   la temperatura del tope de las nubes (satélite infrarrojo) y la corrige con estaciones, en una grilla de 0.05°
   (~5.5 km). Es abierta y casi global (50° S – 50° N).
2. **Solo los meses del reto.** 2018-01 → 2025-12 (96 meses). Nada anterior ni posterior, por la regla de los
   organizadores.
3. **Extracción eficiente.** Cada archivo mensual es global (26 MB) y está en formato COG, que permite leer solo un
   recuadro. Se lee la ventana de Colombia (357 × 302 pixeles) de los 96 meses: unos 40 segundos.
4. **De pixeles a municipios.** Promedio de los pixeles cuyo centro cae dentro del polígono municipal (mediana: 9
   pixeles por municipio). Los 5 municipios en los que no cae ningún centro toman el pixel de su punto interior. Se
   redondea a 2 decimales para que el archivo y la descarga den exactamente lo mismo.
5. **Validación.** Solo meses del reto, sin duplicados, de 0 a 3,000 mm, 96 meses por municipio y **cobertura de todas
   las filas municipio-mes del reto**. Controles de sentido: dos temporadas de lluvia (abril–mayo y octubre–noviembre)
   y 2022 como el año más lluvioso (La Niña). Si algo falla, el notebook sigue sin lluvia (E10S).
6. **Transformaciones:**

| Variable | Cálculo | Intuición |
|---|---|---|
| `lluvia_log` | log(1 + mm del mes) | Nivel de lluvia. El logaritmo hace que 50 → 150 mm pese más que 1,500 → 1,600 mm |
| `lluvia_3m_log` | log(1 + mm del mes + 2 meses anteriores) | Suelo saturado: deslizamientos y crecientes responden a lluvia acumulada |
| `lluvia_anom` | `lluvia_log` − climatología del municipio en ese mes del año | ¿Llovió más o menos de lo normal para ese lugar y esa época? +0.4 ≈ 1.5 veces lo normal |
| `lluvia_3m_anom` | Igual, con el acumulado de 3 meses | Periodos húmedos o secos persistentes |
| `lluvia_anom_rezago1` | Anomalía del mes anterior | Lluvia antecedente |
| `lluvia_anom_x_tasa` | Anomalía × `hist_tasa_mes` | La misma lluvia extra pesa más donde suele haber eventos. Ayuda a un modelo de solo 100 árboles |

* **Climatología:** promedio del log de la lluvia del mismo municipio y mismo mes del año, calculado **solo con
  2018-01 → 2022-09** (periodo de entrenamiento). No se recalcula con los meses de prueba.
* **Primeros meses:** 2018-01 y 2018-02 no tienen acumulado de 3 meses (no se usa 2017) → quedan vacíos; LightGBM
  los maneja.
* **En prueba:** cada fila recibe la lluvia ya observada de su mes, igual que el ONI que traen los archivos del reto.
  La lluvia no contiene información de emergencias.
* **Distribución:** la prueba en vivo tiene en promedio lluvia normal, pero con extremos (final de La Niña 2022 y El
  Niño 2023–24). La oculta es más lluviosa (≈ 13 % sobre lo normal). Solo el 0.3–0.4 % de sus valores queda fuera del
  rango visto en el entrenamiento: no hay extrapolación importante.

**Uso condicionado a las reglas:** la lluvia se limita a los meses del reto. Si los organizadores no la permitieran,
`INCLUIR_LLUVIA = False` entrega E10S.

---

## 6. Las 44 variables

| Grupo | Variables | Nº |
|---|---|---|
| Territorio (reto) | `log_densidad_poblacional`, `log_ingresos_tributarios`, `log_inversion_gestion_riesgo`, `ingresos_rango_anio` (percentil de ingresos dentro del año: quita la inflación), `nbi_pct`, `elevacion_media`, `pendiente_media`, `pct_pendiente_alta` | 8 |
| Banderas de imputación | `densidad_poblacional_imputado`, `ingresos_tributarios_imputado`, `inversion_gestion_riesgo_imputado`, `nbi_pct_imputado` | 4 |
| Clima global | `ONI`, `ONI_rezago_1`, `ONI_rezago_2`, `ONI_rezago_3` | 4 |
| Calendario y lugar | `mes`, `mes_sin`, `mes_cos` (diciembre queda junto a enero), `departamento` | 4 |
| Historial | Las 10 de §4 | 10 |
| Interacciones | `oni_x_inundacion` (ONI × proporción de inundaciones), `oni_x_tasa` (ONI × tasa del municipio) | 2 |
| Espaciales (UPRA) | `lat`, `lon`, `log_area`, `log_poblacion` (densidad × área), `hist_vecinos_tasa`, `hist_vecinos_tasa_mes` | 6 |
| Lluvia (CHIRPS) | Las 6 de §5 | 6 |

Los montos (densidad, ingresos, inversión, área, población) van en logaritmo porque son muy sesgados: unos pocos
municipios grandes no deben dominar los cortes de los árboles.

---

## 7. Los modelos y su configuración

### Gradient boosting en una frase
Se construyen muchos **árboles de decisión** pequeños, uno tras otro. Cada árbol nuevo corrige los errores que dejaron
los anteriores, y su aporte se multiplica por la **tasa de aprendizaje** para que ninguno pese demasiado. La
predicción es la suma de todos.

### Los 4 modelos
| Modelo | Objetivo (`objective`) | Aprende | Produce | Métrica que atiende |
|---|---|---|---|---|
| Clasificador | `binary` | ¿Hay evento? | `prob_evento` | AUC (50 %) |
| Regresión | `regression` (error cuadrático) | log1p(personas), en todas las filas | Estimado = expm1(predicción) | RMSE log (20 %) |
| Cuantil 0.1 | `quantile`, α = 0.1 | Percentil 10 de log1p(personas) | `q10` | Winkler (5 %) |
| Cuantil 0.9 | `quantile`, α = 0.9 | Percentil 90 | `q90` (× factor) | Winkler (5 %) |

Se entrenan **por separado** con las mismas variables. Por eso en la ronda 5 se pudo medir dónde aporta la lluvia
combinando componentes: en el clasificador suma +3.2 puntos y en magnitud y cuantiles, +0.35.

### Hiperparámetros
| Hiperparámetro | Valor | Qué controla | Por qué este valor |
|---|---|---|---|
| Árboles del clasificador (`n_estimators`) | **100, fijos** | Cuántas correcciones se suman | Con 200–800 árboles el modelo memorizaba el ruido de cada año (ronda 1). Con lluvia, 50 y 150 rindieron peor (ronda 5) |
| `learning_rate` | 0.05 | Cuánto aporta cada árbol | Aprendizaje gradual y estable |
| `num_leaves` | **15** | Grupos finales por árbol (complejidad) | Árboles pequeños = reglas generales |
| `min_child_samples` | **200** | Filas mínimas por grupo final | Impide reglas basadas en 3 o 4 casos raros; los eventos son pocos (7.9 %) |
| `colsample_bytree` | 0.8 | Cada árbol ve el 80 % de las variables | Diversidad entre árboles, menos sobreajuste |
| `subsample` / `subsample_freq` | 0.8 / 1 | Cada árbol ve el 80 % de las filas | Igual que el anterior |
| `reg_lambda` | 1 | Penalización L2 de los valores de las hojas | Evita predicciones extremas |
| `random_state` | 42 | Semilla | Reproducibilidad |
| `deterministic`, `force_row_wise` | True | Orden fijo de los cálculos | Mismo resultado en cada corrida |

**Regresión y cuantiles:** misma regularización (15 hojas, mínimo 200, tasa 0.05). El número de árboles se elige con
**early stopping**: dentro de la historia de entrenamiento, los **últimos 12 meses** sirven de chequeo interno. Se
entrenan hasta 3,000 árboles y se para si en 100 árboles seguidos no mejora (paciencia). El mejor número se multiplica
por 1.15 (al reentrenar se suman los 12 meses de chequeo, y con más datos conviene algún árbol más), con un mínimo de
50, y se reentrena con toda la historia. **No se usa el corte que se está evaluando** para elegirlo (eso sería fuga).

**Por qué el clasificador no usa early stopping:** la tasa de eventos sube cada año (efecto de reporte), y el chequeo
con los últimos 12 meses elegía números de árboles inestables. 100 árboles fijos fue más robusto.

---

## 8. Validación y métricas

### Cortes temporales
| Corte | Historia hasta | Valida | Para qué |
|---|---|---|---|
| F1 | 2020-09 | 2020-10 → 2021-09 | Predecir el año siguiente |
| F2 | 2021-09 | 2021-10 → 2022-09 | El corte del notebook base |
| F3 | 2020-09 | 2021-10 → 2022-09 | Dejar **un año de distancia**, como la prueba oculta (que llega a 39 meses) |

Se valida prediciendo meses que el modelo no vio, como en la evaluación real. No se usa validación cruzada aleatoria:
mezclaría meses futuros en el entrenamiento.

### Métricas oficiales (T&C 6.3)
| Métrica | Fórmula / idea | Normalización | Peso |
|---|---|---|---|
| **AUC** | Probabilidad de que un municipio-mes con evento reciba más `prob_evento` que uno sin evento. 0.5 = azar, 1 = perfecto. Solo importa el orden | (AUC − 0.5) / 0.5 | 50 |
| **RMSE log** | √promedio[(log1p(real) − log1p(estimado))²] sobre todas las filas, ceros incluidos | 1 − RMSE / RMSE de la predicción nula | 20 |
| **Winkler (α = 0.2)** | Ancho del intervalo + 10 × lo que el valor real quede por fuera (en log1p) | 1 − Winkler / Winkler de la predicción nula | 5 |

* **Predicción nula:** probabilidad 0.5, 0 personas e intervalo [0, 0]. Es la referencia contra la que se normaliza.
* **Puntaje de validación:** la misma fórmula de la parte automática, 100 × (0.50 · AUC normalizado + 0.20 · RMSE
  normalizado + 0.05 · Winkler normalizado), calculada en nuestros cortes. Máximo 75. Sirve para comparar versiones;
  la nota real la calculan los organizadores con la prueba oculta.
* **Regla práctica:** cada +0.01 de AUC vale cerca de +1 punto. El AUC es la palanca principal.

### Factor de `q90`
Para cada corte se prueba multiplicar el cuantil 0.9 (en log) por 0.7, 0.8, 0.9, 1.0, 1.1, 1.25 o 1.4 y se elige el
que minimiza el Winkler. El factor final es la **mediana** de los tres cortes (con E10S_LL: 1.25, 0.8 y 1.0 → **1.0**).

### Resultados de E10S_LL
| Corte | AUC | RMSE log | Winkler | Puntos |
|---|---|---|---|---|
| F1 | 0.741 | 1.336 | 3.324 | 26.52 |
| F2 | 0.751 | 1.254 | 3.162 | 27.19 |
| F3 | 0.710 | 1.256 | 3.267 | 22.96 |
| **Promedio** | **0.734** | **1.282** | **3.251** | **25.56** |

---

## 9. Cómo se llegó a E10S_LL: experimentos y decisiones

Todas las cifras están en `outputs/reportes/registro_experimentos.csv` (63 filas, rondas 1–5).

**Regla de decisión, fijada antes de ver los resultados:** cambiar de configuración solo si mejora **≥ 0.5 puntos** el
promedio **sin empeorar F3**. Motivo: el **ruido entre corridas** es de ±0.3 puntos por corte; diferencias menores no
son confiables.

| Ronda | Qué se probó | Hallazgo | Decisión |
|---|---|---|---|
| V1 | Plantilla + tratamiento + historial jerárquico + LightGBM con early stopping | 20.06 puntos | Primera entrega |
| 1 | 200/400/800 árboles fijos, magnitud en dos partes, modelo nacional, más regularización | Ninguna supera a V1; **menos árboles rinden mejor** | Ir hacia modelos más simples |
| 2 | 50/100 árboles, modelo simple, tasa 0.02, pesos por año, **contexto espacial**, ensamble | `E10_espacial_simple` 21.96 (+1.9) | Elegir E10S (el ensamble E9 empataba dentro del ruido: se prefirió el simple) |
| 3 | Ablación, definición de vecinos, más información de vecinos, complejidad, semillas | Población +0.40, ubicación +0.34, vecinos +0.12; el suavizado espacial resta 0.63; nada supera a E10S por ≥ 0.5 | E10S se mantiene |
| 4 | CatBoost, XGBoost y ensambles | Peores o iguales (CatBoost gana en F2 y pierde en F3) | E10S se mantiene |
| 5 | Mejores configuraciones + **lluvia CHIRPS**, placebo, combinaciones | **E10S_LL 25.56 (+3.57)**; placebo −0.07; solo anomalías +1.01; ensamble con CatBoost 25.81 (+0.25 sobre E10S_LL, dentro del ruido) | **E10S_LL** |

### Por qué creemos que la ganancia de la lluvia es real
* Mejora en los **tres cortes** (+4.8, +2.6 y +3.3), algo que nada había logrado antes.
* **Placebo:** las mismas variables pero con la lluvia del **mismo mes del año anterior** no mejoran (−0.07). Si la
  ganancia viniera de agregar columnas o de la estacionalidad, el placebo también habría mejorado.
* **Sin fuga del objetivo:** la lluvia no contiene información de emergencias.
* **Tiene sentido físico:** las inundaciones, crecientes y movimientos en masa los dispara la lluvia. La ganancia está
  casi toda en la probabilidad del evento (AUC de 0.701 a 0.734), no en cuántas personas.
* **El nivel aporta más que la anomalía** (+3.6 frente a +1.0). La hipótesis (no comprobada) es que el patrón normal de
  lluvia de cada municipio en cada mes es una señal estacional más suave y física que el historial de eventos, que es
  ruidoso (pocos eventos por municipio-mes).

### Por qué E10S_LL y no el ensamble
El ensamble (E10S_LL + E7_LL + CatBoost) suma +0.25: no cumple la regla de ≥ 0.5. Sus predicciones tienen una
correlación de orden de 0.994 con las de E10S_LL (92 % de coincidencia en el 10 % más riesgoso). Es casi el mismo
modelo con 9 piezas en vez de 4. Se eligió el simple: más reproducible y más fácil de explicar, con el mismo criterio
de la ronda 2.

### Alternativas descartadas
| Alternativa | Por qué no |
|---|---|
| Series de tiempo / redes recurrentes (LSTM) | En prueba no hay rezagos del objetivo; pocos meses; menos explicables |
| Regresión Tweedie para la magnitud | Optimiza la media de personas; la métrica premia la media del log1p |
| Modelo en dos partes (P × magnitud condicionada) | No mejoró en la ronda 1; la regresión directa sobre log1p ya es óptima para el RMSE log |
| Más árboles / árboles más grandes | Sobreajuste al ruido de cada año (rondas 1, 3 y 5) |
| CatBoost / XGBoost | No superaron a LightGBM simple (ronda 4); con lluvia empatan |
| Emergencias anteriores a 2018, índices climáticos externos | Prohibidos por la regla de fechas de los organizadores |
| Lluvia del IDEAM (estaciones) | La API no logra agregar ni un día; las estaciones son escasas fuera de la región Andina; serían necesarias interpolaciones con mucho error |

---

## 10. Reproducibilidad
* **Determinista:** misma semilla y cálculos en orden fijo. Las referencias reprodujeron exactamente los resultados de
  rondas anteriores (diferencia 0.00).
* **Mismo resultado por dos caminos:** con los archivos locales y con las descargas desde la fuente (UPRA y CHIRPS,
  como pasaría en Colab), el CSV es idéntico (diferencia máxima 0.0).
* **Autocontenido:** descarga los datos del reto, la cartografía y la lluvia; instala `rasterio` si falta. Se ejecuta
  en pocos minutos.
* **Trazabilidad:** cada experimento quedó en el registro central con su configuración, sus métricas por corte y la
  huella de los datos.

---

## 11. Limitaciones
* **Lluvia observada del mismo mes.** En una alerta temprana real habría que usar pronósticos (IDEAM, ECMWF) o el dato
  preliminar de CHIRPS (disponible ~2 días después del periodo).
* **CHIRPS subestima la lluvia de montaña y del Pacífico**, y el total mensual esconde los aguaceros de un día. Las
  anomalías (cada municipio comparado consigo mismo) reducen el primer problema.
* **La magnitud es el componente débil:** el RMSE apenas mejora a predecir 0 personas (1.28 frente a ~1.34). Hay poca
  señal mensual sobre cuántas personas se verán afectadas.
* **Tendencia de reporte:** la tasa de eventos sube de 6 % a 11.5 % entre 2018 y 2022, en parte por mejor registro; el
  historial congelado no la proyecta hacia adelante.
* **Historial congelado:** en prueba no se actualiza durante 39 meses; la lluvia es la parte que sí cambia mes a mes.

---

## 12. Preguntas probables del panel
**¿Por qué LightGBM?** Datos tabulares con interacciones no lineales, maneja vacíos, rápido, determinista y con
importancia de variables. Se compararon CatBoost y XGBoost y no fueron mejores.

**¿Cómo evitaron la fuga de información?** Historial solo de años anteriores, congelado en prueba; vecinos con la misma
historia; early stopping dentro de la historia; ningún dato de emergencias del periodo de prueba.

**¿Usar la lluvia del mismo mes no le da una ventaja indebida al modelo?** Es una covariable observada del mismo mes, igual que el ONI que entregan los
organizadores, y no contiene información de emergencias. Para operación habría que reemplazarla por pronósticos.

**¿Cómo saben que la mejora no es ruido?** Mejora en los tres cortes, supera por mucho el ruido de ±0.3, y el placebo no
mejora.

**¿Cómo se lleva a otro país?** Todas las variables tienen un proxy abierto y global (tabla de replicabilidad): CHIRPS
ya es global; población con WorldPop; límites con geoBoundaries; historial con DesInventar o EM-DAT.

**¿Qué hizo la IA y qué hizo el equipo?** La IA (Claude Code, Claude Opus 5.5) apoyó con código, documentación y
explicaciones. El integrante decidió qué probar, ejecutó los notebooks, verificó los resultados y eligió el modelo
con una regla fijada de antemano (ver `DECLARACION_IA.md`).

---

## 13. Glosario

| Término | Significado |
|---|---|
| **Ablación** | Quitar un grupo de variables para medir cuánto aportaba |
| **Anomalía (de lluvia)** | Diferencia entre la lluvia del mes y lo normal para ese municipio en ese mes del año (en logaritmo) |
| **Árbol de decisión** | Modelo que divide los datos con preguntas sucesivas ("¿lluvia > X?") hasta grupos finales (hojas) con una predicción |
| **AUC** | Área bajo la curva ROC: qué tan bien ordena el modelo los casos con evento por encima de los sin evento |
| **Bandera `_imputado`** | Columna 0/1 que indica que el valor original faltaba y se rellenó |
| **Calentamiento** | Periodo inicial (2018) que solo se usa como historia, no como filas de entrenamiento, porque no tiene años anteriores |
| **CHIRPS** | Climate Hazards Group InfraRed Precipitation with Station data: lluvia estimada por satélite y corregida con estaciones |
| **Climatología** | Valor "normal" de una variable para un lugar y una época, promediado sobre varios años |
| **COG** | Cloud Optimized GeoTIFF: imagen geográfica que permite leer solo un recuadro sin descargar todo el archivo |
| **Cobertura** | Porcentaje de casos reales que caen dentro del intervalo [q10, q90] |
| **Corte temporal** | División entrenamiento / validación por fechas: se entrena con el pasado y se valida con meses posteriores |
| **Cuantil** | Valor bajo el cual queda un porcentaje de los casos (q10: el 10 %; q90: el 90 %) |
| **Determinista** | Que con los mismos datos da exactamente el mismo resultado en cada corrida |
| **DIVIPOLA** | Código del DANE de 5 dígitos que identifica cada municipio (los 2 primeros = departamento) |
| **Early stopping** | Detener el entrenamiento cuando el error en un conjunto de chequeo deja de mejorar |
| **ENSO / ONI** | El Niño–Oscilación del Sur. El ONI es su índice: ≥ 0.5 El Niño (más seco en Colombia), ≤ −0.5 La Niña (más lluvioso) |
| **Ensamble** | Promedio de las predicciones de varios modelos |
| **expm1 / log1p** | log1p(x) = log(1 + x) (admite ceros); expm1 es su inversa: e^x − 1 |
| **Factor de q90** | Multiplicador del cuantil 0.9 que minimiza el Winkler en validación |
| **Fuga de información (leakage)** | Usar al entrenar información que no existiría al momento de predecir |
| **Gradient boosting** | Suma de muchos árboles pequeños entrenados en secuencia, cada uno corrigiendo los errores de los anteriores |
| **Haversine** | Fórmula de distancia entre dos puntos sobre la esfera terrestre |
| **Historial congelado** | Historial calculado con todo el entrenamiento y aplicado sin cambios a todos los meses de prueba |
| **Hoja** | Grupo final de un árbol, con su propia predicción |
| **Imputación** | Rellenar valores faltantes con una estimación razonable |
| **Importancia (ganancia)** | Cuánto mejoró el ajuste gracias a los cortes en cada variable, como % del total |
| **Interacción** | Variable que multiplica dos (p. ej. anomalía × tasa histórica) para que el efecto de una dependa de la otra |
| **K (pseudo-observaciones)** | En el suavizado, cuántos "meses ficticios" con la tasa previa se suman a los datos propios |
| **`learning_rate`** | Fracción de su corrección que aporta cada árbol nuevo |
| **MGN / UPRA** | Marco Geoestadístico Nacional del DANE; la UPRA publica una versión generalizada de los límites municipales |
| **`min_child_samples`** | Mínimo de filas que debe tener una hoja |
| **`num_leaves`** | Máximo de hojas por árbol |
| **Pinball (pérdida cuantil)** | Error asimétrico que se minimiza con el cuantil buscado; el Winkler es 10 × pinball de q10 y q90 |
| **Placebo** | Variante con la misma estructura pero sin la información real (lluvia de otro año); si mejora igual, la ganancia no es real |
| **Predicción nula** | Referencia sin modelo (probabilidad 0.5, 0 personas, intervalo [0, 0]) contra la que se normalizan las métricas |
| **Prueba en vivo / oculta** | En vivo: 2022-10 → 2024-04, alimenta la tabla de posiciones. Oculta: 2024-05 → 2025-12, nota oficial |
| **Puntaje de validación** | La fórmula de la parte automática (máx. 75) calculada en nuestros cortes |
| **Rezago** | Valor de una variable en un mes anterior (rezago 1 = el mes anterior) |
| **`reg_lambda`** | Penalización L2 que encoge los valores de las hojas |
| **Replicabilidad (proxy)** | Fuente abierta y global equivalente a cada variable, para aplicar el modelo en otro país |
| **RMSE log** | Raíz del error cuadrático medio en escala log1p |
| **Ruido entre corridas** | Variación de ±0.3 puntos por corte ante cambios mínimos; por debajo de eso, las diferencias no son concluyentes |
| **Semilla** | Número que fija el azar (submuestreo de filas y variables) para poder repetir el resultado |
| **`subsample` / `colsample_bytree`** | Fracción de filas / variables que ve cada árbol |
| **Suavizado jerárquico (encogimiento)** | Mezclar la tasa propia con una más estable (departamento, nación) según cuántos datos haya |
| **Tasa de eventos** | Proporción de meses con evento (con `tiene_evento = 1`) |
| **Winkler** | Puntaje de intervalos: ancho + penalización (10×) por lo que el valor real quede por fuera |
