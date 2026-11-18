# Decisiones técnicas de ciencia de datos

Documento vivo con las decisiones técnicas del equipo: qué se decidió, por qué, qué alternativas se descartaron y en
qué estado está cada una. Complementa el [README](../README.md) (qué hay y cómo se usa) y sirve de base para el
documento de 2 páginas y la sustentación, si quedamos finalistas.

**Estados:** ✅ implementado y probado · 🟡 propuesto, se valida con los datos reales el día D · ⏳ por implementar ·
❓ depende de una respuesta de los organizadores.

Última actualización: 2026-10-06, día del evento. Las secciones 1–11 se escribieron antes de recibir los datos; la 12 y la
13 resumen los experimentos con los datos reales.

---

## Resumen

| # | Decisión | Estado |
|---|---|---|
| D1 | Modelar propensión municipal + moduladores temporales; no una serie de tiempo autorregresiva | ✅ diseño |
| D2 | Cada salida optimiza su propia métrica: orden (AUC), `E[log1p y]` (RMSE log), cuantiles en log1p (Winkler). Se elige con el puntaje de validación; los diagnósticos solo ayudan a entender | ✅ métricas y diagnósticos · ⏳ modelos |
| D3 | Validación con cortes temporales y encodings calculados solo con el pasado | ✅ cortes · 🟡 encodings |
| D4 | Tratamiento de datos: los imposibles pasan a nulo, se imputa según la temporalidad y los outliers se reportan sin eliminarse | ✅ |
| D5 | Climatología municipal con Bayes empírico (encogimiento hacia el departamento) | ⏳ |
| D6 | Solución A: LightGBM. Solución B: hurdle + ensamble + datos externos. Hiperparámetros: configuración base + búsqueda corta solo del clasificador | ⏳ · 🟡 plan |
| D7 | Datos externos: solo cartografía municipal; nada anterior ni posterior al periodo del reto (regla de los organizadores) | ✅ |
| D8 | Desarrollo con datos sintéticos, pipeline único, pruebas y bitácora de experimentos | ✅ |
| D9 | Geovisor HTML autocontenido | ✅ |

---

## 1. Formulación del problema (D1)

**Qué se predice.** Para cada municipio (1,102) y mes (2022-10 a 2025-12) hay tres tareas sobre el mismo panel:
clasificación (`prob_evento`), regresión de la magnitud (`personas_desplazadas_estimadas`) y regresión cuantílica
(`q10`, `q90`).

**Rasgos de los datos que condicionan todo:**
- **Objetivo con muchos ceros y muy asimétrico.** La mayoría de los municipio-mes no tienen evento y, cuando lo hay,
  la cantidad de personas tiene una cola larga.
- **La variable objetivo es "personas afectadas"** reportadas a la UNGRD, aunque el reto la llame "personas
  desplazadas". Además hay eventos sin personas: `tiene_evento = 1` también cuando solo hay viviendas afectadas.
  Esto importa en D2.
- **Los archivos de prueba no traen el objetivo ni sus rezagos,** y el horizonte va de 1 a 39 meses después del último
  dato de entrenamiento.

**Decisión.** El problema no se trata como una serie de tiempo autorregresiva. Se modela como:

> riesgo(municipio, mes) = **propensión estructural del municipio** (su historia, relieve, exposición, vulnerabilidad)
> × **moduladores temporales** (estacionalidad regional y fase ENSO)

La historia de cada municipio se resume en una "climatología" calculada con `entrenamiento.csv` y se aplica como
constante a todos los meses de prueba.

**Lo que implica el diccionario de datos:**
- **Casi todas las covariables son fijas por municipio.** Cuatro nunca cambian (NBI, elevación, pendiente, % de
  pendiente alta), tres son anuales (densidad, ingresos, inversión) y el ONI es igual para todo el país. El orden
  **entre municipios** lo darán la historia y el territorio; el orden **entre meses** solo puede venir del mes y el
  ONI (no se permite usar lluvia observada).
- **Algunos valores de prueba son repeticiones.** La inversión de 2021–2025 copia la de 2020 y los ingresos de 2025
  copian los del último año disponible. Cualquier "tendencia" en esos años es artificial.
- **`ingresos_tributarios` está en pesos corrientes:** sube con la inflación aunque el municipio no cambie. Se
  evaluará usar su posición relativa dentro de cada año (🟡).
- **El objetivo depende del reporte.** Que no haya un evento registrado en la UNGRD no prueba que no haya ocurrido, y
  los municipios con más capacidad institucional pueden reportar más. Ingresos e inversión podrían predecir "quién
  reporta" más que "quién está en riesgo": hay que decirlo al interpretar resultados.
- **`n_otros` incluye sequías.** El ENSO no actúa en un solo sentido: La Niña trae lluvias, pero El Niño también puede
  generar eventos con afectación.
- **El ONI es un promedio móvil centrado:** incluye el mes siguiente. En pronóstico real no se conocería, pero el reto
  lo entrega para los meses de prueba.

**Alternativas descartadas:**

| Alternativa | Por qué no |
|---|---|
| ARIMA, Prophet o similares por municipio | Requieren rezagos que no existen en prueba; 1,102 series cortas (57 meses) y con muchos ceros; horizonte de hasta 39 meses |
| Redes recurrentes o grafos espacio-temporales | Unas 63 mil filas y 4 horas de trabajo: alto riesgo de sobreajuste, costo de ajuste y poca explicabilidad para el panel |
| Modelo bayesiano jerárquico (Solución C) | Incertidumbre e interpretabilidad excelentes, pero el ajuste en Colab dentro de 4 horas es riesgoso. Se adopta su idea central (encogimiento hacia el grupo) en D5 |

---

## 2. De la métrica a la decisión (D2)

Puntaje automático (máximo 75): `50·AUC_n + 20·RMSE_n + 5·Winkler_n`. Nosotros aplicamos la misma fórmula sobre los
cortes temporales del entrenamiento: es el **puntaje de validación** (`src/hackaton/evaluacion.py::puntaje`), con
pruebas automáticas. No es la nota oficial (se calcula con `prueba_oculta`, que no vemos) ni predice nuestra posición:
sirve para comparar nuestros modelos entre sí.

### 2.1 Cuánto vale cada mejora

| Componente | Normalización | Valor de una mejora |
|---|---|---|
| AUC | `(AUC − 0.5)/0.5` | **+0.01 de AUC = +1 punto** |
| RMSE log1p | `1 − RMSE/RMSE_nulo` | `20·ΔRMSE/RMSE_nulo`; con `RMSE_nulo ≈ 1–1.5` (orden de magnitud en simulaciones), bajar 0.05 ≈ +0.7–1 punto |
| Winkler | `1 − W/W_nulo` | Máximo 5 puntos en total |

**Consecuencia:** la mayor parte del esfuerzo va al **orden del riesgo**. La magnitud y los cuantiles se hacen bien,
sin buscar perfección.

### 2.2 `prob_evento` → AUC

- El AUC solo mide el **orden**, así que la calibración no cambia el puntaje y no se invierte tiempo en ella para esta
  columna.
- Para ensamblar modelos se **promedian rangos** (o logits), no probabilidades crudas.
- **No se rebalancean las clases** (ni SMOTE ni submuestreo): no mejoran el AUC y distorsionan las probabilidades que
  la Solución B reutiliza para la magnitud.
- El AUC se calcula agrupando todos los municipio-mes. Importan tanto el orden entre municipios (lo domina la tasa
  histórica) como entre meses (estacionalidad y ENSO).

### 2.3 `personas_desplazadas_estimadas` → RMSE en log1p

- El valor que minimiza el error cuadrático en log1p es `E[log1p(y) | x]`. Por eso se entrega
  **`ŷ = expm1(E[log1p(y)|x])`**, nunca la media en escala lineal `E[y|x]`, que sobreestima y castiga el puntaje.
- Como `log1p(0) = 0`, la descomposición hurdle es exacta:
  `E[log1p y | x] = P(evento | x) · E[log1p y | evento, x]`. El regresor condicional se entrena con **todas** las filas
  con evento, incluidas las que tienen 0 personas, para que el producto sea exacto.
- Esto implica que en la Solución B **la probabilidad sí debe estar calibrada**, porque entra multiplicando
  (🟡 calibración isotónica sobre el corte de validación).
- Resultado esperado: valores pequeños para la mayoría de las filas. No son errores: es el óptimo bajo esta métrica.

### 2.4 `q10` / `q90` → Winkler

- **Identidad clave:** `Winkler = (2/α)·[pinball₀.₁ + pinball₀.₉] = 10 × (pérdida pinball de los cuantiles 0.1 y 0.9)`
  en escala log1p. Está verificada numéricamente en `tests/test_evaluacion.py`. Por eso se usa regresión cuantílica
  sobre `log1p(y)`.
- Los cuantiles no cambian con transformaciones monótonas (`Qτ(log1p y) = log1p(Qτ(y))`), así que se modelan en log1p
  y se devuelven con `expm1`.
- **Regla de los ceros.** Sea `q = P(y > 0 | x)`. Es la probabilidad de que haya personas afectadas, que es menor o igual
  a la de evento (hay eventos solo con viviendas):
  - `q10 = 0` si `q < 0.9`, que es casi siempre.
  - `q90 = 0` si `q < 0.1`: para un municipio de bajo riesgo, el intervalo óptimo es [0, 0].
  - Si `q ≥ 0.1`, `q90` es el cuantil `1 − 0.1/q` de la distribución de las personas cuando hay afectados.
- **La cobertura óptima no es 80 %.** Con tantos ceros, un intervalo [0, 0] en municipios de bajo riesgo acierta casi
  siempre, así que el intervalo óptimo cubre **más** del 80 % de los casos. Forzarlo a 80 % achicaría los intervalos y
  empeoraría el Winkler. El ajuste fino, si se hace, es un factor sobre `q90` elegido para **minimizar el Winkler** en la
  validación (⏳).
- Siempre se garantiza `q10 ≤ q90`: `entrega.armar_entrega` ordena los cuantiles si el modelo los devuelve cruzados ✅.

### 2.5 Cómo se usan las métricas

**Por qué calculamos un puntaje si no somos jurados.** (1) Para comparar modelos con el mismo criterio con que nos
califican: un modelo puede ganar en AUC y perder en RMSE, y solo los pesos oficiales dicen cuál es mejor en total.
(2) Para no gastar entregas (una cada 30 minutos) ni perseguir la tabla en vivo, que mide otro periodo. (3) Para
detectar errores: si el componente de magnitud da 0, no le ganamos a "predecir 0 personas", y casi siempre es un
error. (4) Para priorizar el tiempo. El T&C publica las fórmulas (art. 6.3); lo prohibido es intentar reconstruir el
set oculto (art. 13.1), y solo usamos meses del entrenamiento.

**Métrica oficial vs. lo que optimiza cada modelo:**

| Salida | Métrica oficial | Pérdida al entrenar | Métrica del early stopping |
|---|---|---|---|
| Probabilidad | AUC | Log-loss binaria (el AUC no se optimiza directamente; la log-loss suele dar un buen orden) | AUC |
| Magnitud | RMSE en log1p | Error cuadrático sobre `log1p(y)`: **la misma** métrica | Error cuadrático |
| Cuantiles | Winkler | Pérdida de cuantil 0.1 y 0.9: **equivalente** al Winkler | Pérdida de cuantil |

**Diagnósticos** (`evaluacion.diagnosticos`, se guardan en la bitácora pero **no deciden**) ✅:

| Métrica | Para qué sirve |
|---|---|
| AUC por corte y su dispersión | Estabilidad: ¿el modelo es consistente o tuvo suerte en un periodo? |
| Precisión promedio (PR-AUC) | Con eventos poco frecuentes, qué tan limpia es la parte alta del ranking (alerta temprana) |
| Brier | Calibración de la probabilidad; importa en el hurdle de la Solución B y al leer el geovisor |
| RMSE log1p con y sin afectados | De dónde viene el error de la magnitud |
| Cobertura y ancho medio del intervalo | Entender el Winkler (la cobertura esperada es mayor al 80 %) |

---

## 3. Validación (D3)

### 3.1 Cortes temporales ✅

| Corte | Entrena | Valida | Qué imita |
|---|---|---|---|
| F1 | 2018-01 → 2020-09 | 2020-10 → 2021-09 | Horizonte de 1 a 12 meses |
| F2 | 2018-01 → 2021-09 | 2021-10 → 2022-09 | Horizonte de 1 a 12 meses, el más reciente |
| F3 | 2018-01 → 2020-09 | 2021-10 → 2022-09 | Horizonte de 13 a 24 meses, parecido a `prueba_oculta` |

- Nunca se usa validación cruzada aleatoria: mezclaría futuro y pasado y daría una validación optimista.
- El modelo final se reentrena con todo 2018-01 → 2022-09.
- Los modelos se comparan con el **puntaje de validación promedio de F1–F3** (y su dispersión), no con métricas
  sueltas. Cada corrida queda registrada en `outputs/reportes/experimentos.csv` con las métricas y los diagnósticos.

### 3.2 Controles de fuga de información

| Riesgo | Decisión | Estado |
|---|---|---|
| Usar como variables las columnas de historial del mismo mes (`n_eventos`, viviendas…) | Excluidas: no existen en prueba y son parte del resultado. Solo se usan agregadas desde el pasado | 🟡 |
| Climatología calculada con los meses que se validan | Se recalcula en cada corte usando solo los meses anteriores | 🟡 |
| Climatología de las filas de entrenamiento que contiene su propia etiqueta | **Leave-one-year-out:** para una fila del año Y, la climatología se calcula sin el año Y. Para las filas a predecir se usa toda la historia | 🟡 a validar |
| Parámetros de transformación (límites de recorte, escalado) | `Transformador.fit` solo con la historia del corte | ✅ |
| Early stopping con el mismo periodo que se evalúa (validación optimista) | El early stopping usa los **últimos 12 meses de la historia** de cada corte; luego se reentrena con toda la historia | 🟡 plan |
| Imputar con covariables de los meses de prueba | Permitido: las covariables vienen en los tres archivos y no contienen el objetivo | ✅ |
| Rezagos del ONI que cruzan de entrenamiento a prueba | Permitido: el ONI es una covariable entregada para todos los meses | ✅ diseño |
| Reportes de emergencias de los periodos de prueba | Prohibido por el T&C 13.1; no se usan | ✅ |

### 3.3 Tabla en vivo vs. nota oficial

- La tabla en vivo usa `prueba_equipos` (2022-10 a 2024-04) y la nota oficial `prueba_oculta` (2024-05 a 2025-12).
  **No se ajustan hiperparámetros para escalar en la tabla en vivo.**
- Como cuenta la última entrega válida, la última entrega debe ser el mejor modelo según la validación temporal, no un
  experimento.

### 3.4 Limitación conocida

Todo el entrenamiento cae en El Niño débil, neutral y La Niña. La prueba incluye **un El Niño fuerte (2023-24)** que
la validación no puede imitar: el diagnóstico muestra el ONI de prueba fuera del rango de entrenamiento. Por eso
existen la regla de no recortar el ONI (D4) y la opción de restricciones monótonas sobre el ONI (🟡). La lluvia
observada habría medido el efecto real del clima, pero los organizadores no la permiten.

---

## 4. Calidad y tratamiento de datos (D4) ✅

Implementado en `src/hackaton/calidad.py` (diagnóstico) y `src/hackaton/tratamiento.py` (corrección). Se probó con
datos sintéticos "sucios" con problemas inyectados.

| Tema | Decisión | Por qué |
|---|---|---|
| Código DIVIPOLA | Siempre como texto de 5 dígitos (`zfill`) | Leído como número, `05001` pierde el cero y se rompe la unión de tablas y la entrega |
| Duplicados | Se eliminan las filas idénticas. Si una llave municipio-mes tiene filas distintas, se conserva la de menos nulos | La entrega exige una fila por llave; es la regla más conservadora sin información adicional |
| Objetivo incoherente | Se aplica la definición oficial: si hay personas o viviendas afectadas, `tiene_evento = 1` | Es la definición del diccionario de datos, no una suposición del equipo |
| Valores imposibles (−9999, NBI > 100…) | Pasan a nulo y se imputan | Son errores de captura, no información |
| Nulos | Se imputan según la temporalidad: estáticas con el mismo municipio; anuales con el mismo municipio-año o años vecinos; el ONI con el mismo mes. Luego mediana del departamento y nacional. Cada imputación queda marcada en `<col>_imputado` | Aprovecha la estructura del panel antes de recurrir a promedios; las banderas permiten que el modelo aprenda si la falta de dato es informativa |
| Outliers | **Se reportan pero no se eliminan** (IQR extremo y z robusto en log1p) | Un extremo puede ser real (la densidad de Bogotá, un desastre grande); eliminarlo quitaría justamente los casos de mayor impacto |
| Variables asimétricas | `log1p` de densidad, ingresos tributarios e inversión en gestión del riesgo | Reduce la influencia de las colas y estabiliza modelos lineales |
| Recorte de extremos | Por percentiles 0.5–99.5, **aprendidos solo con la historia del corte**, y solo en covariables municipales | Robustez sin fuga de información |
| ONI | **Nunca se recorta** | El ONI de El Niño 2023 supera el máximo del entrenamiento; recortarlo borraría la señal |
| Escalado | Desactivado por defecto; disponible "estándar" o "robusto" | Los árboles no lo necesitan; los modelos lineales sí |
| `inversion_gestion_riesgo` | Se usará per cápita cuando haya población | Es un total en pesos: sin normalizar, solo mide el tamaño del municipio |

---

## 5. Ingeniería de variables (D5) ⏳

**Climatología municipal con Bayes empírico.** Con 57 meses y eventos poco frecuentes, la tasa cruda de un
municipio es ruidosa: un municipio con 1 evento en 57 meses no es "2 % de riesgo" con certeza. Se usa una tasa
Beta-Binomial encogida hacia la tasa de su departamento:

`tasa_suavizada = (eventos + k · tasa_departamento) / (meses + k)`

Así, los municipios con poca historia se apoyan en su región y los de mucha historia conservan su propia tasa. Esta
es la idea de la Solución C, aplicada de forma barata.

**Variables planeadas:**

| Grupo | Variables |
|---|---|
| Historia del municipio | Tasa suavizada global, por mes del año y por fase ENSO; media y p90 de `log1p(personas)` cuando hay evento; viviendas afectadas; mezcla de amenazas (% inundación, movimiento en masa…); tasa reciente (12 y 24 meses) |
| Historia del departamento | Tasa por mes del año y por fase ENSO (estacionalidad regional: régimen bimodal andino vs. monomodal Caribe y Orinoquía) |
| Covariables oficiales | Las de D4, transformadas |
| Tiempo | Mes en seno y coseno; ONI y sus rezagos de 1 a 3 meses; interacción ONI × tasa de inundación del municipio |
| Solución B | Latitud, longitud, área, región, población (densidad × área) y tasa de eventos de los municipios vecinos (sin fuga) |

---

## 6. Modelos (D6) ⏳

| | Solución A — base robusta | Solución B — máximo desempeño |
|---|---|---|
| `prob_evento` | LightGBM clasificador | Ensamble LightGBM + CatBoost (DIVIPOLA y departamento como categóricas), promedio de rangos |
| Magnitud | LightGBM regresor (MSE) sobre `log1p(y)` | Hurdle: `p · E[log1p y \| evento]` con `p` calibrada |
| Cuantiles | LightGBM cuantílico (0.1 y 0.9) sobre `log1p(y)` + factor sobre `q90` que minimiza el Winkler | Analíticos desde la mezcla (regla de ceros de 2.4) o LightGBM cuantílico, según la validación |
| Datos | Solo los del reto | Reto + bloques externos activables |
| Rol | Primera entrega segura (~10:40) | Mejora principal, si supera a A en la validación temporal |

**Por qué gradient boosting:** datos tabulares con interacciones no lineales (relieve × estación × ENSO), corre en
minutos, tolera valores faltantes y es explicable con importancia de variables y SHAP para el panel.

**Opciones abiertas (🟡):** restricciones monótonas del ONI por región; regresión logística como miembro del
ensamble (extrapola el ONI de forma lineal); pesos del ensamble final según la validación.

**Interfaz común:** cada solución expone `entrenar(X, y_evento, y_personas, semilla)` y `predecir(modelo, X)`. El
orquestador (`scripts/entrenar.py`) no cambia entre soluciones ✅.

**Frente a otras opciones:** la regresión logística es interpretable y extrapola, pero exige construir las interacciones
a mano (queda como posible apoyo del ensamble). Random Forest suele ordenar peor y no trae cuantiles en scikit-learn.
XGBoost es de la misma familia sin ventaja clara; CatBoost entra en B por su manejo de categóricas como DIVIPOLA.
Las redes neuronales necesitan más datos y ajuste. La climatología pura es el piso que hay que superar.

### 6.1 Hiperparámetros 🟡 (plan; valores en `config.LGBM_BASE`, `config.EARLY_STOPPING` y `config.BUSQUEDA`)

**Principio:** el modelo final se entrena **una sola vez** con los hiperparámetros elegidos. Elegirlos sí requiere
varias corridas, siempre sobre los cortes temporales. Se hace una búsqueda **pequeña**: el ajuste fino suele aportar
mucho menos que las buenas variables, una búsqueda grande sobre 3 cortes termina eligiendo ruido, y la nota oficial
(2024-2025) tiene un clima distinto al del entrenamiento, así que se prefieren parámetros robustos.

| Hiperparámetro | Valor base | Qué controla | Por qué este valor |
|---|---|---|---|
| `learning_rate` | 0.05 | Peso de cada árbol nuevo | Pasos cortos, robustos, en un tiempo razonable |
| Número de árboles | Early stopping (paciencia 100, tope 2000) | Cuántos árboles se suman | Lo decide la validación, no se fija a mano |
| `num_leaves` | 31 | Complejidad de cada árbol | Complejidad media; principal control de complejidad en LightGBM |
| `min_child_samples` | 100 | Mínimo de filas por hoja | Eventos poco frecuentes y filas correlacionadas: hojas pequeñas serían ruido |
| `colsample_bytree` | 0.8 | % de variables por árbol | Diversidad; evita depender siempre de la variable más fuerte |
| `subsample` + `subsample_freq` | 0.8 + 1 | % de filas por árbol | Diversidad (sin `subsample_freq` ≥ 1, LightGBM lo ignora) |
| `reg_lambda` | 1 | Penalización L2 de las hojas | Prudencia con reglas apoyadas en pocos casos |
| `random_state` + `deterministic` | 42 + True | Azar del muestreo | Resultados reproducibles (entregable 2) |

**Procedimiento:**
1. **Early stopping sin optimismo:** en cada corte, los últimos 12 meses de la historia sirven para detener el
   entrenamiento; luego se reentrena con toda la historia usando el mejor número de árboles × 1.15. El modelo final
   sigue el mismo procedimiento con todo el entrenamiento.
2. **Búsqueda aleatoria de unas 15 combinaciones, solo para el clasificador** (vale el 50 %): `num_leaves` 15–63,
   `min_child_samples` 50–300, `colsample_bytree` 0.6–0.9, `reg_lambda` 0–10. Limitada a 20–30 minutos, idealmente
   durante el almuerzo. La magnitud y los cuantiles usan la configuración base.
3. **Criterio:** puntaje de validación promedio de F1–F3. Si dos combinaciones difieren menos que la variación normal
   entre cortes, gana la más regularizada y estable.
4. **Opcional:** promediar el modelo final entre 3 semillas (en la probabilidad, promedio de rangos).
5. **La tabla en vivo no elige hiperparámetros:** solo confirma que nada se rompió.
6. En los cuantiles, `alpha=0.1/0.9` es el percentil que se estima; no es el α = 0.2 del Winkler.

---

## 7. Datos externos (D7)

**Criterios para priorizar:** (1) señal esperada sobre inundaciones y movimientos en masa; (2) fuente abierta y de
alcance **global**, porque la tabla de replicabilidad vale 5 % y el PNUD quiere llevar el modelo a otros países;
(3) que se pueda precalcular antes del evento con DIVIPOLA como llave.

| Fuente | Decisión | Estado |
|---|---|---|
| Geografía municipal | Capa generalizada de la UPRA (derivada del MGN del DANE). El geoportal del DANE bloquea las consultas automatizadas y no se intentó saltar ese bloqueo. Validación: las áreas calculadas difieren menos de 3 % de las oficiales | ✅ |
| Población | Se calcula como densidad × área (datos del reto + cartografía); no se usan proyecciones externas | ✅ diseño |
| Lluvia CHIRPS | **No permitida:** los organizadores indicaron el 6 oct que no se pueden usar datos anteriores ni posteriores al periodo del reto | ⛔ |
| Amenaza (JRC GloFAS, NASA LHASA) | No se usa: son capas construidas con datos de otros periodos | ⛔ |
| Emergencias anteriores a 2018 (UNGRD o DesInventar) | **No permitido** (regla de los organizadores, 6 oct) | ⛔ |
| Datos privados o reportes de emergencias de 2022-10 en adelante | Excluidos por el T&C | ✅ |

---

## 8. Reproducibilidad e ingeniería (D8) ✅

| Decisión | Por qué |
|---|---|
| **Generador de datos sintéticos** con el esquema exacto (filas, columnas, códigos DIVIPOLA reales) y una versión "sucia" | Permite construir y probar todo el pipeline antes de recibir los datos, a las 10:00 del día D |
| `config.py` como única fuente de rutas, columnas, cortes y bloques | Evita inconsistencias entre módulos bajo presión de tiempo |
| Orquestador único (`entrenar.py`): tratamiento → validación → modelo final → entrega validada | Un solo comando reproduce la entrega (entregable 2) |
| Validador de formato antes de guardar cada entrega | Una entrega inválida no cuenta y además consume la ventana de 30 minutos |
| Bitácora `experimentos.csv` y entregas con fecha y hora en el nombre | Trazabilidad para elegir la entrega final y para la auditoría |
| Pruebas con pytest (18) sobre las métricas, la entrega, la calidad, el tratamiento y el geovisor | Detectar regresiones al cambiar código el día D |
| Git con commits frecuentes; datos del reto fuera del repositorio | El T&C permite auditar el historial y exige confidencialidad |
| El notebook entregable encuentra el repositorio en Drive, descomprimido en Colab o en local, e incluye la declaración de IA | Lo pueden reproducir los evaluadores, no solo el equipo |
| Declaración de uso de IA en `DECLARACION_IA.md` | Obligatoria (T&C 9.1); las decisiones de este documento son del equipo |

---

## 9. Geovisor (D9) ✅

| Decisión | Por qué |
|---|---|
| HTML único, mapa SVG propio, sin librerías ni mapas base | Abre sin internet y como archivo adjunto; no depende de servicios externos el día D |
| Clases de color por cuantiles calculados sobre **todos** los meses | Los colores son comparables durante la animación; con cuantiles por mes, cada mes parecería igual de riesgoso |
| Escala secuencial de un solo tono (azul) | Representa magnitud sin arcoíris; legible en modo claro y oscuro |
| Probabilidad y personas en gráficos separados, nunca en doble eje | El doble eje sugiere correlaciones que no existen |
| Intervalo 80 % como banda alrededor de la estimación | Comunica la incertidumbre, que es uno de los tres objetivos del reto |
| Ranking y descarga CSV del mes visible | Cada valor del mapa también se puede leer en una tabla (accesibilidad) y se puede exportar |
| Capa de tasa histórica solo con `--historico` | Es información derivada de los datos del reto: uso interno |

---

## 10. Riesgos y mitigación

| Riesgo | Mitigación |
|---|---|
| El ONI de prueba (El Niño 2023) está fuera del rango de entrenamiento | No recortar el ONI; opción de restricciones monótonas o de un componente lineal |
| Cambios en la intensidad del reporte de la UNGRD a lo largo del tiempo | Tasa reciente como variable; cortes de validación recientes (F2) |
| Sobreajuste a la tabla en vivo | Decidir con la validación temporal y el corte F3; no perseguir la tabla |
| Municipios con poca historia | Encogimiento Bayes empírico hacia el departamento (D5) |
| Intervalos mal ajustados | Factor sobre `q90` que minimiza el Winkler en la validación; una cobertura mayor al 80 % es esperable y no es un error |
| El objetivo refleja la capacidad de reporte de cada municipio, no solo el riesgo | Advertirlo al interpretar ingresos e inversión; la tasa reciente captura cambios de reporte |
| Variables en pesos corrientes (inflación) y valores repetidos en prueba | Posición relativa de los ingresos dentro de cada año (🟡); no interpretar tendencias en 2021–2025 |
| Entrega inválida en el último minuto | Validador automático; entrega final planeada para las 13:25 |
| Uso de datos no permitidos | Regla confirmada el 6 oct: solo datos del periodo del reto + cartografía; los scripts de lluvia e índices quedan marcados como no permitidos |

## 11. Decisiones pendientes

- [x] Lluvia observada y emergencias anteriores a 2018: **no permitidas** (organizadores, 6 oct)
- [ ] ¿El intervalo se califica solo con el Winkler? (pregunta 5) ❓
- [ ] Confirmar con datos reales el leave-one-year-out frente a la climatología calculada con todo el pasado 🟡
- [ ] Calibración de `p` para el hurdle de la Solución B 🟡
- [ ] Restricciones monótonas del ONI y pesos del ensamble final 🟡
- [ ] Qué hacer con los outliers del objetivo, una vez vistos los datos reales 🟡
- [ ] Confirmar el plan de hiperparámetros (6.1) 🟡
- [ ] `ingresos_tributarios`: valor nominal o posición relativa dentro de cada año 🟡
- [ ] Formato de publicación del geovisor (archivo local o en línea), según la respuesta de los organizadores ❓

---

## 12. Experimentos del día y decisiones por notebook (6 oct, datos reales)

Todas las cifras son el **puntaje de validación** (fórmula oficial sobre los cortes F1–F3) y están en
`outputs/reportes/registro_experimentos.csv`.

| Notebook | Qué probó | Resultado clave | Decisión |
|---|---|---|---|
| `notebook_AICOLOMBIA_hackathon` (V1) | Plantilla oficial + tratamiento + historial solo pasado, jerárquico y suavizado + LightGBM × 4 con early stopping | 20.06 puntos (F2: 24.07 frente a 18.6 del modelo base); tabla en vivo: AUC 0.759 | Entrega 1 |
| `notebook_experimentos` (ronda 1) | Árboles fijos 200/400/800, magnitud en dos partes, nivel nacional, regularización, combinación | Ninguna supera a la V1; **menos árboles rinden mejor**; dos partes y nivel nacional no aportan; el early stopping es inestable con la tendencia de reporte | Mantener V1; ir hacia modelos más simples |
| `notebook_experimentos2` (ronda 2) | 50/100 árboles, modelo simple, tasa 0.02, pesos por año, **contexto espacial**, ensamble | `E10_espacial_simple` 21.96 (+1.9; F3 +3.1); ensamble 22.04 (empate) | Elegir `E10_espacial_simple` (E10S) |
| `notebook_experimentos3_E10S` (ronda 3) | Alrededor de E10S: ablación, definición de vecinos, más información de vecinos, complejidad, semillas, ensamble fijo | Nada supera a E10S por ≥ 0.5; población y área +0.40, ubicación +0.34, vecinos +0.12; el **ruido** entre corridas es de ±0.3 por corte | **E10S es la configuración final** |

### Configuración final (E10S)

| Componente | Decisión |
|---|---|
| Tratamiento | Duplicados; objetivo según la definición oficial; valores imposibles → nulo; imputación por temporalidad con banderas `_imputado` |
| Historial | Para cada año, solo años anteriores (2018 de calentamiento). Tasas suavizadas: municipio → departamento → nacional (K = 24, 120), mes del año con K = 3, 60 y 1,102; magnitud típica (K = 3); mezcla de amenazas; tasa de los últimos 12 meses |
| Cartografía | Capa pública de la UPRA (MGN del DANE), descargada y validada: centroide (latitud, longitud), área, población = densidad × área, tasa histórica de los 5 municipios más cercanos (con la misma historia de la fila). Permitida por los organizadores si es pública y declarada |
| Clasificador | LightGBM: 100 árboles fijos, tasa 0.05, 15 hojas, mínimo 200 filas por hoja, 80 % de variables y filas por árbol, L2 = 1, semilla 42 |
| Magnitud y cuantiles | LightGBM sobre `log1p(personas)` (error cuadrático; cuantiles 0.1 y 0.9), misma regularización, early stopping con los últimos 12 meses de la historia (paciencia 100, × 1.15, mínimo 50 árboles); factor de q90 que minimiza el Winkler (mediana de los cortes) |
| Validación | F1, F2 y F3; se cambia de configuración solo si mejora ≥ 0.5 puntos sin empeorar F3 |

### Lecciones
- La señal está casi toda en el **historial** y en el **territorio** (población, ubicación). Los modelos complejos aprenden ruido de cada año.
- La **magnitud** apenas le gana a predecir 0 personas (RMSE ≈ 1.26 frente a 1.335): hay poca señal mensual sobre cuántas personas se verán afectadas.
- Con tantos ceros, la cobertura del intervalo queda cerca del 92 %, como se esperaba.
- Cambios mínimos en los datos (último decimal de la cartografía) mueven cada corte ±0.3 puntos: diferencias menores a ~0.3 en el promedio no son concluyentes.

## 13. Cómo mejorarlo (priorizado)

| # | Idea | Por qué | Costo y riesgo |
|---|---|---|---|
| 1 | **Lluvia observada de las mismas fechas del reto** (p. ej. CHIRPS 2018-01 a 2025-12) y sus anomalías respecto a ese mismo periodo | Es lo único que puede ordenar mejor los **meses** (hoy solo hay mes y ONI) | Requiere confirmación explícita de los organizadores (la regla prohíbe datos fuera de las fechas del reto) y extraer lluvia por municipio |
| 2 | **CatBoost** como modelo alternativo y miembro de un ensamble | Otro tipo de boosting (árboles simétricos, manejo de categóricas como DIVIPOLA); aporta diversidad | Bajo; hay que instalarlo en Colab |
| 3 | XGBoost como miembro del ensamble | Misma familia que LightGBM: poca ganancia esperada, algo de diversidad | Bajo |
| 4 | Magnitud: modelo condicionado a que haya evento, con las variables espaciales y la probabilidad calibrada | Es el componente más débil (20 % de la nota) | Medio; la ronda 1 sin espaciales no mejoró |
| 5 | Decisiones más robustas: más cortes temporales y promedio de semillas | Reduce el ruido de ±0.3 que hoy impide distinguir variantes cercanas | Más tiempo de cómputo |
| 6 | En operación: componente autorregresivo mensual (persistencia) | Si el modelo se actualiza cada mes con los eventos reportados, la persistencia sí sirve | No aplica al reto (no hay rezagos en prueba) |

**No recomendados:** redes recurrentes (no hay rezagos en prueba, pocos datos, menos explicables) y regresión
Tweedie para la magnitud (modela la media de personas, mientras que la métrica premia la media del logaritmo; el error
cuadrático sobre `log1p` ya es exactamente la métrica).
