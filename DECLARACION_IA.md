# Declaración de uso de inteligencia artificial generativa

Exigida por los Términos y Condiciones del hackathón (art. 9.1). Omitirla es causal de descalificación (art. 13.1).

| Campo | Detalle |
|---|---|
| Herramienta | Claude Code (Anthropic), aplicación de escritorio |
| Modelo / versión | Claude Opus 5.5 (`claude-opus-5-5`) |
| Periodo de uso | 4 al 6 de octubre de 2026 |
| Equipo | AI COLOMBIA — integrante: Juan Felipe Rodríguez Granados |
| Modelo entregado | `E10S_LL`: 4 LightGBM simples con historial, contexto espacial (cartografía UPRA) y lluvia CHIRPS (`notebooks/notebook_entrega_E10S_LL_AICOLOMBIA.ipynb`) |

## Alcance del uso
- Lectura y resumen de los Términos y Condiciones, el diccionario de datos y el formato de entrega.
- Propuesta de alternativas de modelado, que el equipo debatió y entre las que eligió (Soluciones A y B).
- Escritura y depuración de código: estructura del repositorio, carga de datos, diagnóstico de calidad y tratamiento de datos, exploración y gráficos, réplica de la métrica oficial, validador de formato, generador de datos sintéticos, feature engineering, entrenamiento y validación.
- Identificación de fuentes de datos abiertas y borrador de la tabla de replicabilidad.
- Descarga y validación de las fuentes abiertas declaradas: cartografía pública de la UPRA (MGN del DANE) y lluvia CHIRPS v2.0 solo para los meses del reto (`scripts/externos/03_lluvia_chirps.py`).
- Notebooks de experimentos (rondas 1 a 5), que el integrante ejecutó en Colab y en local, y registro central de resultados (`outputs/reportes/registro_experimentos.csv`).
- Notebooks de entrega basados en la plantilla oficial: `notebook_entrega_E10S_LL_AICOLOMBIA.ipynb` (entregado) y `notebook_entrega_ENS_E10S_E7_CB_LL_AICOLOMBIA.ipynb` (alternativa de ensamble, no entregada).
- Redacción de la documentación (README, decisiones técnicas, tabla de replicabilidad, explicaciones de los notebooks de entrega en `docs/explicacion_notebook_entrega*.md`) y explicaciones de conceptos (métricas, hiperparámetros, validación temporal).
- Decisiones tomadas por el integrante con esa información: qué experimentos correr; agregar la lluvia CHIRPS solo para los meses del reto; validar su aporte con un placebo de control y la regla de mejorar ≥ 0.5 puntos sin empeorar el corte F3; y, entre E10S_LL (25.56 puntos de validación) y el ensamble fijo definido antes de ver resultados (25.81), entregar E10S_LL: la diferencia está dentro del ruido y el modelo simple es más reproducible y explicable.

## Dirección humana y verificación
- La lógica de modelado, la selección de variables y la interpretación de resultados fueron dirigidas y decididas por el equipo.
- El equipo revisó y verificó el código, los resultados y las afirmaciones sugeridas por la herramienta antes de usarlos (art. 9.2).
- El historial de commits de este repositorio documenta el proceso de desarrollo.

## Geovisor / dashboard
- El geovisor (mapa interactivo de `prob_evento`, personas estimadas e intervalo por municipio y mes) se construye con
  la misma herramienta: generador `scripts/generar_geovisor.py`, HTML, CSS y JavaScript propios, a partir del CSV de
  predicciones del modelo entregado y la cartografía pública de la UPRA.
- El art. 9.3 exime al geovisor de esta declaración; se declara de todos modos por transparencia.

<!-- Actualizar esta declaración al cierre si el alcance cambia durante el evento. -->
