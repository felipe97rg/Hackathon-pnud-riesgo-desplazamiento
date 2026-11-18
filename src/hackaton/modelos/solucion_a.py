"""Solución A — Climatología municipal + Gradient Boosting (base robusta).

Modelos (LightGBM, hiperparámetros base en config.LGBM_BASE):
- prob_evento: clasificador binario (pérdida log-loss; métrica de early stopping: AUC).
- personas_estimadas: regresor con error cuadrático sobre log1p(y) → expm1. Es exactamente la métrica oficial.
- q10 / q90: regresión cuantílica (objective="quantile", alpha=0.1 y 0.9) sobre log1p(y) → expm1.
  La pérdida de cuantil equivale al Winkler. Ajuste opcional: un factor sobre q90 elegido para MINIMIZAR el
  Winkler en la validación (no para forzar 80 % de cobertura: con muchos ceros, la cobertura óptima es mayor).

Procedimiento de entrenamiento (config.EARLY_STOPPING):
1. Separar los últimos 12 meses de la historia recibida como conjunto de early stopping (X debe traer 'fecha',
   que no se usa como variable).
2. Entrenar con el resto hasta que la métrica deje de mejorar durante 100 árboles → best_iteration.
3. Reentrenar con toda la historia usando best_iteration × 1.15.
El mismo procedimiento sirve para cada corte de validación y para el modelo final.

Hiperparámetros: configuración base fija; búsqueda aleatoria corta (config.BUSQUEDA) solo para el clasificador,
elegida por el puntaje de validación promedio de F1–F3. Nunca se eligen con la tabla en vivo.
Opcional: promediar el modelo final entre config.SEMILLAS_FINAL (en la probabilidad, promedio de rangos).
"""
# Estado: POR IMPLEMENTAR.


def entrenar(X, y_evento, y_personas, semilla):
    raise NotImplementedError("Pendiente: Solución A")


def predecir(modelo, X):
    raise NotImplementedError("Pendiente: Solución A")
