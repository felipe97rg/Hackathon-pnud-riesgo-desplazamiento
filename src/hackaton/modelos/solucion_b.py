"""Solución B — Hurdle + clima observado + contexto espacial (máximo desempeño).

Plan:
- P(evento): ensamble LightGBM + CatBoost (DIVIPOLA y departamento como categóricas), promedio de rangos.
- m = E[log1p(y) | evento]: regresor entrenado con todas las filas con evento, incluidas las de 0 personas
  (así p · m = E[log1p y] es exacto).
- personas_estimadas = expm1(p · m).
- Cuantiles: (a) analíticos desde la mezcla — con q = P(personas > 0): q90 = cuantil (1 − 0.1/q) de la
  condicional positiva, 0 si q < 0.1; o (b) LightGBM cuantílico. Elegir el mejor en validación.
- p se calibra (isotónica sobre validación) porque entra multiplicando en la magnitud.
- Hiperparámetros y early stopping: mismo procedimiento que la Solución A (config.LGBM_BASE, config.EARLY_STOPPING).
"""
# Estado: POR IMPLEMENTAR.


def entrenar(X, y_evento, y_personas, semilla):
    raise NotImplementedError("Pendiente: Solución B")


def predecir(modelo, X):
    raise NotImplementedError("Pendiente: Solución B")
