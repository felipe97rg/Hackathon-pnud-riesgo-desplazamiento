"""Modelos. Cada módulo de solución expone la misma interfaz:

    entrenar(X, y_evento, y_personas, semilla) -> modelo
    predecir(modelo, X) -> DataFrame con config.PREDICCIONES:
        prob_evento, personas_desplazadas_estimadas, personas_desplazadas_q10, personas_desplazadas_q90

Recordatorio de qué predecir (ver métricas en la skill del proyecto):
- prob_evento: solo importa el orden (AUC).
- personas_estimadas = expm1(E[log1p(y) | x]), nunca la media de y.
- q10/q90: cuantiles 0.1/0.9 de log1p(y) (pinball = Winkler); con muchos ceros q10 ≈ 0 y q90 = 0 si P(personas > 0) < 0.1
  (P(personas > 0) ≤ P(evento): hay eventos solo con viviendas).
"""
import importlib


def cargar_modelo(nombre):
    """Importa hackaton.modelos.<nombre> (p. ej. 'solucion_a')."""
    return importlib.import_module(f"hackaton.modelos.{nombre}")
