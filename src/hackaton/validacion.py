"""Cortes temporales de validación (config.FOLDS).

Cada fold entrega la historia disponible (meses ≤ fin_entreno, con objetivo) y el bloque a validar.
Todo lo que se calcule a partir del objetivo (climatología, vecindad) debe usar SOLO la historia del fold.
"""
import pandas as pd

from hackaton import config


def folds_temporales(entrenamiento, folds=config.FOLDS):
    """Itera (nombre, historia, validacion) sobre los cortes definidos."""
    for f in folds:
        historia = entrenamiento[entrenamiento["fecha"] <= pd.Timestamp(f["fin_entreno"])]
        validacion = entrenamiento[entrenamiento["fecha"].between(pd.Timestamp(f["inicio_valid"]),
                                                                  pd.Timestamp(f["fin_valid"]))]
        yield f["nombre"], historia.reset_index(drop=True), validacion.reset_index(drop=True)
