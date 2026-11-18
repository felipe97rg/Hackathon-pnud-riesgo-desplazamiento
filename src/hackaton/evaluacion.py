"""Métricas oficiales (T&C 6.3) aplicadas a NUESTRA validación, diagnósticos y validador de la entrega.

No calculamos la nota oficial (se calcula con prueba_oculta, que no vemos). Usamos la misma fórmula sobre los
cortes temporales del entrenamiento como brújula para elegir modelos: el "puntaje de validación" (máx. 75).
Sirve para comparar nuestros modelos entre sí, no para predecir la posición en la tabla.

Uso típico:
    from hackaton.evaluacion import puntaje, diagnosticos, validar_entrega
    res = puntaje(valid["tiene_evento"], valid["personas_desplazadas"], prob, est, q10, q90)
    diag = diagnosticos(valid["tiene_evento"], valid["personas_desplazadas"], prob, est, q10, q90)
    validar_entrega(entrega, [prueba_equipos, prueba_oculta])
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from hackaton.config import COLUMNAS_ENTREGA as COLUMNAS

ALPHA = 0.2


def rmse_log(y, y_hat):
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(np.sqrt(np.mean((np.log1p(y_hat) - np.log1p(y)) ** 2)))


def winkler_log(y, q10, q90, alpha=ALPHA):
    z = np.log1p(np.asarray(y, float))
    lo = np.log1p(np.broadcast_to(np.asarray(q10, float), z.shape))
    hi = np.log1p(np.broadcast_to(np.asarray(q90, float), z.shape))
    w = (hi - lo) + (2 / alpha) * np.clip(lo - z, 0, None) + (2 / alpha) * np.clip(z - hi, 0, None)
    return float(np.mean(w))


def puntaje(y_evento, y_personas, prob, est, q10, q90):
    """Métricas oficiales normalizadas y puntaje de validación (máx. 75). Es el criterio para elegir modelos."""
    y_personas = np.asarray(y_personas, float)
    auc = roc_auc_score(y_evento, prob)
    rmse = rmse_log(y_personas, est)
    rmse_nulo = rmse_log(y_personas, np.zeros_like(y_personas))
    wk = winkler_log(y_personas, q10, q90)
    wk_nulo = winkler_log(y_personas, 0.0, 0.0)

    auc_n = max(0.0, (auc - 0.5) / 0.5)
    rmse_n = max(0.0, 1 - rmse / rmse_nulo)
    wk_n = max(0.0, 1 - wk / wk_nulo)
    return {
        "auc": auc,
        "rmse_log": rmse,
        "rmse_log_nulo": rmse_nulo,
        "winkler": wk,
        "winkler_nulo": wk_nulo,
        "auc_n": auc_n,
        "rmse_n": rmse_n,
        "winkler_n": wk_n,
        "puntaje_validacion": 50 * auc_n + 20 * rmse_n + 5 * wk_n,
    }


def diagnosticos(y_evento, y_personas, prob, est, q10, q90):
    """Métricas para ENTENDER el modelo (no para elegirlo).

    - pr_auc: precisión promedio; con eventos poco frecuentes, qué tan limpia es la parte alta del ranking.
    - brier: error cuadrático de la probabilidad; mide calibración (importa en el hurdle de la Solución B).
    - rmse_log_con / sin_afectados: de dónde viene el error de la magnitud.
    - cobertura: % de valores reales dentro de [q10, q90]. Con muchos ceros, la cobertura ÓPTIMA es mayor
      que 80 % ([0, 0] acierta casi siempre en municipios de bajo riesgo): no se fuerza a 80 %.
    - ancho_log: ancho medio del intervalo en log1p.
    """
    y_evento = np.asarray(y_evento)
    y = np.asarray(y_personas, float)
    prob, est = np.asarray(prob, float), np.asarray(est, float)
    lo, hi = np.asarray(q10, float), np.asarray(q90, float)
    hay = y > 0
    return {
        "pr_auc": average_precision_score(y_evento, prob),
        "brier": brier_score_loss(y_evento, np.clip(prob, 0, 1)),
        "rmse_log_con_afectados": rmse_log(y[hay], est[hay]) if hay.any() else np.nan,
        "rmse_log_sin_afectados": rmse_log(y[~hay], est[~hay]) if (~hay).any() else np.nan,
        "cobertura": float(np.mean((y >= lo) & (y <= hi))),
        "ancho_log": float(np.mean(np.log1p(hi) - np.log1p(lo))),
    }


def validar_entrega(entrega, pruebas):
    """Lanza AssertionError con un mensaje claro si la entrega no cumple el formato oficial.

    entrega: DataFrame a subir. pruebas: lista con los DataFrames de prueba_equipos y prueba_oculta
    (leídos con dtype={'DIVIPOLA': str}).
    """
    assert list(entrega.columns) == COLUMNAS, f"Columnas: {list(entrega.columns)} != {COLUMNAS}"
    assert entrega.notna().all().all(), "Hay valores faltantes"

    div = entrega["DIVIPOLA"]
    assert div.map(lambda v: isinstance(v, str) and len(v) == 5 and v.isdigit()).all(), \
        "DIVIPOLA debe ser texto de 5 dígitos (¿se leyó como número?)"
    assert entrega["fecha"].astype(str).str.fullmatch(r"\d{4}-\d{2}-01").all(), "fecha debe ser AAAA-MM-01"

    llaves = entrega[["DIVIPOLA", "fecha"]].astype(str)
    assert not llaves.duplicated().any(), "Hay filas DIVIPOLA-fecha duplicadas"
    esperadas = pd.concat([p[["DIVIPOLA", "fecha"]].astype(str) for p in pruebas], ignore_index=True)
    assert len(entrega) == len(esperadas), f"Filas: {len(entrega)} != {len(esperadas)}"
    faltan = esperadas.merge(llaves, how="left", indicator=True)["_merge"].eq("left_only").sum()
    assert faltan == 0, f"Faltan {faltan} combinaciones DIVIPOLA-fecha de los archivos de prueba"

    p = entrega["prob_evento"]
    assert p.between(0, 1).all(), "prob_evento fuera de [0, 1]"
    num = entrega[COLUMNAS[3:]]
    assert (num >= 0).all().all(), "Hay valores negativos en personas/cuantiles"
    assert (entrega["personas_desplazadas_q10"] <= entrega["personas_desplazadas_q90"]).all(), "q10 > q90"
    print(f"Entrega válida: {len(entrega):,} filas.")
    return True
