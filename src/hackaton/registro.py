"""Registro central de experimentos y entregas (trazabilidad).

outputs/reportes/registro_experimentos.csv  → una fila por variante y corrida: métricas por corte, configuración,
                                              fecha, entorno y huella de los datos (solo agregados, nunca filas).
outputs/reportes/registro_entregas.csv      → cada entrega subida al formulario y el resultado de la tabla en vivo.

Los notebooks de experimentos escriben directamente en el registro si encuentran la carpeta del proyecto (Drive
montado en Colab o ejecución local); si no, dejan `registro_experimentos_<id>.csv` dentro de su zip y se integra con
`python scripts/consolidar_registro.py <archivos>`.
"""
from pathlib import Path

import pandas as pd

from hackaton import config

REGISTRO_EXPERIMENTOS = config.DIR_REPORTES / "registro_experimentos.csv"
REGISTRO_ENTREGAS = config.DIR_REPORTES / "registro_entregas.csv"

CORTES = ["F1", "F2", "F3"]
METRICAS_CORTE = ["puntos", "auc", "rmse_log", "winkler", "cobertura"]
COLUMNAS_EXPERIMENTOS = (
    ["id_corrida", "fecha_hora", "ronda", "notebook", "entorno", "lightgbm", "huella_datos", "eventos_entrenamiento",
     "fuente", "variante", "descripcion", "config", "puntos_prom", "puntos_desv", "auc_prom", "rmse_log_prom",
     "winkler_prom", "cobertura_prom", "factor_q90", "archivo_entrega"]
    + [f"{m}_{c}" for m in METRICAS_CORTE for c in CORTES]
    + ["notas"]
)
COLUMNAS_ENTREGAS = ["fecha_hora", "entrega", "variante", "archivo", "puntaje_validacion", "auc_tabla_vivo",
                     "rmse_log_tabla_vivo", "winkler_tabla_vivo", "notas"]


def _leer(ruta, columnas):
    ruta = Path(ruta)
    if ruta.exists():
        return pd.read_csv(ruta, dtype={"id_corrida": str})
    return pd.DataFrame(columns=columnas)


def agregar_experimentos(filas, ruta=REGISTRO_EXPERIMENTOS):
    """Agrega filas (DataFrame o lista de dicts) sin duplicar el par (id_corrida, variante)."""
    nuevo = pd.DataFrame(filas).reindex(columns=COLUMNAS_EXPERIMENTOS)
    existente = _leer(ruta, COLUMNAS_EXPERIMENTOS)
    if len(existente):
        claves = set(zip(existente["id_corrida"].astype(str), existente["variante"]))
        nuevo = nuevo[[(str(i), v) not in claves for i, v in zip(nuevo["id_corrida"], nuevo["variante"])]]
    total = (pd.concat([existente, nuevo], ignore_index=True) if len(existente) else nuevo).reindex(columns=COLUMNAS_EXPERIMENTOS)
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    total.to_csv(ruta, index=False)
    return len(nuevo), len(total)


def agregar_entrega(ruta=REGISTRO_ENTREGAS, **campos):
    existente = _leer(ruta, COLUMNAS_ENTREGAS)
    nueva = pd.DataFrame([campos])
    total = (pd.concat([existente, nueva], ignore_index=True) if len(existente) else nueva).reindex(columns=COLUMNAS_ENTREGAS)
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    total.to_csv(ruta, index=False)
    return total


def ranking(ruta=REGISTRO_EXPERIMENTOS, eventos=None):
    """Mejor resultado de cada variante (la corrida más reciente si se repitió), ordenado por puntaje.

    eventos: filtra por número de eventos del entrenamiento (con los datos reales del reto son 4977), para no mezclar
    corridas con datos sintéticos.
    """
    r = _leer(ruta, COLUMNAS_EXPERIMENTOS)
    if eventos is not None:
        r = r[pd.to_numeric(r["eventos_entrenamiento"], errors="coerce") == eventos]
    r = r.sort_values(["fecha_hora", "id_corrida"]).drop_duplicates("variante", keep="last")
    return r.sort_values("puntos_prom", ascending=False).reset_index(drop=True)
