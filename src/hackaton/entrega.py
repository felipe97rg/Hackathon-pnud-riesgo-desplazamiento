"""Arma, valida y guarda el archivo de predicciones en el formato oficial."""
from datetime import datetime

import numpy as np
import pandas as pd

from hackaton import config
from hackaton.evaluacion import validar_entrega


def armar_entrega(filas, predicciones):
    """Combina las llaves DIVIPOLA-fecha con las predicciones y aplica las restricciones del formato.

    filas: DataFrame con DIVIPOLA y fecha (orden de filas_a_predecir).
    predicciones: DataFrame alineado por posición con config.PREDICCIONES.
    """
    pred = predicciones.reset_index(drop=True)
    entrega = pd.DataFrame({
        "DIVIPOLA": filas["DIVIPOLA"].astype(str).str.zfill(5).to_numpy(),
        "fecha": pd.to_datetime(filas["fecha"]).dt.strftime("%Y-%m-01").to_numpy(),
    })
    entrega["prob_evento"] = np.clip(pred["prob_evento"].to_numpy(float), 0, 1)
    entrega["personas_desplazadas_estimadas"] = np.clip(pred["personas_desplazadas_estimadas"].to_numpy(float), 0, None)
    q10 = np.clip(pred["personas_desplazadas_q10"].to_numpy(float), 0, None)
    q90 = np.clip(pred["personas_desplazadas_q90"].to_numpy(float), 0, None)
    # Cuantiles no cruzados: se ordenan si el modelo los produce invertidos.
    entrega["personas_desplazadas_q10"] = np.minimum(q10, q90)
    entrega["personas_desplazadas_q90"] = np.maximum(q10, q90)
    return entrega[config.COLUMNAS_ENTREGA]


def guardar_entrega(entrega, pruebas, etiqueta):
    """Valida y guarda en outputs/entregas/entrega_<fecha-hora>_<etiqueta>.csv. Devuelve la ruta."""
    pruebas = [p.assign(fecha=pd.to_datetime(p["fecha"]).dt.strftime("%Y-%m-01")) for p in pruebas]
    validar_entrega(entrega, pruebas)
    config.DIR_ENTREGAS.mkdir(parents=True, exist_ok=True)
    ruta = config.DIR_ENTREGAS / f"entrega_{datetime.now():%Y%m%d_%H%M}_{etiqueta}.csv"
    entrega.to_csv(ruta, index=False)
    print(f"Guardada: {ruta}")
    return ruta
