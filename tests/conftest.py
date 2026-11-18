"""Datos de juguete con el esquema del reto (6 municipios), para probar calidad y tratamiento sin archivos."""
import numpy as np
import pandas as pd
import pytest

from hackaton import config
from hackaton.calidad import fase_esperada

MUNICIPIOS = ["05001", "05002", "08001", "08002", "11001", "27001"]
PERIODOS = {"entrenamiento": ("2018-01-01", "2022-09-01"),
            "prueba_equipos": ("2022-10-01", "2024-04-01"),
            "prueba_oculta": ("2024-05-01", "2025-12-01")}


def construir_datos_juguete(semilla=0):
    rng = np.random.default_rng(semilla)
    estaticas = pd.DataFrame({"DIVIPOLA": MUNICIPIOS, "nbi_pct": rng.uniform(10, 60, 6),
                              "elevacion_media": rng.uniform(0, 2500, 6), "pendiente_media": rng.uniform(1, 25, 6)})
    estaticas["pct_pendiente_alta"] = (estaticas["pendiente_media"] * 2).clip(0, 100)
    datos = {}
    for nombre, (ini, fin) in PERIODOS.items():
        fechas = pd.date_range(ini, fin, freq="MS")
        df = estaticas.merge(pd.DataFrame({"fecha": fechas}), how="cross")
        df["anio"], df["mes"] = df["fecha"].dt.year, df["fecha"].dt.month
        df["densidad_poblacional"] = 50 * (1 + df["DIVIPOLA"].str[-1].astype(int)) * 1.01 ** (df["anio"] - 2018)
        df["ingresos_tributarios"] = 1e5 + 1e4 * (df["anio"] - 2018)
        df["inversion_gestion_riesgo"] = 2e8 + 1e7 * df["DIVIPOLA"].str[-1].astype(int)
        df["ONI"] = df["fecha"].map(dict(zip(fechas, np.round(np.sin(np.arange(len(fechas)) / 6), 2))))
        df["fase_enso"] = fase_esperada(df["ONI"])
        if nombre == "entrenamiento":
            evento = (rng.random(len(df)) < 0.2).astype(int)
            df["tiene_evento"] = evento
            df["personas_desplazadas"] = np.where(evento, rng.integers(1, 200, len(df)), 0)
            for c in config.HISTORIAL:
                df[c] = 0
            df["n_eventos"] = evento
            df["n_inundacion"] = evento
            df["viv_averiadas"] = np.where(evento, 2, 0)
        columnas = config.ID + ([config.OBJETIVO_EVENTO, config.OBJETIVO_PERSONAS] + config.HISTORIAL
                                if nombre == "entrenamiento" else []) + config.COVARIABLES
        datos[nombre] = df[columnas]
    return datos


@pytest.fixture
def datos_juguete():
    return construir_datos_juguete()
