import numpy as np
import pandas as pd

from hackaton import calidad, config
from hackaton.tratamiento import Transformador, preparar_datos


def ensuciar(datos):
    tr = datos["entrenamiento"]
    tr.loc[tr["DIVIPOLA"] == "08001", "nbi_pct"] = np.nan                    # municipio sin dato
    tr.loc[5, "elevacion_media"] = -9999                                      # centinela
    tr.loc[tr["anio"] == 2019, "ingresos_tributarios"] = np.nan              # un año completo
    tr.loc[7, "ONI"] = np.nan
    sin_evento = tr.index[tr["tiene_evento"] == 0][0]
    tr.loc[sin_evento, "personas_desplazadas"] = 9                            # afectación sin evento
    datos["entrenamiento"] = pd.concat([tr, tr.iloc[[1, 2]]], ignore_index=True)
    pe = datos["prueba_equipos"]
    pe["DIVIPOLA"] = pe["DIVIPOLA"].str.lstrip("0")                           # perdió el cero
    datos["prueba_oculta"] = pd.concat([datos["prueba_oculta"], datos["prueba_oculta"].iloc[[0]]], ignore_index=True)
    return datos, sin_evento


def test_preparar_datos_deja_todo_limpio(datos_juguete):
    filas = {k: len(v) for k, v in datos_juguete.items()}
    sucios, _ = ensuciar(datos_juguete)
    limpios, registro = preparar_datos(sucios, verbose=False)

    assert {k: len(v) for k, v in limpios.items()} == filas
    diag = calidad.diagnostico_completo(limpios)
    assert diag["nulos"].empty
    assert diag["duplicados"][["filas_identicas", "llaves_repetidas"]].to_numpy().sum() == 0
    assert diag["panel"]["divipola_mal_formado"].sum() == 0
    r = diag["reglas"]
    assert r[(r["tipo"] == "error") & (r["violaciones"] > 0)].empty
    assert registro["objetivo"]["afectacion_sin_evento_corregida"] == 1
    assert limpios["entrenamiento"]["tiene_evento"].dtype.kind == "i"


def test_imputacion_respeta_temporalidad(datos_juguete):
    original = datos_juguete["entrenamiento"].copy()
    sucios, _ = ensuciar(datos_juguete)
    limpios, _ = preparar_datos(sucios, verbose=False)
    tr = limpios["entrenamiento"]
    # El ONI faltante se recupera del mismo mes (dato nacional) → valor exacto.
    assert tr.loc[7, "ONI"] == original.loc[7, "ONI"]
    # La elevación centinela se reemplaza con el valor real del mismo municipio (estática).
    assert tr.loc[5, "elevacion_media"] == original.loc[5, "elevacion_media"]
    assert tr.loc[5, "elevacion_media_imputado"] == 1
    # Mismas banderas en los tres archivos, para que las matrices tengan las mismas columnas.
    banderas = {c for c in tr.columns if c.endswith("_imputado")}
    assert banderas and all(banderas <= set(df.columns) for df in limpios.values())


def test_transformador_aprende_solo_de_la_historia(datos_juguete):
    historia, prueba = datos_juguete["entrenamiento"], datos_juguete["prueba_oculta"].copy()
    prueba["ONI"] = 2.5                                           # El Niño fuera del rango de entrenamiento
    prueba["nbi_pct"] = 99.0
    t = Transformador(winsorizar=(0.05, 0.95), escalar=None).fit(historia)
    salida = t.transform(prueba)
    assert "log_densidad_poblacional" in salida and "densidad_poblacional" not in salida
    assert salida["nbi_pct"].max() <= historia["nbi_pct"].quantile(0.95) + 1e-9   # recortado con límites de la historia
    assert (salida["ONI"] == 2.5).all()                                          # el ONI no se recorta


def test_transformador_escala_robusta(datos_juguete):
    historia = datos_juguete["entrenamiento"]
    salida = Transformador(winsorizar=None, escalar="robusto").fit_transform(historia)
    assert abs(salida["nbi_pct"].median()) < 1e-9
    assert set(config.COV_LOG) & set(salida.columns) == set()
