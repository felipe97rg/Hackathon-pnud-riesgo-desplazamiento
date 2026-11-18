import numpy as np
import pandas as pd

from hackaton import calidad


def errores(diag):
    r = diag["reglas"]
    return r[(r["tipo"] == "error") & (r["violaciones"] > 0)]


def test_datos_limpios_sin_problemas(datos_juguete):
    diag = calidad.diagnostico_completo(datos_juguete)
    assert diag["nulos"].empty
    assert diag["duplicados"][["filas_identicas", "llaves_repetidas"]].to_numpy().sum() == 0
    assert diag["panel"].drop(columns="archivo").to_numpy().sum() == 0
    assert errores(diag).empty
    assert diag["consistencia"]["grupos_inconsistentes"].sum() == 0


def test_detecta_problemas_inyectados(datos_juguete):
    tr = datos_juguete["entrenamiento"]
    tr.loc[0:4, "nbi_pct"] = np.nan
    tr.loc[10, "elevacion_media"] = -9999
    tr.loc[tr.index[tr["tiene_evento"] == 0][0], "personas_desplazadas"] = 5
    datos_juguete["entrenamiento"] = pd.concat([tr, tr.iloc[[3]]], ignore_index=True)

    diag = calidad.diagnostico_completo(datos_juguete)
    assert diag["nulos"].query("columna == 'nbi_pct'")["nulos"].item() == 6   # 5 + la fila duplicada
    assert diag["duplicados"].query("archivo == 'entrenamiento'")["filas_identicas"].item() == 1
    reglas = set(errores(diag)["regla"])
    assert "elevacion_media en [-50, 6000]" in reglas
    assert "afectación (personas o viviendas) implica tiene_evento=1" in reglas
    assert "elevacion_media" in set(diag["consistencia"].query("grupos_inconsistentes > 0")["columna"])


def test_outliers_y_cambio_distribucion(datos_juguete):
    datos_juguete["prueba_oculta"]["ONI"] = 3.0
    diag = calidad.diagnostico_completo(datos_juguete)
    oni = diag["cambio_distribucion"].query("columna == 'ONI' and comparacion.str.endswith('prueba_oculta')")
    assert oni["pct_fuera_rango"].item() == 100
    assert {"outliers_iqr", "outliers_mad"} <= set(diag["outliers"].columns)
