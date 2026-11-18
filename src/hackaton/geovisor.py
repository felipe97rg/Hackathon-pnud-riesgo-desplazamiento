"""Construye el geovisor: un HTML autocontenido (sin dependencias externas) con mapa municipal, controles
de variable/mes/departamento, ficha por municipio con series y ranking.

Entrada: un DataFrame con el formato de entrega + data/externos/procesados/municipios.geojson.
Plantilla: geovisor/plantilla.html (los datos se inyectan en /*__DATOS__*/ y /*__GEO__*/).
"""
import copy
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from hackaton import config

PLANTILLA = config.RAIZ / "geovisor" / "plantilla.html"
DESPLAZAMIENTO_SAN_ANDRES = 3.5   # grados de longitud hacia el este, para un recuadro cercano a la costa
FUENTES = ("Predicciones del modelo del equipo; geometría municipal UPRA (derivada del MGN del DANE); "
           "datos del reto: UNGRD, TerriData (DNP), Copernicus DEM y NOAA CPC")


def _redondear(serie, decimales):
    return [None if pd.isna(v) else round(float(v), decimales) for v in serie]


def _geometria(geojson_ruta):
    geo = json.loads(Path(geojson_ruta).read_text(encoding="utf-8"))
    features = []
    for f in geo["features"]:
        p = f["properties"]
        codigo = str(p["DIVIPOLA"]).zfill(5)
        geometria = f["geometry"]
        props = {"c": codigo, "n": p.get("municipio", codigo), "d": p.get("departamento", "")}
        if codigo.startswith("88"):   # San Andrés y Providencia: se reubica en un recuadro
            props["i"] = 1
            geometria = copy.deepcopy(geometria)
            _desplazar(geometria["coordinates"], DESPLAZAMIENTO_SAN_ANDRES)
        features.append({"type": "Feature", "properties": props, "geometry": geometria})
    return {"type": "FeatureCollection", "features": features}


def _desplazar(coordenadas, dx):
    """Suma dx a cada longitud, en sitio, sea cual sea el anidamiento (Polygon o MultiPolygon)."""
    for i, elemento in enumerate(coordenadas):
        if isinstance(elemento[0], (int, float)):
            coordenadas[i] = [round(elemento[0] + dx, 4), elemento[1]]
        else:
            _desplazar(elemento, dx)


def construir_geovisor(entrega, ruta_salida, geojson_ruta=None, historico=None, modelo="", demo=False):
    """Escribe el geovisor y devuelve (ruta, resumen).

    entrega:   DataFrame con config.COLUMNAS_ENTREGA.
    historico: DataFrame de entrenamiento (opcional) → capa con la tasa histórica de eventos por municipio.
               Solo para uso interno: es información derivada de los datos del reto.
    """
    geojson_ruta = Path(geojson_ruta or config.DIR_EXT_PROCESADOS / config.ARCHIVO_GEOJSON)
    if not geojson_ruta.exists():
        raise FileNotFoundError(f"Falta {geojson_ruta}. Corre: python scripts/externos/01_geografia.py")
    geo = _geometria(geojson_ruta)
    nombres = {f["properties"]["c"]: f["properties"] for f in geo["features"]}

    df = entrega.copy()
    df["DIVIPOLA"] = df["DIVIPOLA"].astype(str).str.zfill(5)
    df["mes"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m")
    fechas = sorted(df["mes"].unique())
    columnas = {"p": ("prob_evento", 4), "e": ("personas_desplazadas_estimadas", 2),
                "l": ("personas_desplazadas_q10", 2), "u": ("personas_desplazadas_q90", 2)}
    tablas = {k: df.pivot_table(index="DIVIPOLA", columns="mes", values=col, aggfunc="first").reindex(columns=fechas)
              for k, (col, _) in columnas.items()}

    tasa = None
    if historico is not None:
        tasa = historico.assign(DIVIPOLA=historico["DIVIPOLA"].astype(str).str.zfill(5)) \
            .groupby("DIVIPOLA")[config.OBJETIVO_EVENTO].mean()

    municipios = {}
    for codigo in tablas["p"].index:
        props = nombres.get(codigo, {})
        registro = {"n": props.get("n", codigo), "d": props.get("d", "")}
        for k, (_, decimales) in columnas.items():
            registro[k] = _redondear(tablas[k].loc[codigo].to_numpy(), decimales)
        if tasa is not None and codigo in tasa.index:
            registro["h"] = round(float(tasa[codigo]), 4)
        municipios[codigo] = registro

    datos = {
        "meta": {"titulo": "Geovisor de riesgo de desplazamiento", "modelo": modelo, "demo": bool(demo),
                 "generado": f"{datetime.now():%Y-%m-%d %H:%M}", "historico": tasa is not None, "fuentes": FUENTES},
        "fechas": fechas,
        "municipios": municipios,
    }

    def a_js(objeto):
        return json.dumps(objeto, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    html = PLANTILLA.read_text(encoding="utf-8")
    html = html.replace("/*__DATOS__*/null", a_js(datos)).replace("/*__GEO__*/null", a_js(geo))
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(html, encoding="utf-8")

    sin_poligono = sorted(set(municipios) - set(nombres))
    resumen = {"municipios": len(municipios), "meses": len(fechas), "sin_poligono": sin_poligono,
               "poligonos_sin_prediccion": len(set(nombres) - set(municipios)),
               "mb": round(ruta_salida.stat().st_size / 1e6, 2)}
    return ruta_salida, resumen
