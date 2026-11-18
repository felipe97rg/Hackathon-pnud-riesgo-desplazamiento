"""Geografía municipal: polígonos simplificados, centroides, área y región (y vecinos si hay shapely).

Fuente: UPRA (Unidad de Planificación Rural Agropecuaria) — capa "municipios_generalizada", derivada del
Marco Geoestadístico Nacional del DANE, servicio ArcGIS público. La simplificación la hace el servidor
(maxAllowableOffset). El geoportal del DANE bloquea las consultas automatizadas, por eso se usa esta capa.
Proxy global para replicabilidad: GADM nivel 2 / geoBoundaries ADM2.

Salidas (data/externos/procesados/):
    municipios.geojson → polígonos con DIVIPOLA, municipio, departamento (geovisor y vecinos)
    geografia.csv      → DIVIPOLA, lat, lon, area_km2, region
    vecinos.csv        → DIVIPOLA, DIVIPOLA_vecino (solo si shapely está instalado: requirements-externos.txt)

Uso: python scripts/externos/01_geografia.py [--tolerancia 0.002]
"""
import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hackaton import config  # noqa: E402

SERVICIO = ("https://geoservicios.upra.gov.co/arcgis/rest/services/referencia_geografica/"
            "municipios_generalizada/MapServer/0/query")
MINUSCULAS = {"de", "del", "la", "las", "los", "el", "y", "e", "en"}
KM_POR_GRADO_LAT = 110.574
KM_POR_GRADO_LON_ECUADOR = 111.320


def nombre_propio(texto):
    """'SAN JOSÉ DE LA MONTAÑA' → 'San José de la Montaña'."""
    palabras = str(texto).strip().lower().split()
    return " ".join(p if (i and p in MINUSCULAS) else p.capitalize() for i, p in enumerate(palabras))


def descargar(tolerancia):
    parametros = {
        "where": "1=1", "outFields": "cod_dane_mpio,municipio,departamen",
        "outSR": 4326, "maxAllowableOffset": tolerancia, "geometryPrecision": 4,
        "resultRecordCount": 2000, "f": "geojson",
    }
    with urllib.request.urlopen(f"{SERVICIO}?{urllib.parse.urlencode(parametros)}", timeout=180) as respuesta:
        return json.loads(respuesta.read().decode("utf-8"))


def poligonos(geometria):
    """Lista de polígonos (cada uno: [anillo exterior, huecos…]) de un Polygon o MultiPolygon."""
    return [geometria["coordinates"]] if geometria["type"] == "Polygon" else geometria["coordinates"]


def _shoelace(anillo):
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(anillo, anillo[1:] + anillo[:1]):
        cruz = x0 * y1 - x1 * y0
        a += cruz
        cx += (x0 + x1) * cruz
        cy += (y0 + y1) * cruz
    return a / 2, cx, cy


def centroide_y_area(geometria):
    """Centroide (del polígono exterior más grande) y área en km² (proyección local equirectangular)."""
    area_km2, mejor, mejor_area = 0.0, None, -1.0
    for poligono in poligonos(geometria):
        lat_media = sum(y for _, y in poligono[0]) / len(poligono[0])
        escala = KM_POR_GRADO_LAT * KM_POR_GRADO_LON_ECUADOR * math.cos(math.radians(lat_media))
        for k, anillo in enumerate(poligono):
            a, cx, cy = _shoelace(anillo)
            area_km2 += (abs(a) if k == 0 else -abs(a)) * escala
            if k == 0 and abs(a) > mejor_area and a != 0:
                mejor_area, mejor = abs(a), (cx / (6 * a), cy / (6 * a))
    return mejor, area_km2


def vecinos(geojson):
    try:
        from shapely.geometry import shape
        from shapely.strtree import STRtree
    except ImportError:
        print("  (vecinos.csv omitido: instala requirements-externos.txt para calcularlo)")
        return None
    codigos = [f["properties"]["DIVIPOLA"] for f in geojson["features"]]
    formas = [shape(f["geometry"]).buffer(0.001) for f in geojson["features"]]  # tolera la simplificación
    arbol = STRtree(formas)
    pares = [(codigos[i], codigos[j]) for i, forma in enumerate(formas)
             for j in arbol.query(forma, predicate="intersects") if i != j]
    return pd.DataFrame(pares, columns=["DIVIPOLA", "DIVIPOLA_vecino"]).drop_duplicates()


def main():
    parser = argparse.ArgumentParser(description="Descarga y procesa la geografía municipal.")
    parser.add_argument("--tolerancia", type=float, default=0.002, help="grados (0.002 ≈ 220 m)")
    args = parser.parse_args()

    crudo = descargar(args.tolerancia)
    features, filas = [], []
    for f in crudo["features"]:
        if not f.get("geometry"):
            continue
        p = f["properties"]
        codigo = str(p["cod_dane_mpio"]).zfill(5)
        features.append({"type": "Feature", "geometry": f["geometry"], "properties": {
            "DIVIPOLA": codigo, "municipio": nombre_propio(p["municipio"]),
            "departamento": nombre_propio(p["departamen"])}})
        (lon, lat), area = centroide_y_area(f["geometry"])
        filas.append({"DIVIPOLA": codigo, "lat": round(lat, 5), "lon": round(lon, 5), "area_km2": round(area, 2),
                      "region": config.REGION_POR_DEPARTAMENTO.get(codigo[:2], "Otra")})

    geojson = {"type": "FeatureCollection", "features": features}
    config.DIR_EXT_PROCESADOS.mkdir(parents=True, exist_ok=True)
    ruta_geo = config.DIR_EXT_PROCESADOS / config.ARCHIVO_GEOJSON
    ruta_geo.write_text(json.dumps(geojson, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    geografia = pd.DataFrame(filas).drop_duplicates("DIVIPOLA").sort_values("DIVIPOLA")
    geografia.to_csv(config.DIR_EXT_PROCESADOS / "geografia.csv", index=False)
    print(f"  {len(features):,} polígonos -> {ruta_geo.name} ({ruta_geo.stat().st_size / 1e6:.1f} MB)")
    print("  regiones:", geografia["region"].value_counts().to_dict())

    pares = vecinos(geojson)
    if pares is not None:
        pares.to_csv(config.DIR_EXT_PROCESADOS / config.ARCHIVO_VECINOS, index=False)
        print(f"  vecinos.csv: {len(pares):,} pares")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
