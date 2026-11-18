"""Lluvia mensual por municipio desde CHIRPS v2.0, solo para los meses del reto (2018-01 … 2025-12).

Usarla en el modelo solo si los organizadores confirman que se permite la lluvia observada de los mismos meses del
reto (la regla del 6 oct prohíbe datos anteriores o posteriores a esas fechas; no se trae nada fuera de ellas).

Fuente: CHIRPS v2.0 mensual (Climate Hazards Center, UC Santa Barbara; Funk et al., 2015, Scientific Data 2:150066),
archivos COG en https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/cogs/. Se lee solo la ventana de
Colombia con lecturas parciales HTTP (no se descarga el archivo global de ~26 MB). Es cuasi-global (50° S – 50° N),
así que también es su propio proxy de replicabilidad.

Método: promedio de los pixeles (0.05°, ~5.5 km) cuyo centro cae dentro del polígono municipal (cartografía UPRA).
Si ninguno cae dentro (municipios pequeños), el pixel del centroide; si ese pixel no tiene dato (islas, costa), el
pixel válido más cercano al centroide.

Salida: data/externos/procesados/lluvia_chirps_mensual.csv → DIVIPOLA, fecha (AAAA-MM-01), lluvia_mm, pixeles
Uso:    python scripts/externos/03_lluvia_chirps.py [--geojson RUTA] [--hilos 8]
Requiere rasterio y shapely (requirements-externos.txt).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

import rasterio  # noqa: E402
import shapely  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from shapely.geometry import shape  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/cogs/chirps-v2.0.{a}.{m:02d}.cog"
MESES = pd.date_range("2018-01-01", "2025-12-01", freq="MS")   # solo los meses del reto
GEOJSON = [RAIZ / "data/externos/procesados/municipios.geojson", RAIZ / "notebooks/cartografia_municipios.geojson"]
SALIDA = RAIZ / "data/externos/procesados/lluvia_chirps_mensual.csv"
MARGEN = 0.1   # grados alrededor de Colombia


def leer_mes(fecha, limites):
    """Lee la ventana de Colombia de un mes. Devuelve (fecha, matriz en mm, transformación)."""
    oeste, sur, este, norte = limites
    for intento in range(3):
        try:
            with rasterio.open("/vsicurl/" + URL.format(a=fecha.year, m=fecha.month)) as src:
                f0, c0 = src.index(oeste, norte)
                f1, c1 = src.index(este, sur)
                ventana = Window(c0, f0, c1 - c0 + 1, f1 - f0 + 1)
                matriz = src.read(1, window=ventana).astype("float64")
                transformacion = src.window_transform(ventana)
                if src.nodata is not None:
                    matriz[matriz == src.nodata] = np.nan
            matriz[matriz < 0] = np.nan
            return fecha, matriz, transformacion
        except Exception:
            if intento == 2:
                raise
            time.sleep(3)


def indices_por_municipio(geometrias, transformacion, alto, ancho, validos):
    """Índices planos de los pixeles de cada municipio y el método usado."""
    xs = transformacion.c + (np.arange(ancho) + 0.5) * transformacion.a
    ys = transformacion.f + (np.arange(alto) + 0.5) * transformacion.e
    fil_validas, col_validas = np.nonzero(validos)
    indices, metodos = [], []
    for geom in geometrias:
        minx, miny, maxx, maxy = geom.bounds
        cols = np.nonzero((xs >= minx) & (xs <= maxx))[0]
        fils = np.nonzero((ys >= miny) & (ys <= maxy))[0]
        sel = np.array([], dtype=int)
        if len(cols) and len(fils):
            cc, ff = np.meshgrid(cols, fils)
            dentro = shapely.contains_xy(geom, xs[cc], ys[ff])
            sel = (ff[dentro] * ancho + cc[dentro]).astype(int)
            sel = sel[validos.ravel()[sel]]
        metodo = "poligono"
        if len(sel) == 0:
            p = geom.representative_point()
            c = int((p.x - transformacion.c) / transformacion.a)
            f = int((p.y - transformacion.f) / transformacion.e)
            if 0 <= f < alto and 0 <= c < ancho and validos[f, c]:
                sel, metodo = np.array([f * ancho + c]), "centroide"
            else:
                d = (xs[col_validas] - p.x) ** 2 + (ys[fil_validas] - p.y) ** 2
                k = int(np.argmin(d))
                sel, metodo = np.array([fil_validas[k] * ancho + col_validas[k]]), "pixel_valido_mas_cercano"
        indices.append(sel)
        metodos.append(metodo)
    return indices, metodos


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--geojson", type=Path, default=None)
    ap.add_argument("--hilos", type=int, default=8)
    args = ap.parse_args()

    ruta = args.geojson or next((p for p in GEOJSON if p.exists()), None)
    if ruta is None:
        sys.exit("No hay cartografía: correr antes scripts/externos/01_geografia.py")
    feats = json.loads(ruta.read_text(encoding="utf-8"))["features"]
    clave = next(k for k in ("DIVIPOLA", "cod_dane_mpio") if k in feats[0]["properties"])
    divipola = [int(f["properties"][clave]) for f in feats]
    geometrias = [shape(f["geometry"]) for f in feats]
    b = np.array([g.bounds for g in geometrias])
    limites = (b[:, 0].min() - MARGEN, b[:, 1].min() - MARGEN, b[:, 2].max() + MARGEN, b[:, 3].max() + MARGEN)
    print(f"Cartografía: {ruta.name}, {len(feats)} municipios | ventana {np.round(limites, 2).tolist()}")

    t0 = time.time()
    with ThreadPoolExecutor(args.hilos) as ex:
        resultados = sorted(ex.map(lambda m: leer_mes(m, limites), MESES), key=lambda r: r[0])
    formas = {r[1].shape for r in resultados}
    assert len(formas) == 1, f"Las ventanas no tienen la misma forma: {formas}"
    pila = np.stack([r[1] for r in resultados])           # (meses, alto, ancho)
    _, alto, ancho = pila.shape
    print(f"Leídos {len(resultados)} meses en {time.time() - t0:.0f} s | ventana de {alto} × {ancho} pixeles")

    validos = ~np.isnan(pila).any(axis=0)
    indices, metodos = indices_por_municipio(geometrias, resultados[0][2], alto, ancho, validos)
    plana = pila.reshape(len(MESES), -1)
    valores = np.column_stack([plana[:, idx].mean(axis=1) for idx in indices])   # (meses, municipios)

    tabla = pd.DataFrame({
        "DIVIPOLA": np.tile(divipola, len(MESES)),
        "fecha": np.repeat(MESES.strftime("%Y-%m-%d"), len(divipola)),
        "lluvia_mm": np.round(valores.ravel(), 2),
        "pixeles": np.tile([len(i) for i in indices], len(MESES)),
    })

    # Validaciones
    assert tabla["lluvia_mm"].notna().all(), "Quedaron municipios-mes sin lluvia"
    assert not tabla.duplicated(["DIVIPOLA", "fecha"]).any(), "Llaves duplicadas"
    assert tabla["fecha"].nunique() == len(MESES) and tabla["DIVIPOLA"].nunique() == len(set(divipola))
    assert tabla["lluvia_mm"].between(0, 3000).all(), "Lluvia mensual fuera de rango (0–3000 mm)"

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA, index=False)
    print(f"Guardado {SALIDA.relative_to(RAIZ)}: {len(tabla):,} filas")
    print("Método por municipio:", pd.Series(metodos).value_counts().to_dict())
    print("Pixeles por municipio: mediana", int(np.median([len(i) for i in indices])),
          "| con 1 pixel:", sum(len(i) == 1 for i in indices))
    print("lluvia_mm: mín", tabla.lluvia_mm.min(), "| mediana", tabla.lluvia_mm.median(), "| máx", tabla.lluvia_mm.max())
    t = tabla.assign(anio=tabla.fecha.str[:4], mes=tabla.fecha.str[5:7].astype(int))
    print("Promedio nacional por mes del año (mm):", t.groupby("mes").lluvia_mm.mean().round(0).astype(int).to_dict())
    print("Promedio nacional por año (mm/mes):", t.groupby("anio").lluvia_mm.mean().round(0).astype(int).to_dict())


if __name__ == "__main__":
    main()
