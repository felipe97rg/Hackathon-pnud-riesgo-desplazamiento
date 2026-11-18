import json

import pandas as pd

from hackaton.geovisor import construir_geovisor


def cuadrado(lon, lat, lado=0.1):
    return {"type": "Polygon", "coordinates": [[[lon, lat], [lon + lado, lat], [lon + lado, lat + lado],
                                                [lon, lat + lado], [lon, lat]]]}


def test_construye_html_con_datos_y_recuadro(tmp_path):
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": cuadrado(-75.6, 6.2),
         "properties": {"DIVIPOLA": "05001", "municipio": "Medellín", "departamento": "Antioquia"}},
        {"type": "Feature", "geometry": cuadrado(-81.7, 12.5),
         "properties": {"DIVIPOLA": "88001", "municipio": "San Andrés", "departamento": "San Andrés"}},
    ]}
    ruta_geo = tmp_path / "municipios.geojson"
    ruta_geo.write_text(json.dumps(geo), encoding="utf-8")
    entrega = pd.DataFrame({
        "DIVIPOLA": ["05001", "05001", "99999", "99999"], "fecha": ["2022-10-01", "2022-11-01"] * 2,
        "prob_evento": [0.2, 0.3, 0.1, 0.1], "personas_desplazadas_estimadas": [1.5, 2.0, 0.1, 0.1],
        "personas_desplazadas_q10": [0.0] * 4, "personas_desplazadas_q90": [10.0, 12.0, 0.0, 0.0]})

    ruta, resumen = construir_geovisor(entrega, tmp_path / "geovisor.html", geojson_ruta=ruta_geo, modelo="Prueba")

    html = ruta.read_text(encoding="utf-8")
    assert "/*__DATOS__*/" not in html and "/*__GEO__*/" not in html
    assert '"Medellín"' in html and '"Prueba"' in html
    assert resumen["meses"] == 2 and resumen["sin_poligono"] == ["99999"] and resumen["poligonos_sin_prediccion"] == 1
    # San Andrés se desplaza al recuadro; la geometría original no se modifica.
    assert "-78.2" in html and json.loads(ruta_geo.read_text(encoding="utf-8"))["features"][1]["geometry"]["coordinates"][0][0][0] == -81.7
