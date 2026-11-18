"""⛔ NO SE USA: son capas construidas con datos de otros periodos; la regla de los organizadores solo admite datos del periodo del reto.

Capas estáticas de amenaza y exposición por municipio.

Fuentes (todas globales):
    - JRC / GloFAS — mapas de amenaza de inundación fluvial (periodo de retorno 100 años).
    - NASA LHASA — susceptibilidad global a deslizamientos.
    - HydroSHEDS / HydroRIVERS — red hídrica.
    - ESA WorldCover — cobertura del suelo.
    - Hansen Global Forest Change — pérdida de bosque 2001–2017.

Salida (data/externos/procesados/):
    amenazas.csv → DIVIPOLA, pct_area_inundable, susceptibilidad_deslizamiento, densidad_rios_km_km2,
                   pct_urbano, pct_bosque, pct_perdida_bosque

Requiere requirements-externos.txt (o Earth Engine). Estado: POR IMPLEMENTAR.
"""

if __name__ == "__main__":
    raise NotImplementedError("Pendiente: ver docstring")
