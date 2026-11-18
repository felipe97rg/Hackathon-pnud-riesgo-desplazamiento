# Datos

| Carpeta | Contenido | ¿Se versiona? |
|---|---|---|
| `reto/` | Los 3 CSV oficiales (`entrenamiento.csv`, `prueba_equipos.csv`, `prueba_oculta.csv`), entregados el día D a las 10:00 | **No.** Confidenciales (T&C 9 y 13.1) |
| `sinteticos/` | Datos falsos con el mismo esquema, generados con `scripts/generar_datos_sinteticos.py`, más `verdad_pruebas.csv` para simular el puntaje | No (se regeneran) |
| `sinteticos_sucios/` | Los mismos sintéticos con problemas inyectados (nulos, duplicados, DIVIPOLA sin cero, −9999, objetivo incoherente) para practicar el diagnóstico y el tratamiento | No (se regeneran) |
| `externos/crudos/` | Descargas originales de fuentes abiertas (shapefiles, rasters, Excel) | No (pesadas) |
| `externos/procesados/` | Tablas listas para unir por `DIVIPOLA` (y `anio` o `fecha`), producidas por `scripts/externos/` | **Sí** (son datos abiertos y livianos) |

## Tablas externas esperadas (contrato con `src/hackaton/datos.py::unir_externos`)

| Archivo | Llave | Columnas | Bloque |
|---|---|---|---|
| `municipios.geojson` | DIVIPOLA (en properties) | municipio, departamento + polígono simplificado | geovisor, vecindad |
| `geografia.csv` | DIVIPOLA | lat, lon, area_km2, region | geografia |
| `vecinos.csv` | (pares) | DIVIPOLA, DIVIPOLA_vecino | vecindad |

⛔ Regla de los organizadores (6 oct 2026): no se usan datos anteriores ni posteriores a las fechas del reto
(sin lluvia, índices climáticos externos ni emergencias previas a 2018). La población se calcula como
`densidad_poblacional × area_km2`.

`DIVIPOLA` siempre como texto de 5 caracteres (ej. `05001`); `fecha` como `AAAA-MM-01`.
