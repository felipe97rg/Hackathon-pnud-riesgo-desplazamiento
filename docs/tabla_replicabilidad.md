# Tabla de replicabilidad (entregable 4)

Para cada variable usada por el modelo: fuente en Colombia y proxy equivalente en fuentes abiertas de
alcance global, para llevar el modelo a otros países (T&C 5.1). El modelo entregado (**E10S_LL**) es la
Solución B sin hurdle + lluvia: usa todas las filas marcadas con B y la fila de lluvia.

| Variable del modelo | Descripción | Fuente en Colombia | Proxy abierto global | Solución |
|---|---|---|---|---|
| Climatología municipal (tasa de eventos, perfil mensual, severidad, mezcla de amenazas) | Propensión histórica a eventos con afectación | UNGRD (dataset del reto) | DesInventar Sendai (UNDRR), EM-DAT, GDACS, IDMC (desplazamiento por desastres) | A, B |
| `densidad_poblacional` / población | Exposición | TerriData (DNP), DANE | WorldPop, GHS-POP (GHSL) | A, B |
| `ingresos_tributarios` | Capacidad financiera municipal | TerriData (DNP) | Luces nocturnas VIIRS per cápita; PIB subnacional grillado (Kummu et al.) | A, B |
| `inversion_gestion_riesgo` | Capacidad institucional en gestión del riesgo | TerriData (DNP) | INFORM Subnational (capacidad de afrontamiento); Sendai Framework Monitor | A, B |
| `nbi_pct` | Vulnerabilidad social | Censo 2018 / TerriData | Relative Wealth Index (Meta, HDX), DHS, MPI subnacional (OPHI), SHDI (Global Data Lab) | A, B |
| `elevacion_media`, `pendiente_media`, `pct_pendiente_alta` | Relieve y susceptibilidad a movimientos en masa | Copernicus DEM GLO-90 | Copernicus DEM GLO-90 / SRTM (ya global) | A, B |
| `ONI`, `fase_enso` y rezagos | Variabilidad climática ENSO | NOAA CPC | NOAA CPC (global); otros índices regionales según el país | A, B |
| Mes del año | Estacionalidad | — | Universal | A, B |
| Geometría: lat/lon, área, región, vecinos | Contexto espacial | MGN (DANE), vía capa generalizada de la UPRA | GADM / geoBoundaries | B (y geovisor) |
| Población (derivada) | Exposición absoluta | densidad × área (reto + MGN) | WorldPop, GHS-POP | B |
| Tasa de eventos de los municipios vecinos | Contagio espacial del riesgo | Historial del reto + vecindad del MGN | Historial nacional (DesInventar, EM-DAT) + geoBoundaries | B |
| Lluvia mensual: nivel (log), acumulado de 3 meses, anomalías frente a lo normal del municipio en ese mes del año, anomalía del mes anterior e interacción con el historial | Detonante de inundaciones, crecientes y movimientos en masa | CHIRPS v2.0 mensual (Climate Hazards Center, UCSB; Funk et al., 2015), solo 2018-01 → 2025-12; promedio de los pixeles de ~5 km dentro de cada municipio | CHIRPS (ya es global, 50° S – 50° N); ERA5-Land; GPM IMERG | B (E10S_LL) |

La lluvia se usa **solo para los meses del reto** (2018-01 → 2025-12); la climatología para las anomalías se calcula
con 2018-01 → 2022-09. No se usan índices climáticos externos ni emergencias anteriores a 2018 (regla de los organizadores
del 6 oct: nada anterior ni posterior a las fechas del reto). Script: `scripts/externos/03_lluvia_chirps.py`.
