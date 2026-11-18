# Geovisor (entregable 3)

Un **único archivo HTML autocontenido**: no usa librerías ni mapas base externos, así que abre sin internet
(por ejemplo, adjunto en el formulario de entrega). Puede construirse completamente con IA sin declararlo (T&C 9.3).

## Generarlo

```bash
python scripts/generar_geovisor.py --modelo "Solución A"                 # usa la entrega más reciente
python scripts/generar_geovisor.py --entrega outputs/entregas/entrega_X.csv --modelo "Solución B"
python scripts/generar_geovisor.py --demo --historico sinteticos         # prueba con datos sintéticos
```

Salida: `outputs/geovisor/geovisor.html` (~2 MB, no versionado porque contiene predicciones).
Requiere `data/externos/procesados/municipios.geojson` (lo crea `scripts/externos/01_geografia.py`; ya está en el repo).

| Opción | Efecto |
|---|---|
| `--entrega` | CSV en formato de entrega. Por defecto, el más reciente de `outputs/entregas/` |
| `--modelo` | Texto que aparece en el subtítulo (p. ej. "Solución A") |
| `--historico reto` | Agrega la capa "Tasa histórica de eventos". Es información derivada de los datos del reto: **uso interno, no publicar** |
| `--demo` | Predicciones de demostración (climatología simple sobre sintéticos), con aviso visible. No es el modelo |

## Qué muestra
- **Mapa municipal** coloreado según la variable elegida: probabilidad de evento, personas afectadas estimadas,
  incertidumbre (q90 − q10) y, opcionalmente, la tasa histórica. Las clases salen de cuantiles calculados sobre
  todos los meses, así que los colores son comparables a lo largo de la animación.
- **Selector de mes** con animación (▶), **filtro por departamento** (acerca el mapa y atenúa el resto) y
  **búsqueda de municipio**.
- **Indicadores** del mes y del filtro: probabilidad media, suma de personas estimadas y municipio de mayor riesgo.
- **Ficha municipal**: cifras del mes y series de los 39 meses (probabilidad; personas con la banda del intervalo
  80 %). Al hacer clic en la serie se cambia el mes.
- **Ranking** de los 15 municipios de mayor valor y **descarga en CSV** del mes y filtro visibles.
- Zoom con la rueda del mouse o los botones, y arrastre para desplazarse. San Andrés y Providencia aparecen en un
  recuadro reubicado. Modo claro y oscuro según el sistema.

## Archivos
| Archivo | Rol |
|---|---|
| `geovisor/plantilla.html` | Plantilla (HTML, CSS y JS). Los datos se inyectan en `/*__DATOS__*/` y `/*__GEO__*/` |
| `src/hackaton/geovisor.py` | Arma los datos compactos por municipio y escribe el HTML |
| `scripts/generar_geovisor.py` | Línea de comandos (entrega más reciente, `--demo`, `--historico`) |

## Ver en el navegador
Abrir `outputs/geovisor/geovisor.html` con doble clic, o servir la carpeta `outputs/` con
`python -m http.server 8765` y abrir `http://localhost:8765/geovisor/geovisor.html`.

## Confidencialidad
Sin `--historico`, el geovisor solo contiene predicciones y geometría pública. Con `--historico` incluye datos del
reto, así que esa versión nunca se publica.
