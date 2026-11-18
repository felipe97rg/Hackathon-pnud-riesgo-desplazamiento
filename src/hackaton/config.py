"""Rutas, nombres de columnas, cortes de validación y configuración de las soluciones.

Todo valor que se repita en más de un módulo vive aquí.
"""
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parents[2]
DIR_DATOS = RAIZ / "data"
DIR_RETO = DIR_DATOS / "reto"                      # CSV oficiales (no versionados)
DIR_SINTETICOS = DIR_DATOS / "sinteticos"          # datos falsos con el mismo esquema
DIR_SINTETICOS_SUCIOS = DIR_DATOS / "sinteticos_sucios"  # los mismos, con problemas inyectados (nulos, duplicados…)
DIR_EXT_CRUDOS = DIR_DATOS / "externos" / "crudos"
DIR_EXT_PROCESADOS = DIR_DATOS / "externos" / "procesados"
DIR_SALIDAS = RAIZ / "outputs"
DIR_ENTREGAS = DIR_SALIDAS / "entregas"
DIR_MODELOS = DIR_SALIDAS / "modelos"
DIR_REPORTES = DIR_SALIDAS / "reportes"

ARCHIVOS_RETO = {
    "entrenamiento": "entrenamiento.csv",
    "prueba_equipos": "prueba_equipos.csv",
    "prueba_oculta": "prueba_oculta.csv",
}
FILAS_ESPERADAS = {"entrenamiento": 62_814, "prueba_equipos": 20_938, "prueba_oculta": 22_040}
FUENTES_DATOS = {"reto": DIR_RETO, "sinteticos": DIR_SINTETICOS, "sinteticos_sucios": DIR_SINTETICOS_SUCIOS}
N_MUNICIPIOS = 1_102

# ── Columnas (diccionario_datos.csv) ─────────────────────────────────────────
ID = ["DIVIPOLA", "fecha", "anio", "mes"]
OBJETIVO_EVENTO = "tiene_evento"
OBJETIVO_PERSONAS = "personas_desplazadas"
HISTORIAL = [
    "n_eventos", "familias_desplazadas", "viv_destruidas", "viv_averiadas",
    "n_movimiento_masa", "n_inundacion", "n_vendaval", "n_creciente_subita",
    "n_avenida_torrencial", "n_otros",
]
COVARIABLES = [
    "densidad_poblacional", "ingresos_tributarios", "inversion_gestion_riesgo", "nbi_pct",
    "elevacion_media", "pendiente_media", "pct_pendiente_alta", "ONI", "fase_enso",
]
# Covariables por temporalidad (define cómo se revisan y cómo se imputan)
COV_ESTATICAS = ["nbi_pct", "elevacion_media", "pendiente_media", "pct_pendiente_alta"]
COV_ANUALES = ["densidad_poblacional", "ingresos_tributarios", "inversion_gestion_riesgo"]
COV_NACIONALES = ["ONI"]                       # igual para todos los municipios en un mes
COV_NUMERICAS = COV_ESTATICAS + COV_ANUALES + COV_NACIONALES
COV_LOG = ["densidad_poblacional", "ingresos_tributarios", "inversion_gestion_riesgo"]  # muy asimétricas
UMBRAL_ENSO = 0.5                              # El Nino si ONI ≥ 0.5, La Nina si ONI ≤ −0.5
FASES_ENSO = ["El Nino", "La Nina", "Neutral"]

# Rangos físicamente válidos (mín, máx); None = sin límite. Fuera de rango = error de dato, no outlier.
RANGOS_VALIDOS = {
    "densidad_poblacional": (0, None),
    "ingresos_tributarios": (0, None),
    "inversion_gestion_riesgo": (0, None),
    "nbi_pct": (0, 100),
    "elevacion_media": (-50, 6000),
    "pendiente_media": (0, 90),
    "pct_pendiente_alta": (0, 100),
    "ONI": (-3.5, 3.5),
}

COLUMNAS_ENTREGA = [
    "DIVIPOLA", "fecha", "prob_evento", "personas_desplazadas_estimadas",
    "personas_desplazadas_q10", "personas_desplazadas_q90",
]
PREDICCIONES = COLUMNAS_ENTREGA[2:]

# ── Validación temporal (ver docs del enfoque en el README) ──────────────────
FIN_ENTRENAMIENTO = "2022-09-01"
FOLDS = [
    {"nombre": "F1", "fin_entreno": "2020-09-01", "inicio_valid": "2020-10-01", "fin_valid": "2021-09-01"},
    {"nombre": "F2", "fin_entreno": "2021-09-01", "inicio_valid": "2021-10-01", "fin_valid": "2022-09-01"},
    {"nombre": "F3", "fin_entreno": "2020-09-01", "inicio_valid": "2021-10-01", "fin_valid": "2022-09-01"},
]

SEMILLA = 42

# ── Hiperparámetros de LightGBM (plan y justificación: docs/decisiones_tecnicas.md, sección 6.1) ──
LGBM_BASE = {
    "learning_rate": 0.05,          # peso de cada árbol: pasos cortos, más robustos
    "n_estimators": 2000,           # tope; el número real lo decide el early stopping
    "num_leaves": 31,               # complejidad de cada árbol
    "min_child_samples": 100,       # mínimo de filas por hoja (eventos poco frecuentes)
    "colsample_bytree": 0.8,        # cada árbol ve el 80 % de las variables
    "subsample": 0.8,               # … y el 80 % de las filas
    "subsample_freq": 1,            # sin esto, LightGBM ignora 'subsample'
    "reg_lambda": 1.0,              # regularización L2 de los valores de las hojas
    "random_state": SEMILLA,
    "deterministic": True, "force_row_wise": True,   # mismo resultado en cada corrida
    "verbose": -1,
}
# Early stopping con los últimos meses de la HISTORIA de cada corte (no con el corte que se evalúa, para que
# la validación no sea optimista). Luego se reentrena con toda la historia usando best_iteration × factor.
EARLY_STOPPING = {"meses": 12, "paciencia": 100, "factor_arboles": 1.15}
# Búsqueda aleatoria corta, solo para el clasificador (50 % de la nota). Criterio: puntaje de validación promedio.
BUSQUEDA = {
    "combinaciones": 15,
    "rangos": {"num_leaves": (15, 63), "min_child_samples": (50, 300),
               "colsample_bytree": (0.6, 0.9), "reg_lambda": (0.0, 10.0)},
}
SEMILLAS_FINAL = [42, 7, 2026]      # opcional: promediar el modelo final entre semillas

# ── Tratamiento de covariables (tratamiento.Transformador) ───────────────────
# Los árboles (LightGBM, CatBoost) no necesitan escalado; sí los modelos lineales.
TRATAMIENTO = {
    "winsorizar": (0.005, 0.995),   # percentiles de recorte aprendidos en entrenamiento; None = no recortar
    "escalar": None,                # None | "estandar" | "robusto"
}
DIR_EDA = DIR_SALIDAS / "eda"       # reportes de exploración (NO versionados: resumen datos del reto)

# ── Datos externos precomputados (data/externos/procesados/) ─────────────────
# La llave de cada tabla se detecta por sus columnas: DIVIPOLA, DIVIPOLA+anio o DIVIPOLA+fecha / solo fecha.
# Regla de los organizadores (6 oct 2026): no se pueden usar datos anteriores ni posteriores a las fechas del reto.
# Quedan fuera la lluvia observada, los índices climáticos externos y las emergencias previas a 2018.
# Solo se usa cartografía estática (límites municipales → lat, lon, área, región); población = densidad × área.
BLOQUES_EXTERNOS = {
    "geografia": ["geografia.csv"],                          # lat, lon, area_km2, region
}
ARCHIVO_VECINOS = "vecinos.csv"                              # pares DIVIPOLA, DIVIPOLA_vecino
ARCHIVO_GEOJSON = "municipios.geojson"                       # polígonos MGN simplificados (geovisor)

# Región natural por código de departamento (aproximación estándar por departamento)
REGION_POR_DEPARTAMENTO = {
    **dict.fromkeys(["05", "11", "15", "17", "25", "41", "54", "63", "66", "68", "73"], "Andina"),
    **dict.fromkeys(["08", "13", "20", "23", "44", "47", "70", "88"], "Caribe"),
    **dict.fromkeys(["19", "27", "52", "76"], "Pacifica"),
    **dict.fromkeys(["50", "81", "85", "99"], "Orinoquia"),
    **dict.fromkeys(["18", "86", "91", "94", "95", "97"], "Amazonia"),
}

# ── Soluciones ───────────────────────────────────────────────────────────────
# "vecindad" no es una tabla: se calcula en features.py a partir de vecinos.csv y del historial.
SOLUCIONES = {
    "A": {"modelo": "solucion_a", "bloques": []},
    "B": {"modelo": "solucion_b", "bloques": ["geografia", "vecindad"]},
}
