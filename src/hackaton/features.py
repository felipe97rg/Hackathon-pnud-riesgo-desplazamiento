"""Construcción de las matrices de features para entrenar y predecir.

Solución A: climatología municipal + covariables oficiales + tiempo (mes, ONI y rezagos).
Solución B: lo de A + contexto espacial (config.SOLUCIONES["B"]["bloques"]):
- geografia  → lat, lon, área, región; población = densidad × área (y derivados: inversión per cápita)
- vecindad   → tasa de eventos de los municipios vecinos (desde vecinos.csv + historia, sin fuga)
Regla de los organizadores: nada de datos anteriores ni posteriores al periodo del reto (sin lluvia observada,
sin índices externos). Los rezagos del ONI se calculan solo con el ONI de los tres archivos (los primeros meses
de 2018 quedan sin rezago: LightGBM maneja los faltantes).

Los datos llegan ya tratados (tratamiento.preparar_datos: sin nulos ni duplicados, con banderas <col>_imputado).
Transformaciones: tratamiento.Transformador ajustado SOLO con 'historia' (log1p de densidad, ingresos e inversión,
recorte por percentiles; escalado solo si el modelo es lineal); inversión per cápita si hay población; mes en
seno/coseno, ONI rezagado 1–3 meses (vía datos.panel_covariables), interacciones ONI × tasa de inundación.

Cuidados que salen del diccionario de datos:
- Las columnas de historial (n_eventos, viviendas, tipos…) NO se usan del mismo mes: no existen en prueba.
- ingresos_tributarios está en pesos corrientes (sube con la inflación): opción de usar su posición relativa
  dentro de cada año en vez del valor nominal.
- inversion_gestion_riesgo es un total y repite el valor de 2020 en 2021–2025: llevarla a per cápita.
- La mayoría de covariables son fijas por municipio: el orden entre meses solo viene del mes y el ONI.
- Conservar 'fecha' en las matrices (sin usarla como variable): la necesita el early stopping de los modelos.
"""
# Estado: POR IMPLEMENTAR.


def features_tiempo(panel):
    """Mes cíclico y rezagos del ONI calculados sobre el panel continuo (entrenamiento + pruebas)."""
    raise NotImplementedError("Pendiente")


def features_covariables(df):
    """Transformaciones de las covariables oficiales."""
    raise NotImplementedError("Pendiente")


def features_vecindad(historia, filas):
    """Tasa histórica de eventos de los vecinos de cada municipio (usa config.ARCHIVO_VECINOS)."""
    raise NotImplementedError("Pendiente (Solución B)")


def construir_matrices(historia, a_predecir, panel, bloques):
    """Contrato que usa scripts/entrenar.py.

    historia:   filas con objetivo disponibles (entrenamiento del fold o todo el entrenamiento).
    a_predecir: filas a predecir (validación del fold o pruebas).
    panel:      datos.panel_covariables(datos), para rezagos que cruzan entre archivos.
    bloques:    lista de bloques externos a activar.

    Devuelve (X_historia, X_predecir) con las mismas columnas y alineadas por posición con las entradas.
    X_historia usa climatología fuera de bloque; X_predecir usa la climatología de toda la historia.
    """
    raise NotImplementedError("Pendiente: ver docstring del módulo")
