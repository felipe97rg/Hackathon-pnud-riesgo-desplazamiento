"""Climatología municipal: resume el historial (con objetivo) en variables por municipio.

Los archivos de prueba no traen objetivo ni rezagos, así que el pasado de cada municipio se convierte
en encodings constantes que se aplican a todos los meses a predecir.

Regla anti-fuga (crítica):
- Filas a PREDECIR (validación o prueba): encodings calculados con toda la historia disponible.
- Filas de ENTRENAMIENTO del modelo: encodings "fuera de bloque" (leave-one-year-out): para una fila del
  año Y se usa la historia sin el año Y. Si no, el modelo aprende de encodings que contienen su propia
  etiqueta y sobreestima su peso.

Encodings planeados (Bayes empírico Beta-Binomial, encogidos hacia el departamento):
- tasa_evento, tasa_evento_mes (perfil mes-del-año), tasa_evento_fase (La Niña / Neutral / El Niño)
- media y p90 de log1p(personas | evento), viviendas destruidas/averiadas promedio
- mezcla de amenazas (% inundación, % movimiento en masa, …)
- tasa reciente (últimos 12 y 24 meses)
- equivalentes departamentales (tasa × mes, tasa × fase)
"""
# Estado: POR IMPLEMENTAR (Solución A).


def tasa_suavizada(eventos, expuestos, eventos_prior, expuestos_prior, fuerza):
    """Tasa Beta-Binomial: (eventos + fuerza·p_prior) / (expuestos + fuerza)."""
    raise NotImplementedError("Pendiente: Bayes empírico para tasas por municipio")


def calcular_climatologia(historia):
    """Encodings por municipio (y por municipio × mes / × fase) a partir de 'historia'.

    Devuelve un dict de DataFrames con llaves DIVIPOLA, [DIVIPOLA, mes] o [DIVIPOLA, fase_enso].
    """
    raise NotImplementedError("Pendiente: encodings municipales y departamentales")


def aplicar_climatologia(filas, climatologia):
    """Une los encodings a 'filas' (por DIVIPOLA, mes y fase_enso)."""
    raise NotImplementedError("Pendiente")


def climatologia_fuera_de_bloque(historia):
    """Encodings para las filas de entrenamiento con leave-one-year-out (ver regla anti-fuga)."""
    raise NotImplementedError("Pendiente")
