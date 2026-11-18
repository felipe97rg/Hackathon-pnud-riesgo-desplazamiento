"""Lectura de los CSV del reto y unión con las tablas externas precomputadas."""
from pathlib import Path
import warnings

import pandas as pd

from hackaton import config


def leer_csv(ruta):
    """Lee un CSV del reto con DIVIPOLA como texto de 5 caracteres y fecha como datetime.

    Fechas ilegibles quedan como NaT (las reporta calidad.py y las repara tratamiento.py).
    """
    df = pd.read_csv(ruta, dtype={"DIVIPOLA": str})
    df["DIVIPOLA"] = df["DIVIPOLA"].str.strip().str.zfill(5)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


def cargar_reto(directorio=config.DIR_RETO, estricto=True):
    """Devuelve {'entrenamiento', 'prueba_equipos', 'prueba_oculta'} → DataFrame, con chequeos básicos.

    estricto=True lanza error si faltan columnas; estricto=False solo avisa (útil para explorar).
    Los duplicados y conteos de filas inesperados solo generan avisos: se tratan en tratamiento.py.
    """
    directorio = Path(directorio)

    def problema(mensaje):
        if estricto:
            raise ValueError(mensaje)
        warnings.warn(mensaje)

    datos = {}
    for nombre, archivo in config.ARCHIVOS_RETO.items():
        ruta = directorio / archivo
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {ruta}. ¿Copiaste los CSV del reto (o generaste los sintéticos)?")
        df = leer_csv(ruta)
        faltan = [c for c in config.ID + config.COVARIABLES if c not in df.columns]
        if faltan:
            problema(f"{archivo}: faltan columnas {faltan}")
        if len(df) != config.FILAS_ESPERADAS[nombre]:
            warnings.warn(f"{archivo}: {len(df):,} filas (se esperaban {config.FILAS_ESPERADAS[nombre]:,})")
        repetidas = df.duplicated(["DIVIPOLA", "fecha"]).sum()
        if repetidas:
            warnings.warn(f"{archivo}: {repetidas} llaves DIVIPOLA-fecha repetidas (usar tratamiento.preparar_datos)")
        datos[nombre] = df
    faltan = [c for c in (config.OBJETIVO_EVENTO, config.OBJETIVO_PERSONAS) if c not in datos["entrenamiento"]]
    if faltan:
        problema(f"entrenamiento.csv: faltan columnas objetivo {faltan}")
    return datos


def panel_covariables(datos):
    """Concatena los tres archivos (solo IDs + covariables) con una columna 'conjunto'.

    Sirve para variables que dependen del tiempo pero no del objetivo (p. ej. rezagos del ONI),
    que necesitan continuidad entre entrenamiento y prueba.
    """
    columnas = config.ID + config.COVARIABLES
    partes = [df[columnas].assign(conjunto=nombre) for nombre, df in datos.items()]
    return pd.concat(partes, ignore_index=True).sort_values(["DIVIPOLA", "fecha"], ignore_index=True)


def filas_a_predecir(datos):
    """Filas DIVIPOLA-fecha de ambos archivos de prueba, en el orden en que se entregarán."""
    return pd.concat([datos["prueba_equipos"], datos["prueba_oculta"]], ignore_index=True)


def unir_externos(df, bloques, directorio=config.DIR_EXT_PROCESADOS):
    """Une las tablas externas de los bloques pedidos. La llave se deduce de las columnas de cada tabla."""
    directorio = Path(directorio)
    for bloque in bloques:
        for archivo in config.BLOQUES_EXTERNOS.get(bloque, []):
            ruta = directorio / archivo
            if not ruta.exists():
                raise FileNotFoundError(f"Bloque '{bloque}': falta {ruta}. Corre scripts/externos/ primero.")
            tabla = pd.read_csv(ruta, dtype={"DIVIPOLA": str})
            if "fecha" in tabla:
                tabla["fecha"] = pd.to_datetime(tabla["fecha"])
            llave = [c for c in ("DIVIPOLA", "anio", "fecha") if c in tabla.columns]
            antes = len(df)
            df = df.merge(tabla, on=llave, how="left", validate="many_to_one")
            assert len(df) == antes, f"La unión con {archivo} cambió el número de filas"
    return df
