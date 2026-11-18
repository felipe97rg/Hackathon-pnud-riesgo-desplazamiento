"""Tratamiento de datos: tipos, duplicados, objetivo incoherente, imputación y transformaciones.

Dos niveles, por la regla anti-fuga:
1. preparar_datos(datos): limpieza que usa solo covariables (vienen en los 3 archivos) o reglas fijas.
   Se aplica una vez a todo. Imputar un valor estático con el de otro mes del mismo municipio no filtra
   información del objetivo.
2. Transformador: aprende parámetros de distribución (límites de recorte, centro y escala). Se ajusta
   SOLO con la historia de cada fold: Transformador().fit(historia).transform(a_predecir).

Cada paso devuelve un registro de lo que cambió para poder reportarlo (y defenderlo ante el panel).
"""
import numpy as np
import pandas as pd

from hackaton import config
from hackaton.calidad import LLAVE, fase_esperada


def normalizar_tipos(df):
    """DIVIPOLA a texto de 5 dígitos, fecha al día 1 del mes (o reconstruida desde anio/mes) y anio/mes desde fecha."""
    df = df.copy()
    df["DIVIPOLA"] = (df["DIVIPOLA"].astype(str).str.strip()
                      .str.replace(r"\.0$", "", regex=True).str.zfill(5))
    fecha = pd.to_datetime(df["fecha"], errors="coerce")
    if {"anio", "mes"} <= set(df.columns):
        desde_partes = pd.to_datetime(dict(year=df["anio"], month=df["mes"], day=1), errors="coerce")
        fecha = fecha.fillna(desde_partes)
    df["fecha"] = fecha.dt.to_period("M").dt.to_timestamp()
    df["anio"], df["mes"] = df["fecha"].dt.year, df["fecha"].dt.month
    return df


def deduplicar(df):
    """Quita filas idénticas; si una llave DIVIPOLA-fecha sigue repetida, conserva la fila con menos nulos."""
    n0 = len(df)
    df = df.drop_duplicates()
    identicas = n0 - len(df)
    conflictos = int(df.duplicated(LLAVE).sum())
    if conflictos:
        df = (df.assign(_nulos=df.isna().sum(axis=1)).sort_values(LLAVE + ["_nulos"], kind="stable")
              .drop_duplicates(LLAVE).drop(columns="_nulos").sort_index())
    return df.reset_index(drop=True), {"filas_identicas": identicas, "llaves_en_conflicto": conflictos}


def limpiar_objetivo(tr):
    """Corrige el objetivo según la definición oficial de tiene_evento (personas > 0 o viviendas > 0)."""
    tr = tr.copy()
    ev, per = config.OBJETIVO_EVENTO, config.OBJETIVO_PERSONAS
    conteos = [per] + [c for c in config.HISTORIAL if c in tr]
    registro = {}

    negativos = tr[conteos] < 0
    registro["conteos_negativos_a_nulo"] = int(negativos.to_numpy().sum())
    tr[conteos] = tr[conteos].mask(negativos)

    viv = tr[[c for c in ("viv_destruidas", "viv_averiadas") if c in tr]].fillna(0).sum(axis=1)
    afectacion = ((tr[per].fillna(0) > 0) | (viv > 0)).astype(int)
    invalido = ~tr[ev].isin([0, 1])
    registro["tiene_evento_derivado_de_la_definicion"] = int(invalido.sum())
    tr.loc[invalido, ev] = afectacion[invalido]
    contradiccion = (tr[ev] == 0) & (afectacion == 1)
    registro["afectacion_sin_evento_corregida"] = int(contradiccion.sum())
    tr.loc[contradiccion, ev] = 1
    tr[ev] = tr[ev].astype(int)

    sin_evento = tr[per].isna() & (tr[ev] == 0)
    tr.loc[sin_evento, per] = 0
    con_evento = tr[per].isna() & (tr[ev] == 1)
    if con_evento.any():
        positivos = tr.loc[tr[per] > 0]
        mediana_mun = positivos.groupby("DIVIPOLA")[per].median()
        relleno = tr.loc[con_evento, "DIVIPOLA"].map(mediana_mun).fillna(positivos[per].median())
        tr.loc[con_evento, per] = relleno.round()
    registro["personas_nulo_a_cero_sin_evento"] = int(sin_evento.sum())
    registro["personas_nulo_imputado_con_evento"] = int(con_evento.sum())

    historial = [c for c in config.HISTORIAL if c in tr]
    registro["historial_nulo_a_cero"] = int(tr[historial].isna().to_numpy().sum())
    tr[historial] = tr[historial].fillna(0)
    return tr, registro


def imputar_covariables(datos):
    """Imputa covariables sobre el panel completo (entrenamiento + pruebas) según su temporalidad.

    Orden: (0) valores imposibles → nulo; (1) ONI por mes y fase_enso desde el ONI; (2) estáticas con el
    valor del mismo municipio; (3) anuales con el mismo municipio-año y luego años vecinos del municipio;
    (4) mediana del departamento; (5) mediana nacional. Agrega <col>_imputado (0/1) donde hubo faltantes.
    """
    partes = [df.assign(_conjunto=nombre, _orden=np.arange(len(df))) for nombre, df in datos.items()]
    panel = pd.concat(partes, ignore_index=True)
    registro = []

    def presentes(columnas):
        return [c for c in columnas if c in panel]

    def rellenar(col, regla, valores):
        antes = panel[col].isna()
        panel[col] = panel[col].fillna(valores)
        n = int((antes & panel[col].notna()).sum())
        if n:
            registro.append({"columna": col, "regla": regla, "imputados": n})

    for col in presentes(config.RANGOS_VALIDOS):
        lo, hi = config.RANGOS_VALIDOS[col]
        fuera = pd.Series(False, index=panel.index)
        if lo is not None:
            fuera |= panel[col] < lo
        if hi is not None:
            fuera |= panel[col] > hi
        if fuera.any():
            panel.loc[fuera, col] = np.nan
            registro.append({"columna": col, "regla": "valor imposible -> nulo", "imputados": int(fuera.sum())})
    faltaba = {c: panel[c].isna() for c in presentes(config.COV_NUMERICAS) if panel[c].isna().any()}

    for col in presentes(config.COV_NACIONALES):
        rellenar(col, "mismo mes (dato nacional)", panel.groupby("fecha")[col].transform("median"))
    if {"ONI", "fase_enso"} <= set(panel.columns):
        esperada = fase_esperada(panel["ONI"])
        corregir = esperada.notna() & (panel["fase_enso"] != esperada)
        if corregir.any():
            panel.loc[corregir, "fase_enso"] = esperada[corregir]
            registro.append({"columna": "fase_enso", "regla": "derivada del ONI", "imputados": int(corregir.sum())})

    for col in presentes(config.COV_ESTATICAS):
        rellenar(col, "mismo municipio", panel.groupby("DIVIPOLA")[col].transform("median"))
    claves = pd.MultiIndex.from_frame(panel[["DIVIPOLA", "anio"]])
    for col in presentes(config.COV_ANUALES):
        rellenar(col, "mismo municipio-año", panel.groupby(["DIVIPOLA", "anio"])[col].transform("median"))
        anual = panel.groupby(["DIVIPOLA", "anio"])[col].median()
        anual = anual.groupby(level="DIVIPOLA").ffill().groupby(level="DIVIPOLA").bfill()
        rellenar(col, "años vecinos del municipio", pd.Series(anual.reindex(claves).to_numpy(), index=panel.index))

    departamento = panel["DIVIPOLA"].str[:2]
    for col in presentes(config.COV_ESTATICAS + config.COV_ANUALES):
        grupos = [departamento, panel["anio"]] if col in config.COV_ANUALES else [departamento]
        rellenar(col, "mediana del departamento", panel.groupby(grupos)[col].transform("median"))
    for col in presentes(config.COV_NUMERICAS):
        rellenar(col, "mediana nacional", panel[col].median())

    banderas = []
    for col, mascara in faltaba.items():
        panel[f"{col}_imputado"] = mascara.astype(int)
        banderas.append(f"{col}_imputado")

    salida = {}
    for nombre, parte in panel.groupby("_conjunto", sort=False):
        columnas = list(datos[nombre].columns) + banderas
        # Al concatenar, columnas enteras del entrenamiento pasan a float (NaN en pruebas): se restauran.
        salida[nombre] = (parte.sort_values("_orden")[columnas].reset_index(drop=True)
                          .astype(datos[nombre].dtypes.to_dict()))
    return salida, pd.DataFrame(registro, columns=["columna", "regla", "imputados"])


def preparar_datos(datos, verbose=True):
    """Tipos → duplicados → objetivo → imputación. Devuelve (datos_limpios, registro)."""
    datos = {k: normalizar_tipos(v) for k, v in datos.items()}
    registro = {"duplicados": {}}
    for nombre in list(datos):
        datos[nombre], registro["duplicados"][nombre] = deduplicar(datos[nombre])
    datos["entrenamiento"], registro["objetivo"] = limpiar_objetivo(datos["entrenamiento"])
    datos, registro["imputacion"] = imputar_covariables(datos)
    if verbose:
        imprimir_registro(registro)
    return datos, registro


def imprimir_registro(registro):
    print("Tratamiento aplicado")
    for nombre, d in registro["duplicados"].items():
        if d["filas_identicas"] or d["llaves_en_conflicto"]:
            print(f"  {nombre}: {d['filas_identicas']} filas idénticas eliminadas, "
                  f"{d['llaves_en_conflicto']} llaves en conflicto resueltas")
    for clave, n in registro["objetivo"].items():
        if n:
            print(f"  objetivo - {clave.replace('_', ' ')}: {n:,}")
    for fila in registro["imputacion"].itertuples():
        print(f"  {fila.columna}: {fila.imputados:,} ({fila.regla})")
    if (not any(d["filas_identicas"] or d["llaves_en_conflicto"] for d in registro["duplicados"].values())
            and not any(registro["objetivo"].values()) and registro["imputacion"].empty):
        print("  Sin cambios: los datos venían limpios.")


class Transformador:
    """log1p de las asimétricas, recorte por percentiles y escalado opcional, ajustados con la historia.

    - Recorte (winsorización) solo en covariables municipales: el ONI de prueba (El Niño 2023, ~2.0)
      está fuera del rango de entrenamiento y recortarlo borraría la señal.
    - Escalado: innecesario para árboles; útil para modelos lineales ("estandar" o "robusto").
    - Las columnas log quedan renombradas como log_<col>.
    """

    def __init__(self, winsorizar=config.TRATAMIENTO["winsorizar"], escalar=config.TRATAMIENTO["escalar"],
                 columnas_log=tuple(config.COV_LOG), columnas_recorte=tuple(config.COV_ESTATICAS + config.COV_ANUALES),
                 columnas_escala=tuple(config.COV_NUMERICAS)):
        self.winsorizar, self.escalar = winsorizar, escalar
        self.columnas_log, self.columnas_recorte, self.columnas_escala = columnas_log, columnas_recorte, columnas_escala

    def _log(self, df):
        df = df.copy()
        for c in self.columnas_log:
            if c in df:
                df[c] = np.log1p(df[c].clip(lower=0))
        return df

    def fit(self, df):
        x = self._log(df)
        self.limites_ = {}
        if self.winsorizar:
            for c in self.columnas_recorte:
                if c in x:
                    self.limites_[c] = tuple(x[c].quantile(list(self.winsorizar)))
                    x[c] = x[c].clip(*self.limites_[c])
        cols = [c for c in self.columnas_escala if c in x]
        if self.escalar == "estandar":
            self.centro_, self.escala_ = x[cols].mean(), x[cols].std()
        elif self.escalar == "robusto":
            self.centro_ = x[cols].median()
            self.escala_ = x[cols].quantile(0.75) - x[cols].quantile(0.25)
        elif self.escalar is not None:
            raise ValueError(f"escalar debe ser None, 'estandar' o 'robusto' (no {self.escalar!r})")
        if self.escalar:
            self.escala_ = self.escala_.replace(0, 1)
        return self

    def transform(self, df):
        x = self._log(df)
        for c, (lo, hi) in self.limites_.items():
            x[c] = x[c].clip(lo, hi)
        if self.escalar:
            cols = list(self.centro_.index)
            x[cols] = (x[cols] - self.centro_) / self.escala_
        return x.rename(columns={c: f"log_{c}" for c in self.columnas_log if c in x})

    def fit_transform(self, df):
        return self.fit(df).transform(df)
