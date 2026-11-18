"""Diagnóstico de calidad: esquema, nulos, duplicados, completitud del panel, reglas de dominio,
consistencia temporal, outliers y cambio de distribución entre entrenamiento y prueba.

Las funciones solo DIAGNOSTICAN (devuelven DataFrames, no modifican nada). La corrección está en
tratamiento.py. Uso rápido:

    diag = diagnostico_completo(datos)
    imprimir_resumen(diag)
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from hackaton import config

LLAVE = ["DIVIPOLA", "fecha"]


def _panel_covariables(datos):
    columnas = ["DIVIPOLA", "fecha", "anio"] + config.COVARIABLES
    partes = [df[[c for c in columnas if c in df]].assign(conjunto=nombre) for nombre, df in datos.items()]
    return pd.concat(partes, ignore_index=True)


def _escala(col, x):
    """log1p para las covariables muy asimétricas; el resto sin transformar."""
    return np.log1p(x.clip(lower=0)) if col in config.COV_LOG else x


def fase_esperada(oni):
    """Fase ENSO que corresponde al ONI según el umbral del diccionario (NaN si no hay ONI)."""
    fase = np.select([oni >= config.UMBRAL_ENSO, oni <= -config.UMBRAL_ENSO], ["El Nino", "La Nina"], "Neutral")
    return pd.Series(fase, index=oni.index).where(oni.notna())


# ── Estructura ───────────────────────────────────────────────────────────────

def resumen_esquema(datos):
    filas = []
    for nombre, df in datos.items():
        esperadas = set(config.ID + config.COVARIABLES)
        if nombre == "entrenamiento":
            esperadas |= {config.OBJETIVO_EVENTO, config.OBJETIVO_PERSONAS, *config.HISTORIAL}
        filas.append({
            "archivo": nombre,
            "filas": len(df),
            "filas_esperadas": config.FILAS_ESPERADAS.get(nombre),
            "municipios": df["DIVIPOLA"].nunique(),
            "meses": df["fecha"].nunique(),
            "desde": df["fecha"].min(),
            "hasta": df["fecha"].max(),
            "columnas_faltantes": ", ".join(sorted(esperadas - set(df.columns))) or "-",
            "columnas_extra": ", ".join(sorted(set(df.columns) - esperadas)) or "-",
        })
    return pd.DataFrame(filas)


def nulos(datos):
    filas = []
    for nombre, df in datos.items():
        conteo = df.isna().sum()
        for columna, n in conteo[conteo > 0].items():
            filas.append({"archivo": nombre, "columna": columna, "nulos": int(n),
                          "pct": round(100 * n / len(df), 2),
                          "municipios_afectados": df.loc[df[columna].isna(), "DIVIPOLA"].nunique()})
    return pd.DataFrame(filas, columns=["archivo", "columna", "nulos", "pct", "municipios_afectados"])


def duplicados(datos):
    filas = []
    for nombre, df in datos.items():
        identicas = int(df.duplicated().sum())
        llaves = int(df.duplicated(LLAVE).sum())
        filas.append({"archivo": nombre, "filas_identicas": identicas, "llaves_repetidas": llaves,
                      "llaves_en_conflicto": int(df.drop_duplicates().duplicated(LLAVE).sum())})
    return pd.DataFrame(filas)


def completitud_panel(datos):
    """¿Están todos los municipios en todos los meses y en todos los archivos? ¿Llaves bien formadas?"""
    todos = set().union(*(set(df["DIVIPOLA"]) for df in datos.values()))
    filas = []
    for nombre, df in datos.items():
        unicas = df.drop_duplicates(LLAVE)
        filas.append({
            "archivo": nombre,
            "combinaciones_faltantes": df["DIVIPOLA"].nunique() * df["fecha"].nunique() - len(unicas),
            "municipios_ausentes": len(todos - set(df["DIVIPOLA"])),
            "divipola_mal_formado": int((~df["DIVIPOLA"].astype(str).str.fullmatch(r"\d{5}")).sum()),
            "fecha_ilegible": int(df["fecha"].isna().sum()),
            "fecha_no_dia_1": int((df["fecha"].dt.day != 1).sum()),
            "anio_mes_inconsistente": int(((df["anio"] != df["fecha"].dt.year)
                                           | (df["mes"] != df["fecha"].dt.month)).sum()),
        })
    return pd.DataFrame(filas)


# ── Reglas de negocio ────────────────────────────────────────────────────────

def reglas_dominio(datos):
    """Violaciones de reglas que deben cumplirse siempre. tipo='error' es dato imposible; 'aviso' es raro."""
    filas = []

    def registrar(archivo, regla, mascara, tipo="error", total=None):
        n = int(mascara.sum())
        filas.append({"archivo": archivo, "regla": regla, "tipo": tipo, "violaciones": n,
                      "pct": round(100 * n / (total or len(mascara)), 3)})

    for nombre, df in datos.items():
        for col, (lo, hi) in config.RANGOS_VALIDOS.items():
            if col in df:
                x = df[col]
                fuera = pd.Series(False, index=df.index)
                if lo is not None:
                    fuera |= x < lo
                if hi is not None:
                    fuera |= x > hi
                registrar(nombre, f"{col} en [{lo}, {hi}]", fuera)
        if "fase_enso" in df:
            registrar(nombre, "fase_enso con categoría válida", df["fase_enso"].notna() & ~df["fase_enso"].isin(config.FASES_ENSO))
            if "ONI" in df:
                esperada = fase_esperada(df["ONI"])
                registrar(nombre, "fase_enso coherente con ONI",
                          esperada.notna() & df["fase_enso"].notna() & (df["fase_enso"] != esperada))

    tr = datos["entrenamiento"]
    ev, per = config.OBJETIVO_EVENTO, config.OBJETIVO_PERSONAS
    if {ev, per} <= set(tr.columns):
        viv = tr[[c for c in ("viv_destruidas", "viv_averiadas") if c in tr]].fillna(0).sum(axis=1)
        afectacion = (tr[per].fillna(0) > 0) | (viv > 0)
        registrar("entrenamiento", "tiene_evento en {0, 1}", tr[ev].notna() & ~tr[ev].isin([0, 1]))
        for col in [per] + [c for c in config.HISTORIAL if c in tr]:
            registrar("entrenamiento", f"{col} >= 0", tr[col] < 0)
        registrar("entrenamiento", "afectación (personas o viviendas) implica tiene_evento=1",
                  (tr[ev] == 0) & afectacion)
        registrar("entrenamiento", "tiene_evento=1 sin personas ni viviendas reportadas",
                  (tr[ev] == 1) & ~afectacion, tipo="aviso")
        if "n_eventos" in tr:
            registrar("entrenamiento", "tiene_evento=1 con n_eventos=0", (tr[ev] == 1) & (tr["n_eventos"] == 0), "aviso")
            tipos = [c for c in config.HISTORIAL if c.startswith("n_") and c != "n_eventos" and c in tr]
            registrar("entrenamiento", "suma de tipos de evento = n_eventos",
                      tr[tipos].sum(axis=1) != tr["n_eventos"], "aviso")
    return pd.DataFrame(filas)


def consistencia_temporal(datos):
    """Las estáticas no deben cambiar por municipio, las anuales no dentro del año, las nacionales no entre municipios."""
    panel = _panel_covariables(datos)
    filas = []
    for col in config.COV_ESTATICAS:
        n = (panel.groupby("DIVIPOLA")[col].nunique() > 1).sum()
        filas.append({"columna": col, "esperado": "constante por municipio", "grupos_inconsistentes": int(n)})
    for col in config.COV_ANUALES:
        n = (panel.groupby(["DIVIPOLA", "anio"])[col].nunique() > 1).sum()
        filas.append({"columna": col, "esperado": "constante por municipio-año", "grupos_inconsistentes": int(n)})
    for col in config.COV_NACIONALES + ["fase_enso"]:
        n = (panel.groupby("fecha")[col].nunique() > 1).sum()
        filas.append({"columna": col, "esperado": "igual para todos los municipios en un mes",
                      "grupos_inconsistentes": int(n)})
    return pd.DataFrame(filas)


# ── Distribuciones ───────────────────────────────────────────────────────────

def _base_por_nivel(panel, col):
    """Una observación por unidad real de variación (evita contar 96 veces un valor estático)."""
    if col in config.COV_ESTATICAS:
        return "municipio", panel.drop_duplicates("DIVIPOLA")[col]
    if col in config.COV_ANUALES:
        return "municipio-año", panel.drop_duplicates(["DIVIPOLA", "anio"])[col]
    return "mes", panel.drop_duplicates("fecha")[col]


def _marcas_outlier(v, k_iqr=3.0, z_max=3.5):
    q1, q3 = v.quantile([0.25, 0.75])
    iqr = q3 - q1
    fuera_iqr = (v < q1 - k_iqr * iqr) | (v > q3 + k_iqr * iqr)
    mad = 1.4826 * (v - v.median()).abs().median()
    z = (v - v.median()) / mad if mad > 0 else pd.Series(0.0, index=v.index)
    return fuera_iqr, z.abs() > z_max, z


def outliers(datos, k_iqr=3.0, z_max=3.5):
    """Outliers por IQR extremo (k=3) y z robusto (MAD > 3.5). Variables asimétricas en escala log1p.

    Un outlier NO es necesariamente un error (Bogotá tiene densidad extrema de verdad); los imposibles los
    detecta reglas_dominio.
    """
    panel = _panel_covariables(datos)
    filas = []

    def resumir(nombre, nivel, x, log):
        x = x.dropna()
        v = np.log1p(x.clip(lower=0)) if log else x
        fuera_iqr, fuera_mad, _ = _marcas_outlier(v, k_iqr, z_max)
        filas.append({"columna": nombre, "nivel": nivel, "escala": "log1p" if log else "original", "n": len(x),
                      "outliers_iqr": int(fuera_iqr.sum()), "outliers_mad": int(fuera_mad.sum()),
                      "min": x.min(), "p01": x.quantile(0.01), "p50": x.median(),
                      "p99": x.quantile(0.99), "max": x.max()})

    for col in config.COV_NUMERICAS:
        if col in panel:
            nivel, x = _base_por_nivel(panel, col)
            resumir(col, nivel, x, col in config.COV_LOG)
    tr = datos["entrenamiento"]
    if config.OBJETIVO_PERSONAS in tr:
        positivos = tr.loc[tr[config.OBJETIVO_PERSONAS] > 0, config.OBJETIVO_PERSONAS]
        resumir(config.OBJETIVO_PERSONAS, "municipio-mes con personas > 0", positivos, True)
    return pd.DataFrame(filas)


def top_outliers(datos, columna, n=10):
    """Municipios (o meses) con los valores más extremos de una covariable, por z robusto."""
    panel = _panel_covariables(datos)
    nivel, x = _base_por_nivel(panel, columna)
    x = x.dropna()
    _, _, z = _marcas_outlier(_escala(columna, x))
    claves = ["fecha"] if nivel == "mes" else ["DIVIPOLA"] + (["anio"] if nivel == "municipio-año" else [])
    tabla = panel.loc[x.index, claves].assign(valor=x, z_robusto=z.round(2))
    return tabla.reindex(z.abs().sort_values(ascending=False).index).head(n)


def cambio_distribucion(datos):
    """Compara cada covariable entre entrenamiento y cada archivo de prueba.

    dif_estandarizada = (media prueba − media entreno) / desv. entreno (|d| > 0.25 merece atención).
    pct_fuera_rango = % de filas de prueba fuera del [mín, máx] de entrenamiento (los árboles no extrapolan).
    """
    filas = []
    tr = datos["entrenamiento"]
    for col in config.COV_NUMERICAS:
        if col not in tr:
            continue
        a = _escala(col, tr[col].dropna())
        for nombre in ("prueba_equipos", "prueba_oculta"):
            if col not in datos[nombre]:
                continue
            b_raw = datos[nombre][col].dropna()
            b = _escala(col, b_raw)
            filas.append({
                "columna": col, "comparacion": f"entrenamiento vs {nombre}",
                "escala": "log1p" if col in config.COV_LOG else "original",
                "media_entreno": a.mean(), "media_prueba": b.mean(),
                "dif_estandarizada": (b.mean() - a.mean()) / a.std() if a.std() > 0 else 0.0,
                "ks": ks_2samp(a, b).statistic,
                "pct_fuera_rango": 100 * ((b_raw < tr[col].min()) | (b_raw > tr[col].max())).mean(),
            })
    return pd.DataFrame(filas).round(4)


# ── Todo junto ───────────────────────────────────────────────────────────────

def diagnostico_completo(datos):
    return {
        "esquema": resumen_esquema(datos),
        "nulos": nulos(datos),
        "duplicados": duplicados(datos),
        "panel": completitud_panel(datos),
        "reglas": reglas_dominio(datos),
        "consistencia": consistencia_temporal(datos),
        "outliers": outliers(datos),
        "cambio_distribucion": cambio_distribucion(datos),
    }


def imprimir_resumen(diag):
    """Resumen corto en consola: [OK] sin problemas, [!] revisar, [i] informativo."""
    def linea(ok, texto):
        print(f"  {'[OK]' if ok else '[!] '} {texto}")

    print("Diagnóstico de calidad")
    e = diag["esquema"]
    filas_raras = e[e["filas"] != e["filas_esperadas"]]
    linea(filas_raras.empty, "Filas por archivo: " + ", ".join(f"{r.archivo}={r.filas:,}" for r in e.itertuples()))
    faltantes = e[e["columnas_faltantes"] != "-"]
    linea(faltantes.empty, "Columnas completas" if faltantes.empty else
          "Columnas faltantes: " + "; ".join(f"{r.archivo}: {r.columnas_faltantes}" for r in faltantes.itertuples()))
    n = diag["nulos"]
    linea(n.empty, "Sin nulos" if n.empty else
          f"Nulos en {len(n)} columna-archivo, {n['nulos'].sum():,} celdas (más afectada: "
          f"{n.sort_values('nulos').iloc[-1]['columna']})")
    d = diag["duplicados"]
    linea(d[["filas_identicas", "llaves_repetidas"]].to_numpy().sum() == 0,
          "Duplicados: " + ", ".join(f"{r.archivo} {r.filas_identicas} idénticas / {r.llaves_en_conflicto} en conflicto"
                                     for r in d.itertuples()))
    p = diag["panel"]
    problemas = p.drop(columns="archivo").to_numpy().sum()
    linea(problemas == 0, "Panel completo y llaves bien formadas" if problemas == 0 else
          f"Panel/llaves con {problemas:,} problemas (ver tabla 'panel')")
    r = diag["reglas"]
    errores = r[(r["tipo"] == "error") & (r["violaciones"] > 0)]
    linea(errores.empty, "Reglas de dominio OK" if errores.empty else
          "Reglas violadas: " + "; ".join(f"{x.regla} ({x.archivo}: {x.violaciones})" for x in errores.itertuples()))
    avisos = r[(r["tipo"] == "aviso") & (r["violaciones"] > 0)]
    for x in avisos.itertuples():
        print(f"  [i]  {x.regla}: {x.violaciones:,} ({x.pct}%)")
    c = diag["consistencia"]
    inconsistentes = c[c["grupos_inconsistentes"] > 0]
    linea(inconsistentes.empty, "Temporalidad de covariables coherente" if inconsistentes.empty else
          "Temporalidad incoherente en: " + ", ".join(inconsistentes["columna"]))
    o = diag["outliers"]
    con_out = o[o["outliers_iqr"] > 0]
    print("  [i]  Outliers (IQR extremo): " + (", ".join(f"{x.columna}={x.outliers_iqr}" for x in con_out.itertuples())
                                              or "ninguno"))
    cd = diag["cambio_distribucion"]
    deriva = cd[(cd["dif_estandarizada"].abs() > 0.25) | (cd["pct_fuera_rango"] > 5)]
    print("  [i]  Cambio de distribución relevante: " + (", ".join(
        f"{x.columna} ({x.comparacion.split()[-1]}: d={x.dif_estandarizada:+.2f}, fuera de rango {x.pct_fuera_rango:.0f}%)"
        for x in deriva.itertuples()) or "ninguno"))
