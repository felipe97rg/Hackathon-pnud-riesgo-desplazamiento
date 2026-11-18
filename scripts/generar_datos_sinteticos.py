"""Genera datos sintéticos con el MISMO esquema que los CSV del reto, para desarrollar sin los datos reales.

Los valores son inventados, aunque con patrones plausibles (tasa propia por municipio, estacionalidad
regional, efecto ENSO, muchos ceros, magnitudes asimétricas). Sirven para probar el código, NO para sacar
conclusiones sobre Colombia. Si existe data/externos/procesados/geografia.csv se usan 1,102 códigos DIVIPOLA
reales (para probar el geovisor); si no, códigos ficticios con el prefijo de departamento real.

Salida:
    data/sinteticos/         entrenamiento.csv, prueba_equipos.csv, prueba_oculta.csv (limpios)
                             verdad_pruebas.csv → objetivo simulado de las pruebas (tabla en vivo / nota oficial)
    data/sinteticos_sucios/  los mismos con problemas típicos inyectados (nulos, duplicados, DIVIPOLA sin cero,
                             valores centinela, objetivo incoherente…) para probar calidad.py y tratamiento.py

Uso: python scripts/generar_datos_sinteticos.py [--semilla 42]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hackaton import config  # noqa: E402

# (código de departamento, número de municipios, región dominante)
DEPARTAMENTOS = [
    ("05", 125, "Andina"), ("08", 23, "Caribe"), ("11", 1, "Andina"), ("13", 45, "Caribe"),
    ("15", 123, "Andina"), ("17", 27, "Andina"), ("18", 16, "Amazonia"), ("19", 42, "Pacifica"),
    ("20", 25, "Caribe"), ("23", 30, "Caribe"), ("25", 116, "Andina"), ("27", 31, "Pacifica"),
    ("41", 37, "Andina"), ("44", 15, "Caribe"), ("47", 30, "Caribe"), ("50", 29, "Orinoquia"),
    ("52", 64, "Pacifica"), ("54", 40, "Andina"), ("63", 12, "Andina"), ("66", 14, "Andina"),
    ("68", 87, "Andina"), ("70", 26, "Caribe"), ("73", 47, "Andina"), ("76", 42, "Pacifica"),
    ("81", 7, "Orinoquia"), ("85", 19, "Orinoquia"), ("86", 13, "Amazonia"), ("88", 2, "Caribe"),
    ("91", 2, "Amazonia"), ("94", 1, "Amazonia"), ("95", 4, "Amazonia"), ("97", 3, "Amazonia"),
    ("99", 4, "Orinoquia"),
]

# ONI aproximado 2018-01 … 2025-12 (solo para que los sintéticos tengan una forma realista;
# el día D se usa el ONI que viene en los datos).
ONI = [
    -0.9, -0.9, -0.7, -0.5, -0.2, 0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 0.8,
    0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.3, 0.1, 0.2, 0.3, 0.5, 0.5,
    0.5, 0.5, 0.4, 0.2, -0.1, -0.3, -0.4, -0.6, -0.9, -1.2, -1.3, -1.2,
    -1.0, -0.9, -0.8, -0.7, -0.5, -0.4, -0.4, -0.5, -0.7, -0.8, -1.0, -1.0,
    -1.0, -0.9, -1.0, -1.1, -1.0, -0.9, -0.8, -0.9, -1.0, -1.0, -0.9, -0.8,
    -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.1, 1.3, 1.6, 1.8, 1.9, 2.0,
    1.8, 1.5, 1.1, 0.7, 0.4, 0.2, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5,
    -0.6, -0.4, -0.2, -0.1, -0.1, -0.1, -0.1, -0.3, -0.4, -0.5, -0.6, -0.6,
]

# Efecto estacional (escala logit) por mes y región, y sensibilidad a La Niña (−ONI).
ESTACIONALIDAD = {
    "Andina":    [-.4, -.3, .1, .5, .5, 0, -.4, -.4, 0, .5, .6, .1],
    "Caribe":    [-.8, -.9, -.7, -.2, .3, .2, 0, .2, .5, .8, .6, -.2],
    "Pacifica":  [.1, 0, 0, .2, .3, .1, 0, 0, .1, .3, .3, .2],
    "Orinoquia": [-.9, -.9, -.5, .2, .6, .7, .6, .3, .1, 0, -.3, -.7],
    "Amazonia":  [0, .1, .3, .4, .4, .2, 0, -.2, -.2, 0, .1, .1],
}
SENSIBILIDAD_ENSO = {"Andina": .35, "Caribe": .40, "Pacifica": .10, "Orinoquia": .15, "Amazonia": .05}
ELEVACION = {"Andina": (300, 3000), "Caribe": (0, 300), "Pacifica": (0, 1500),
             "Orinoquia": (100, 500), "Amazonia": (100, 400)}


def z(x):
    return (x - x.mean()) / x.std()


def lista_municipios(rng):
    """Códigos DIVIPOLA reales si ya se corrió scripts/externos/01_geografia.py; si no, ficticios por departamento."""
    ruta = config.DIR_EXT_PROCESADOS / "geografia.csv"
    if ruta.exists():
        codigos = pd.read_csv(ruta, dtype={"DIVIPOLA": str})["DIVIPOLA"].to_numpy()
        if len(codigos) >= config.N_MUNICIPIOS:
            codigos = np.sort(rng.choice(codigos, config.N_MUNICIPIOS, replace=False))
            regiones = [config.REGION_POR_DEPARTAMENTO.get(c[:2], "Andina") for c in codigos]
            return pd.DataFrame({"DIVIPOLA": codigos, "dep": [c[:2] for c in codigos], "region": regiones})
    filas = [(f"{dep}{2 * k + 1:03d}", dep, region) for dep, n, region in DEPARTAMENTOS for k in range(n)]
    return pd.DataFrame(filas, columns=["DIVIPOLA", "dep", "region"])


def generar_municipios(rng):
    mun = lista_municipios(rng)
    assert len(mun) == config.N_MUNICIPIOS, len(mun)
    m = len(mun)
    andina = (mun["region"] == "Andina").to_numpy()
    lo = mun["region"].map(lambda r: ELEVACION[r][0]).to_numpy()
    hi = mun["region"].map(lambda r: ELEVACION[r][1]).to_numpy()
    mun["elevacion_media"] = rng.uniform(lo, hi)
    mun["pendiente_media"] = np.where(andina, rng.gamma(6, 2.5, m), rng.gamma(2, 1.5, m)).clip(0.2, 40)
    mun["pct_pendiente_alta"] = (mun["pendiente_media"] * 2 - 10 + rng.normal(0, 5, m)).clip(0, 90)
    remoto = mun["region"].isin(["Pacifica", "Amazonia"]).to_numpy()
    mun["nbi_pct"] = (rng.beta(2, 4, m) * 100 + remoto * 20).clip(1, 99)
    mun["densidad_2018"] = np.exp(rng.normal(3.7, 1.2, m))
    mun["ingresos_2018"] = np.exp(rng.normal(np.log(150_000), 0.8, m))
    mun["inversion_base"] = np.exp(rng.normal(np.log(2e8), 1.3, m))
    mun["logit_base"] = (-2.6 + 0.6 * z(np.log(mun["densidad_2018"])) + 0.4 * z(mun["pendiente_media"])
                         + 0.2 * z(mun["nbi_pct"]) + rng.normal(0, 0.9, m))
    mun["efecto_magnitud"] = 0.35 * z(np.log(mun["densidad_2018"])) + rng.normal(0, 0.5, m)
    mun["w_masa"] = 0.5 + mun["pct_pendiente_alta"] / 20
    mun["w_inundacion"] = (1.5 - mun["elevacion_media"] / 2000).clip(0.1, None)
    return mun


def generar_panel(mun, rng):
    fechas = pd.date_range("2018-01-01", "2025-12-01", freq="MS")
    oni = pd.Series(ONI, index=fechas)
    df = mun.merge(pd.DataFrame({"fecha": fechas}), how="cross")
    n = len(df)
    df["anio"], df["mes"] = df["fecha"].dt.year, df["fecha"].dt.month
    anios = df["anio"].to_numpy() - 2018
    df["ONI"] = df["fecha"].map(oni).to_numpy()
    df["fase_enso"] = np.select([df["ONI"] >= 0.5, df["ONI"] <= -0.5], ["El Nino", "La Nina"], "Neutral")

    # Covariables con su temporalidad del diccionario
    df["densidad_poblacional"] = df["densidad_2018"] * 1.01 ** anios
    df["ingresos_tributarios"] = df["ingresos_2018"] * 1.05 ** np.minimum(anios, 6)   # 2025 repite 2024
    ruido_inv = pd.Series(rng.lognormal(0, 0.3, len(mun) * 3)).to_numpy().reshape(len(mun), 3)
    idx_mun = df.index // len(fechas)
    df["inversion_gestion_riesgo"] = df["inversion_base"] * ruido_inv[idx_mun, np.minimum(anios, 2)]

    # Objetivo
    estacional = np.array([ESTACIONALIDAD[r][m - 1] for r, m in zip(df["region"], df["mes"])])
    sens = df["region"].map(SENSIBILIDAD_ENSO).to_numpy()
    logit = df["logit_base"] + estacional - sens * df["ONI"] + 0.05 * anios
    p = 1 / (1 + np.exp(-logit))
    evento = rng.random(n) < p
    z_mag = rng.normal(2.5 + df["efecto_magnitud"] + 0.3 * np.clip(-df["ONI"], 0, None), 1.2)
    personas = np.where(evento, np.round(np.expm1(z_mag.clip(0))), 0)
    solo_viviendas = evento & (rng.random(n) < 0.12)
    personas[solo_viviendas] = 0
    df["tiene_evento"] = evento.astype(int)
    df["personas_desplazadas"] = personas.astype(int)

    # Historial
    n_ev = np.where(evento, 1 + rng.poisson(0.25, n), 0)
    df["n_eventos"] = n_ev
    df["familias_desplazadas"] = np.round(personas / 3.4).astype(int)
    df["viv_destruidas"] = rng.poisson(personas / 200)
    df["viv_averiadas"] = np.where(solo_viviendas, 1 + rng.poisson(3, n), rng.poisson(personas / 12))
    pesos = np.column_stack([df["w_masa"], df["w_inundacion"], np.full(n, .6), np.full(n, .3),
                             np.full(n, .15), np.full(n, .4)])
    pesos = pesos / pesos.sum(axis=1, keepdims=True)
    tipos = np.array([rng.multinomial(k, w) if k else np.zeros(6, int) for k, w in zip(n_ev, pesos)])
    for j, col in enumerate(["n_movimiento_masa", "n_inundacion", "n_vendaval", "n_creciente_subita",
                             "n_avenida_torrencial", "n_otros"]):
        df[col] = tipos[:, j]
    return df


def ensuciar(partes, rng):
    """Inyecta problemas típicos de datos reales. Devuelve copias; no toca las partes limpias."""
    partes = {k: v.copy() for k, v in partes.items()}
    tr = partes["entrenamiento"]
    municipios = tr["DIVIPOLA"].unique()
    sin_nbi = rng.choice(municipios, 3, replace=False)
    centinela = rng.choice(municipios, 2, replace=False)
    for df in partes.values():
        dispersos = rng.choice(df.index, size=int(0.015 * len(df)), replace=False)
        df.loc[dispersos, "ingresos_tributarios"] = np.nan                       # nulos dispersos
        df.loc[df["DIVIPOLA"].isin(sin_nbi), "nbi_pct"] = np.nan                 # municipio sin dato
        df.loc[df["DIVIPOLA"].isin(centinela), "elevacion_media"] = -9999        # valor centinela
    sin_inversion = rng.choice(municipios, 20, replace=False)
    tr.loc[tr["DIVIPOLA"].isin(sin_inversion) & (tr["anio"] == 2019), "inversion_gestion_riesgo"] = np.nan
    extremo = tr["DIVIPOLA"].isin(rng.choice(municipios, 1)) & (tr["anio"] == 2020)
    tr.loc[extremo, "densidad_poblacional"] *= 50                                # outlier posible pero raro
    mes = tr["fecha"] == "2019-06-01"
    tr.loc[tr.index[mes][:15], "ONI"] = np.nan                                   # ONI faltante en un mes
    tr.loc[tr.index[tr["fecha"] == "2021-01-01"][:10], "fase_enso"] = "Neutral"  # fase incoherente con ONI
    sin_evento = tr.index[tr["tiene_evento"] == 0][:5]
    tr.loc[sin_evento, "personas_desplazadas"] = 7                               # afectación sin evento
    conflicto = tr.sample(2, random_state=1).assign(personas_desplazadas=999)
    partes["entrenamiento"] = pd.concat([tr, tr.sample(6, random_state=0), conflicto], ignore_index=True)
    po = partes["prueba_oculta"]
    partes["prueba_oculta"] = pd.concat([po, po.sample(1, random_state=0)], ignore_index=True)
    pe = partes["prueba_equipos"]
    pe["DIVIPOLA"] = pe["DIVIPOLA"].str.lstrip("0")                              # como si pasara por Excel
    return partes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--semilla", type=int, default=config.SEMILLA)
    args = parser.parse_args()
    rng = np.random.default_rng(args.semilla)

    df = generar_panel(generar_municipios(rng), rng)
    df["fecha"] = df["fecha"].dt.strftime("%Y-%m-01")
    for c in config.COVARIABLES:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)

    objetivo = [config.OBJETIVO_EVENTO, config.OBJETIVO_PERSONAS]
    cortes = {
        "entrenamiento": ("2018-01-01", "2022-09-01", config.ID + objetivo + config.HISTORIAL + config.COVARIABLES),
        "prueba_equipos": ("2022-10-01", "2024-04-01", config.ID + config.COVARIABLES),
        "prueba_oculta": ("2024-05-01", "2025-12-01", config.ID + config.COVARIABLES),
    }
    partes = {nombre: df[(df["fecha"] >= ini) & (df["fecha"] <= fin)][columnas].reset_index(drop=True)
              for nombre, (ini, fin, columnas) in cortes.items()}
    for nombre, parte in partes.items():
        tasa = df.loc[(df["fecha"] >= cortes[nombre][0]) & (df["fecha"] <= cortes[nombre][1]), "tiene_evento"].mean()
        print(f"{nombre:15s} {len(parte):>7,} filas  tasa de eventos {tasa:.3f}")

    pruebas = df[df["fecha"] >= "2022-10-01"]
    verdad = pruebas.assign(conjunto=np.where(pruebas["fecha"] < "2024-05-01", "prueba_equipos", "prueba_oculta"))[
        ["DIVIPOLA", "fecha", "conjunto"] + objetivo]
    for carpeta, contenido in ((config.DIR_SINTETICOS, partes), (config.DIR_SINTETICOS_SUCIOS, ensuciar(partes, rng))):
        carpeta.mkdir(parents=True, exist_ok=True)
        for nombre, parte in contenido.items():
            parte.to_csv(carpeta / config.ARCHIVOS_RETO[nombre], index=False)
        verdad.to_csv(carpeta / "verdad_pruebas.csv", index=False)
        print(f"Escrito en {carpeta}")


if __name__ == "__main__":
    main()
