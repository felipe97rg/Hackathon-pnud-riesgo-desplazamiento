"""Exploración del objetivo y de su relación con las covariables (sobre entrenamiento ya tratado).

Preguntas que responde:
- ¿Qué tan raro es un evento y qué tan asimétrica es la magnitud? (resumen_objetivo)
- ¿Dónde y cuándo ocurren? (tasa_por: año, mes, departamento, fase ENSO; tasas_municipales)
- ¿Qué covariables ordenan el riesgo por sí solas? (auc_univariado)
- ¿Cuánto rinde la climatología ingenua en un corte temporal? (linea_base_climatologia) → piso a superar.
- ¿Hay persistencia mes a mes? (persistencia) — informativa: en prueba no hay rezagos.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from hackaton import config
from hackaton.evaluacion import rmse_log, winkler_log

EV, PER = config.OBJETIVO_EVENTO, config.OBJETIVO_PERSONAS


def _con_derivadas(tr):
    return tr.assign(departamento=tr["DIVIPOLA"].str[:2], log1p_personas=np.log1p(tr[PER]))


def resumen_objetivo(tr):
    """Una fila con las cifras clave del objetivo y los puntajes de la predicción nula."""
    y, ev = tr[PER], tr[EV]
    pos = y[y > 0]
    por_mun = tr.groupby("DIVIPOLA")[EV].sum().sort_values(ascending=False)
    top = max(1, int(round(0.10 * len(por_mun))))
    return pd.DataFrame([{
        "filas": len(tr),
        "tasa_evento": ev.mean(),
        "pct_filas_personas_cero": 100 * (y == 0).mean(),
        "pct_eventos_sin_personas": 100 * ((ev == 1) & (y == 0)).sum() / max(ev.sum(), 1),
        "personas_mediana_si_hay": pos.median(),
        "personas_p90_si_hay": pos.quantile(0.9),
        "personas_p99_si_hay": pos.quantile(0.99),
        "personas_max": y.max(),
        "media_log1p_personas": np.log1p(y).mean(),
        "rmse_log_nulo": rmse_log(y, np.zeros(len(y))),
        "winkler_nulo": winkler_log(y, 0.0, 0.0),
        "municipios_sin_eventos": int((por_mun == 0).sum()),
        "pct_eventos_top10pct_municipios": 100 * por_mun.iloc[:top].sum() / max(por_mun.sum(), 1),
    }]).round(4)


def tasa_por(tr, por):
    """Tasa de eventos y magnitud media por grupo ('anio', 'mes', 'departamento', 'fase_enso' o lista)."""
    return (_con_derivadas(tr).groupby(por)
            .agg(filas=(EV, "size"), eventos=(EV, "sum"), tasa_evento=(EV, "mean"),
                 media_log1p_personas=("log1p_personas", "mean"), personas_total=(PER, "sum"))
            .reset_index().round(4))


def tasas_municipales(tr):
    """Por municipio: meses observados, eventos, tasa y magnitud media (base de la climatología)."""
    return (_con_derivadas(tr).groupby("DIVIPOLA")
            .agg(meses=(EV, "size"), eventos=(EV, "sum"), tasa_evento=(EV, "mean"),
                 media_log1p_personas=("log1p_personas", "mean"))
            .reset_index())


def auc_univariado(tr, columnas=None):
    """AUC de cada covariable sola para tiene_evento y correlación de Spearman con log1p(personas).

    AUC < 0.5 significa relación inversa; se reporta también auc_ajustado = max(AUC, 1 − AUC).
    """
    columnas = columnas or [c for c in config.COV_NUMERICAS if c in tr]
    filas = []
    for col in columnas:
        validos = tr[[col, EV, PER]].dropna()
        if validos[col].nunique() < 2:
            continue
        auc = roc_auc_score(validos[EV], validos[col])
        rho = spearmanr(validos[col], np.log1p(validos[PER])).statistic
        filas.append({"columna": col, "auc": auc, "auc_ajustado": max(auc, 1 - auc),
                      "direccion": "+" if auc >= 0.5 else "-", "spearman_log1p_personas": rho})
    return pd.DataFrame(filas).sort_values("auc_ajustado", ascending=False).round(4).reset_index(drop=True)


def linea_base_climatologia(tr, corte="2021-09-01", fuerza=12):
    """AUC de climatologías ingenuas: tasas calculadas hasta 'corte' y evaluadas en los meses posteriores.

    Tasas suavizadas hacia la tasa global con 'fuerza' meses equivalentes (Bayes empírico simple).
    Es el piso que cualquier modelo debe superar en validación.
    """
    df = _con_derivadas(tr)
    hist, futuro = df[df["fecha"] <= pd.Timestamp(corte)], df[df["fecha"] > pd.Timestamp(corte)]
    if futuro.empty or futuro[EV].nunique() < 2:
        return pd.DataFrame()
    global_ = hist[EV].mean()

    def tasa(claves, previa):
        g = hist.groupby(claves)[EV].agg(["sum", "count"])
        suavizada = (g["sum"] + fuerza * previa) / (g["count"] + fuerza)
        return futuro.join(suavizada.rename("tasa"), on=claves)["tasa"]

    tasa_mun = tasa(["DIVIPOLA"], global_).fillna(global_)
    tasa_mes = futuro["mes"].map(hist.groupby("mes")[EV].mean()).fillna(global_)
    tasa_dep_mes = tasa(["departamento", "mes"], global_).fillna(tasa_mes)
    candidatos = {
        "tasa nacional por mes": tasa_mes,
        "tasa del departamento por mes": tasa_dep_mes,
        "tasa del municipio": tasa_mun,
        "tasa del municipio × estacionalidad departamental": tasa_mun * tasa_dep_mes / tasa_mes.clip(lower=1e-6),
    }
    return pd.DataFrame([{"linea_base": nombre, "auc": roc_auc_score(futuro[EV], pred),
                          "meses_historia": hist["fecha"].nunique(), "meses_evaluados": futuro["fecha"].nunique()}
                         for nombre, pred in candidatos.items()]).round(4)


def persistencia(tr):
    """P(evento | evento el mes anterior) vs P(evento | sin evento el mes anterior)."""
    df = tr.sort_values(["DIVIPOLA", "fecha"])
    previo = df.groupby("DIVIPOLA")[EV].shift(1)
    validos = previo.notna()
    return pd.DataFrame([{
        "p_evento": df.loc[validos, EV].mean(),
        "p_evento_si_hubo_mes_anterior": df.loc[validos & (previo == 1), EV].mean(),
        "p_evento_si_no_hubo_mes_anterior": df.loc[validos & (previo == 0), EV].mean(),
    }]).round(4)


def mezcla_tipos(tr):
    """Participación de cada tipo de evento en el total reportado."""
    tipos = [c for c in config.HISTORIAL if c.startswith("n_") and c != "n_eventos" and c in tr]
    total = tr[tipos].sum()
    return (pd.DataFrame({"tipo": [t.removeprefix("n_") for t in tipos], "eventos": total.to_numpy()})
            .assign(pct=lambda d: (100 * d["eventos"] / max(d["eventos"].sum(), 1)).round(2))
            .sort_values("eventos", ascending=False).reset_index(drop=True))


def exploracion_completa(tr):
    return {
        "resumen_objetivo": resumen_objetivo(tr),
        "tasa_por_anio": tasa_por(tr, "anio"),
        "tasa_por_mes": tasa_por(tr, "mes"),
        "tasa_por_departamento": tasa_por(tr, "departamento").sort_values("tasa_evento", ascending=False),
        "tasa_por_fase_enso": tasa_por(tr, "fase_enso"),
        "serie_mensual": tasa_por(tr, "fecha"),
        "tasas_municipales": tasas_municipales(tr),
        "auc_univariado": auc_univariado(tr),
        "linea_base_climatologia": linea_base_climatologia(tr),
        "persistencia": persistencia(tr),
        "mezcla_tipos": mezcla_tipos(tr),
    }


def imprimir_resumen(exp):
    r = exp["resumen_objetivo"].iloc[0]
    print("Exploración del objetivo (entrenamiento)")
    print(f"  Tasa de eventos: {r.tasa_evento:.3f} | filas con personas = 0: {r.pct_filas_personas_cero:.1f}% "
          f"| eventos sin personas: {r.pct_eventos_sin_personas:.1f}%")
    print(f"  Personas si hay afectados: mediana {r.personas_mediana_si_hay:,.0f}, p90 {r.personas_p90_si_hay:,.0f}, "
          f"p99 {r.personas_p99_si_hay:,.0f}, máx {r.personas_max:,.0f}")
    print(f"  Predicción nula: RMSE log1p = {r.rmse_log_nulo:.3f} | Winkler = {r.winkler_nulo:.3f}")
    print(f"  Municipios sin ningún evento: {int(r.municipios_sin_eventos)} | el 10% con más eventos concentra "
          f"{r.pct_eventos_top10pct_municipios:.1f}% de los eventos")
    lb = exp["linea_base_climatologia"]
    if not lb.empty:
        mejor = lb.sort_values("auc").iloc[-1]
        print(f"  Línea base climatología (corte 2021-09): mejor AUC {mejor.auc:.3f} ({mejor.linea_base})")
    top = exp["auc_univariado"].head(3)
    print("  Covariables con más señal (AUC sola): " + ", ".join(
        f"{x.columna} {x.auc_ajustado:.3f}{x.direccion}" for x in top.itertuples()))
    p = exp["persistencia"].iloc[0]
    print(f"  Persistencia: P(evento | evento mes anterior) = {p.p_evento_si_hubo_mes_anterior:.3f} "
          f"vs {p.p_evento_si_no_hubo_mes_anterior:.3f} (no usable en prueba, pero indica concentración)")
