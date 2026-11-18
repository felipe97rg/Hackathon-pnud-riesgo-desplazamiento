"""Gráficos de exploración (Plotly) y reporte HTML autocontenido.

Convenciones: una serie → azul; los tres archivos siempre con el mismo color (entrenamiento azul,
prueba_equipos naranja, prueba_oculta aqua); nunca doble eje (medidas distintas van en paneles separados);
líneas de 2 px, grilla tenue, tooltip en cada marca. Cada gráfico tiene su tabla equivalente en el reporte/CSV.
"""
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hackaton import config

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLOR_CONJUNTO = {"entrenamiento": AZUL, "prueba_equipos": NARANJA, "prueba_oculta": AQUA}
SUPERFICIE, TINTA, TINTA_2, MUTED, GRILLA, EJE, NEUTRO = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec")
FUENTE = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _estilo(fig, titulo, alto=360):
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=15, color=TINTA), x=0, xanchor="left"),
        height=alto, margin=dict(l=60, r=24, t=56, b=48),
        paper_bgcolor=SUPERFICIE, plot_bgcolor=SUPERFICIE,
        font=dict(family=FUENTE, size=12, color=TINTA_2),
        hoverlabel=dict(bgcolor="white", font=dict(family=FUENTE, color=TINTA)),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right", title=None),
        barcornerradius=4, bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=EJE, ticks="outside", tickcolor=EJE, title_font_color=MUTED)
    fig.update_yaxes(gridcolor=GRILLA, zeroline=False, linecolor=EJE, title_font_color=MUTED)
    return fig


def barras(df, x, y, titulo, eje_x=None, eje_y=None, horizontal=False, formato=".3f"):
    """Barras de una sola serie (azul)."""
    if horizontal:
        traza = go.Bar(x=df[y], y=df[x].astype(str), orientation="h", marker_color=AZUL,
                       hovertemplate=f"%{{y}}: %{{x:{formato}}}<extra></extra>")
    else:
        traza = go.Bar(x=df[x].astype(str), y=df[y], marker_color=AZUL,
                       hovertemplate=f"%{{x}}: %{{y:{formato}}}<extra></extra>")
    fig = go.Figure(traza)
    fig.update_xaxes(title=eje_y if horizontal else eje_x)
    fig.update_yaxes(title=eje_x if horizontal else eje_y)
    alto = max(360, 18 * len(df) + 120) if horizontal else 360
    if horizontal:
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor=GRILLA)
    return _estilo(fig, titulo, alto)


def histograma(valores, titulo, eje_x, bins=40):
    fig = go.Figure(go.Histogram(x=valores, nbinsx=bins, marker_color=AZUL,
                                 hovertemplate="%{x}: %{y:,} observaciones<extra></extra>"))
    fig.update_layout(bargap=0.05)
    fig.update_xaxes(title=eje_x)
    fig.update_yaxes(title="observaciones")
    return _estilo(fig, titulo)


def serie_nacional(serie_mensual, datos):
    """Arriba: tasa mensual de eventos (entrenamiento). Abajo: ONI de todos los periodos. Mismo eje x."""
    oni = (pd.concat([df[["fecha", "ONI"]] for df in datos.values()]).dropna()
           .groupby("fecha", as_index=False)["ONI"].median())
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Tasa mensual de eventos (entrenamiento)",
                                        "ONI (El Niño ≥ 0.5, La Niña ≤ −0.5)"))
    fig.add_trace(go.Scatter(x=serie_mensual["fecha"], y=serie_mensual["tasa_evento"], mode="lines",
                             line=dict(color=AZUL, width=2), name="tasa de eventos", showlegend=False,
                             hovertemplate="%{x|%Y-%m}: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=oni["fecha"], y=oni["ONI"], mode="lines", line=dict(color=AZUL, width=2),
                             name="ONI", showlegend=False, hovertemplate="%{x|%Y-%m}: %{y:.2f}<extra></extra>"),
                  row=2, col=1)
    for nombre in ("prueba_equipos", "prueba_oculta"):
        f = datos[nombre]["fecha"]
        fig.add_vrect(x0=f.min(), x1=f.max(), fillcolor=NEUTRO, line_width=0, layer="below", row="all", col=1)
        fig.add_annotation(x=f.min() + (f.max() - f.min()) / 2, y=1, yref="y2 domain", text=nombre,
                           showarrow=False, font=dict(color=MUTED, size=11), yanchor="bottom")
    for umbral in (config.UMBRAL_ENSO, -config.UMBRAL_ENSO):
        fig.add_hline(y=umbral, line=dict(color=EJE, width=1), row=2, col=1)
    return _estilo(fig, "Serie nacional y ENSO", alto=520)


def covariables_por_conjunto(datos):
    """Una caja por archivo para cada covariable (asimétricas en log1p), a su nivel real de variación."""
    columnas = [c for c in config.COV_NUMERICAS if all(c in df for df in datos.values())]
    filas = int(np.ceil(len(columnas) / 2))
    titulos = [f"log1p({c})" if c in config.COV_LOG else c for c in columnas]
    fig = make_subplots(rows=filas, cols=2, subplot_titles=titulos, vertical_spacing=0.08, horizontal_spacing=0.08)
    for i, col in enumerate(columnas):
        claves = (["DIVIPOLA"] if col in config.COV_ESTATICAS else
                  ["DIVIPOLA", "anio"] if col in config.COV_ANUALES else ["fecha"])
        for nombre, df in datos.items():
            v = df.drop_duplicates(claves)[col].dropna()
            v = np.log1p(v.clip(lower=0)) if col in config.COV_LOG else v
            fig.add_trace(go.Box(y=v, name=nombre, marker_color=COLOR_CONJUNTO[nombre], line_width=1.5,
                                 boxpoints="outliers", marker_size=4, legendgroup=nombre, showlegend=(i == 0)),
                          row=i // 2 + 1, col=i % 2 + 1)
    fig.update_xaxes(showticklabels=False)
    return _estilo(fig, "Covariables: entrenamiento vs. pruebas", alto=260 * filas)


def auc_univariado(tabla):
    """Puntos: AUC de cada covariable sola; la línea vertical en 0.5 es 'sin señal'."""
    t = tabla.sort_values("auc")
    fig = go.Figure(go.Scatter(x=t["auc"], y=t["columna"], mode="markers", marker=dict(size=10, color=AZUL),
                               hovertemplate="%{y}: AUC %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color=EJE, width=1))
    fig.update_xaxes(title="AUC de la covariable sola (0.5 = sin señal)", showgrid=True, gridcolor=GRILLA)
    fig.update_yaxes(showgrid=False)
    return _estilo(fig, "Señal univariada de cada covariable", alto=max(300, 34 * len(t) + 120))


def figuras_exploracion(exp, datos):
    tr = datos["entrenamiento"]
    positivos = np.log1p(tr.loc[tr[config.OBJETIVO_PERSONAS] > 0, config.OBJETIVO_PERSONAS])
    orden_fases = pd.CategoricalDtype(["La Nina", "Neutral", "El Nino"], ordered=True)
    fases = exp["tasa_por_fase_enso"].assign(fase_enso=lambda d: d["fase_enso"].astype(orden_fases)).sort_values("fase_enso")
    return {
        "serie_nacional": serie_nacional(exp["serie_mensual"], datos),
        "estacionalidad": barras(exp["tasa_por_mes"], "mes", "tasa_evento", "Tasa de eventos por mes del año",
                                 "mes", "tasa de eventos"),
        "fase_enso": barras(fases, "fase_enso", "tasa_evento", "Tasa de eventos por fase ENSO",
                            "fase", "tasa de eventos"),
        "departamentos": barras(exp["tasa_por_departamento"], "departamento", "tasa_evento",
                                "Tasa de eventos por departamento (código DANE)", "departamento",
                                "tasa de eventos", horizontal=True),
        "tasas_municipales": histograma(exp["tasas_municipales"]["tasa_evento"],
                                        "Distribución de la tasa de eventos por municipio", "tasa de eventos"),
        "magnitud": histograma(positivos, "Magnitud cuando hay afectados", "log1p(personas afectadas)"),
        "auc_univariado": auc_univariado(exp["auc_univariado"]),
        "covariables": covariables_por_conjunto(datos),
    }


def _tabla_html(df, max_filas=40):
    recortada = df.head(max_filas)
    nota = f"<p class='nota'>Primeras {max_filas} de {len(df):,} filas (completa en CSV).</p>" if len(df) > max_filas else ""
    return recortada.to_html(index=False, border=0, classes="tabla", na_rep="",
                             float_format=lambda v: f"{v:,.4g}") + nota


def reporte_html(secciones, ruta, titulo):
    """secciones: lista de dicts {titulo, texto (opcional), figuras: [fig], tablas: [(nombre, df)]}."""
    cuerpo, primera = [], True
    for s in secciones:
        cuerpo.append(f"<section><h2>{html.escape(s['titulo'])}</h2>")
        if s.get("texto"):
            cuerpo.append(f"<p>{s['texto']}</p>")
        for fig in s.get("figuras", []):
            cuerpo.append(fig.to_html(full_html=False, include_plotlyjs="inline" if primera else False,
                                      config={"displaylogo": False}))
            primera = False
        for nombre, df in s.get("tablas", []):
            cuerpo.append(f"<h3>{html.escape(nombre)}</h3>{_tabla_html(df)}")
        cuerpo.append("</section>")
    pagina = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(titulo)}</title>
<style>
:root {{ color-scheme: light; --superficie:{SUPERFICIE}; --plano:#f9f9f7; --tinta:{TINTA}; --tinta-2:{TINTA_2};
        --muted:{MUTED}; --grilla:{GRILLA}; }}
body {{ margin:0; background:var(--plano); color:var(--tinta); font-family:{FUENTE}; line-height:1.5; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:0 0 12px; }} h3 {{ font-size:14px; margin:20px 0 6px; color:var(--tinta-2); }}
section {{ background:var(--superficie); border:1px solid rgba(11,11,11,.10); border-radius:8px; padding:20px; margin:16px 0; overflow-x:auto; }}
.tabla {{ border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }}
.tabla th, .tabla td {{ padding:4px 10px; border-bottom:1px solid var(--grilla); text-align:right; white-space:nowrap; }}
.tabla th {{ color:var(--tinta-2); font-weight:600; }}
.nota, .sub {{ color:var(--muted); font-size:12px; }}
</style></head><body><main><h1>{html.escape(titulo)}</h1>
<p class="sub">Reporte local generado por scripts/explorar_datos.py. No publicar: resume datos del reto.</p>
{''.join(cuerpo)}</main></body></html>"""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(pagina, encoding="utf-8")
    return ruta
