"""Genera el geovisor (entregable 3): un HTML autocontenido con el mapa de las predicciones.

Uso:
    python scripts/generar_geovisor.py --modelo "Solución A"              # usa la entrega más reciente
    python scripts/generar_geovisor.py --entrega outputs/entregas/entrega_X.csv --historico reto
    python scripts/generar_geovisor.py --demo                             # prueba con datos sintéticos

--demo arma predicciones de DEMOSTRACIÓN a partir de una climatología simple sobre data/sinteticos
(no es el modelo del equipo); el geovisor lo indica con un aviso visible.
--historico agrega la capa "tasa histórica de eventos": es información derivada de los datos del reto,
así que ese geovisor es de uso interno y no debe publicarse.

Salida: outputs/geovisor/geovisor.html (no versionado: contiene predicciones).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hackaton import config  # noqa: E402
from hackaton.datos import cargar_reto, filas_a_predecir  # noqa: E402
from hackaton.entrega import armar_entrega  # noqa: E402
from hackaton.geovisor import construir_geovisor  # noqa: E402
from hackaton.tratamiento import preparar_datos  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def entrega_mas_reciente():
    archivos = sorted(config.DIR_ENTREGAS.glob("entrega_*.csv"), key=lambda p: p.stat().st_mtime)
    if not archivos:
        raise FileNotFoundError("No hay entregas en outputs/entregas/. Corre scripts/entrenar.py o usa --demo.")
    return archivos[-1]


def predicciones_demo():
    """Climatología municipio × mes sobre los sintéticos. Solo para probar el geovisor."""
    datos, _ = preparar_datos(cargar_reto(config.DIR_SINTETICOS), verbose=False)
    tr, filas = datos["entrenamiento"], filas_a_predecir(datos)
    base = tr[config.OBJETIVO_EVENTO].mean()
    conteo = tr.groupby(["DIVIPOLA", "mes"])[config.OBJETIVO_EVENTO].agg(["sum", "count"])
    tasa = ((conteo["sum"] + 6 * base) / (conteo["count"] + 6)).rename("p")
    positivos = np.log1p(tr.loc[tr[config.OBJETIVO_PERSONAS] > 0].groupby("DIVIPOLA")[config.OBJETIVO_PERSONAS].median())
    p = filas.join(tasa, on=["DIVIPOLA", "mes"])["p"].fillna(base).to_numpy()
    m = filas["DIVIPOLA"].map(positivos).fillna(positivos.median()).to_numpy()
    onda = 1 + 0.25 * np.clip(-filas["ONI"].to_numpy(), -1, 2)      # La Niña sube el riesgo en la demo
    p = np.clip(p * onda, 0, 1)
    pred = pd.DataFrame({
        "prob_evento": p,
        "personas_desplazadas_estimadas": np.expm1(p * m),
        "personas_desplazadas_q10": 0.0,
        "personas_desplazadas_q90": np.where(p >= 0.1, np.expm1(m + 1.0), 0.0),
    })
    return armar_entrega(filas, pred)


def main():
    parser = argparse.ArgumentParser(description="Genera el geovisor HTML.")
    parser.add_argument("--entrega", type=Path, help="CSV en formato de entrega (por defecto, el más reciente)")
    parser.add_argument("--demo", action="store_true", help="predicciones de demostración con datos sintéticos")
    parser.add_argument("--historico", choices=sorted(config.FUENTES_DATOS),
                        help="agrega la capa de tasa histórica (uso interno)")
    parser.add_argument("--modelo", default="", help='texto que identifica el modelo, p. ej. "Solución A"')
    parser.add_argument("--salida", type=Path, default=config.DIR_SALIDAS / "geovisor" / "geovisor.html")
    args = parser.parse_args()

    if args.demo:
        entrega, origen = predicciones_demo(), "demo (sintéticos)"
    else:
        ruta = args.entrega or entrega_mas_reciente()
        entrega, origen = pd.read_csv(ruta, dtype={"DIVIPOLA": str}), ruta.name
    historico = None
    if args.historico:
        datos, _ = preparar_datos(cargar_reto(config.FUENTES_DATOS[args.historico]), verbose=False)
        historico = datos["entrenamiento"]

    ruta, resumen = construir_geovisor(entrega, args.salida, historico=historico,
                                       modelo=args.modelo or ("Demostración" if args.demo else ""), demo=args.demo)
    print(f"Geovisor: {ruta} ({resumen['mb']} MB)")
    print(f"  Predicciones: {origen} · {resumen['municipios']:,} municipios × {resumen['meses']} meses")
    print(f"  Polígonos sin predicción (gris): {resumen['poligonos_sin_prediccion']}")
    if resumen["sin_poligono"]:
        print(f"  [!] {len(resumen['sin_poligono'])} municipios con predicción sin polígono: {resumen['sin_poligono'][:10]}")


if __name__ == "__main__":
    main()
