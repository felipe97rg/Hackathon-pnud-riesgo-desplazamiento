"""Pipeline completo: validación temporal → modelo final → archivo de entrega validado.

Ejemplos:
    python scripts/entrenar.py --solucion A --datos sinteticos
    python scripts/entrenar.py --solucion B --datos reto
    python scripts/entrenar.py --solucion A --datos reto --solo-validacion

Criterio para elegir modelos: el puntaje de validación (fórmula oficial de 75 puntos sobre los cortes temporales,
promedio de F1–F3). Los diagnósticos (pr_auc, brier, cobertura…) ayudan a entender, no deciden.
Cada corrida agrega una fila por corte a outputs/reportes/experimentos.csv (bitácora de experimentos).
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hackaton import config, features  # noqa: E402
from hackaton import datos as D  # noqa: E402
from hackaton.entrega import armar_entrega, guardar_entrega  # noqa: E402
from hackaton.evaluacion import diagnosticos, puntaje  # noqa: E402
from hackaton.modelos import cargar_modelo  # noqa: E402
from hackaton.tratamiento import preparar_datos  # noqa: E402
from hackaton.validacion import folds_temporales  # noqa: E402

METRICAS = ["auc", "rmse_log", "winkler", "puntaje_validacion"]          # se imprimen
DIAGNOSTICOS = ["pr_auc", "brier", "cobertura", "ancho_log"]                 # se imprimen en una segunda línea


def ajustar_y_predecir(modelo_mod, historia, a_predecir, panel, bloques):
    X_hist, X_pred = features.construir_matrices(historia, a_predecir, panel, bloques)
    modelo = modelo_mod.entrenar(X_hist, historia[config.OBJETIVO_EVENTO],
                                 historia[config.OBJETIVO_PERSONAS], config.SEMILLA)
    return modelo_mod.predecir(modelo, X_pred)


def evaluar(objetivo, pred):
    """Métricas oficiales + diagnósticos en un solo dict."""
    argumentos = (objetivo[config.OBJETIVO_EVENTO], objetivo[config.OBJETIVO_PERSONAS],
                  *[pred[c] for c in config.PREDICCIONES])
    return {**puntaje(*argumentos), **diagnosticos(*argumentos)}


def linea(res, claves):
    return " · ".join(f"{m}={res[m]:.3f}" for m in claves)


def main():
    parser = argparse.ArgumentParser(description="Entrena, valida y genera la entrega.")
    parser.add_argument("--solucion", choices=sorted(config.SOLUCIONES), default="A")
    parser.add_argument("--datos", choices=sorted(config.FUENTES_DATOS), default="sinteticos")
    parser.add_argument("--solo-validacion", action="store_true")
    args = parser.parse_args()

    bloques = list(config.SOLUCIONES[args.solucion]["bloques"])
    etiqueta = args.solucion
    directorio = config.FUENTES_DATOS[args.datos]

    datos, _ = preparar_datos(D.cargar_reto(directorio))   # tipos, duplicados, objetivo, imputación
    tablas = [b for b in bloques if b in config.BLOQUES_EXTERNOS]
    datos = {k: D.unir_externos(v, tablas) for k, v in datos.items()}
    panel = D.panel_covariables(datos)
    modelo_mod = cargar_modelo(config.SOLUCIONES[args.solucion]["modelo"])
    print(f"Solución {etiqueta} · datos={args.datos} · bloques={bloques or 'ninguno'}")

    # 1. Validación temporal
    filas = []
    for nombre, historia, validacion in folds_temporales(datos["entrenamiento"]):
        t0 = time.time()
        res = evaluar(validacion, ajustar_y_predecir(modelo_mod, historia, validacion, panel, bloques))
        res.update(fold=nombre, segundos=round(time.time() - t0, 1))
        filas.append(res)
        print(f"  {nombre}: {linea(res, METRICAS)}")
        print(f"        {linea(res, DIAGNOSTICOS)}")
    reporte = pd.DataFrame(filas)
    media = reporte[METRICAS].mean()
    print(f"  media: {linea(media, METRICAS)}  (desv. del puntaje entre cortes: "
          f"{reporte['puntaje_validacion'].std():.2f})")

    config.DIR_REPORTES.mkdir(parents=True, exist_ok=True)
    bitacora = config.DIR_REPORTES / "experimentos.csv"
    reporte.assign(momento=f"{datetime.now():%Y-%m-%d %H:%M}", solucion=etiqueta, datos=args.datos,
                   bloques="+".join(bloques)).to_csv(bitacora, mode="a", header=not bitacora.exists(), index=False)

    if args.solo_validacion:
        return

    # 2. Modelo final con todo el entrenamiento → entrega
    a_predecir = D.filas_a_predecir(datos)
    pred = ajustar_y_predecir(modelo_mod, datos["entrenamiento"], a_predecir, panel, bloques)
    entrega = armar_entrega(a_predecir, pred)
    guardar_entrega(entrega, [datos["prueba_equipos"], datos["prueba_oculta"]], f"{etiqueta}_{args.datos}")

    # 3. Solo con sintéticos: simular la tabla en vivo y la nota oficial
    verdad_ruta = directorio / "verdad_pruebas.csv"
    if args.datos != "reto" and verdad_ruta.exists():
        verdad = pd.read_csv(verdad_ruta, dtype={"DIVIPOLA": str})
        unido = entrega.merge(verdad, on=["DIVIPOLA", "fecha"], validate="one_to_one")
        for conjunto, parte in unido.groupby("conjunto"):
            res = evaluar(parte, parte)
            print(f"  [simulado] {conjunto}: {linea(res, METRICAS)}")


if __name__ == "__main__":
    main()
