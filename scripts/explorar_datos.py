"""Exploración y diagnóstico de calidad de los datos, con tratamiento y reporte HTML.

Uso:
    python scripts/explorar_datos.py --datos sinteticos_sucios   # práctica con problemas inyectados
    python scripts/explorar_datos.py --datos reto                # el día D, apenas lleguen los CSV

Pasos: (1) diagnóstico de calidad de los datos crudos, (2) tratamiento (tratamiento.preparar_datos),
(3) diagnóstico de nuevo para confirmar que quedaron limpios, (4) exploración del objetivo, (5) reporte.
Salida en outputs/eda/<datos>/ (no versionada): reporte.html y las tablas en CSV.
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hackaton import calidad, config, exploracion, graficos  # noqa: E402
from hackaton.datos import cargar_reto  # noqa: E402
from hackaton.tratamiento import preparar_datos  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows no siempre usa UTF-8


def guardar_tablas(tablas, carpeta, prefijo):
    for nombre, df in tablas.items():
        df.to_csv(carpeta / f"{prefijo}_{nombre}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico, tratamiento y exploración de los datos.")
    parser.add_argument("--datos", choices=sorted(config.FUENTES_DATOS), default="sinteticos")
    args = parser.parse_args()
    directorio = config.FUENTES_DATOS[args.datos]

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        crudos = cargar_reto(directorio, estricto=False)
    for aviso in avisos:
        print(f"  aviso: {aviso.message}")

    print("\n1) DATOS CRUDOS")
    diag_crudo = calidad.diagnostico_completo(crudos)
    calidad.imprimir_resumen(diag_crudo)

    print("\n2) TRATAMIENTO")
    datos, registro = preparar_datos(crudos)

    print("\n3) DATOS TRATADOS")
    diag_tratado = calidad.diagnostico_completo(datos)
    calidad.imprimir_resumen(diag_tratado)

    print("\n4) EXPLORACIÓN")
    exp = exploracion.exploracion_completa(datos["entrenamiento"])
    exploracion.imprimir_resumen(exp)

    salida = config.DIR_EDA / args.datos
    salida.mkdir(parents=True, exist_ok=True)
    tabla_duplicados = pd.DataFrame(registro["duplicados"]).T.rename_axis("archivo").reset_index()
    tabla_objetivo = pd.DataFrame(list(registro["objetivo"].items()), columns=["correccion", "filas"])
    guardar_tablas(diag_crudo, salida, "calidad_crudo")
    guardar_tablas(diag_tratado, salida, "calidad_tratado")
    guardar_tablas({"duplicados": tabla_duplicados, "objetivo": tabla_objetivo,
                    "imputacion": registro["imputacion"]}, salida, "tratamiento")
    guardar_tablas(exp, salida, "eda")

    figs = graficos.figuras_exploracion(exp, datos)
    secciones = [
        {"titulo": "1. Calidad de los datos crudos",
         "texto": "Errores (reglas de dominio) son datos imposibles; outliers son valores extremos que pueden ser reales.",
         "tablas": [("Esquema", diag_crudo["esquema"]), ("Nulos", diag_crudo["nulos"]),
                    ("Duplicados", diag_crudo["duplicados"]), ("Completitud del panel", diag_crudo["panel"]),
                    ("Reglas de dominio", diag_crudo["reglas"]), ("Consistencia temporal", diag_crudo["consistencia"]),
                    ("Outliers", diag_crudo["outliers"])]},
        {"titulo": "2. Tratamiento aplicado",
         "tablas": [("Duplicados", tabla_duplicados), ("Correcciones del objetivo", tabla_objetivo),
                    ("Imputación de covariables", registro["imputacion"]),
                    ("Nulos después del tratamiento", diag_tratado["nulos"]),
                    ("Reglas después del tratamiento", diag_tratado["reglas"].query("violaciones > 0"))]},
        {"titulo": "3. Variable objetivo",
         "figuras": [figs["serie_nacional"], figs["estacionalidad"], figs["fase_enso"], figs["magnitud"]],
         "tablas": [("Resumen", exp["resumen_objetivo"]), ("Por año", exp["tasa_por_anio"]),
                    ("Persistencia", exp["persistencia"]), ("Tipos de evento", exp["mezcla_tipos"])]},
        {"titulo": "4. Geografía del riesgo",
         "figuras": [figs["departamentos"], figs["tasas_municipales"]],
         "tablas": [("Por departamento", exp["tasa_por_departamento"])]},
        {"titulo": "5. Covariables",
         "figuras": [figs["covariables"], figs["auc_univariado"]],
         "tablas": [("Señal univariada", exp["auc_univariado"]),
                    ("Cambio de distribución entrenamiento vs. pruebas", diag_tratado["cambio_distribucion"])]},
        {"titulo": "6. Líneas base de climatología",
         "texto": "Tasas calculadas hasta 2021-09 y evaluadas en 2021-10…2022-09. Es el piso que el modelo debe superar.",
         "tablas": [("AUC de líneas base", exp["linea_base_climatologia"])]},
    ]
    ruta = graficos.reporte_html(secciones, salida / "reporte.html", f"Exploración de datos ({args.datos})")
    print(f"\nReporte: {ruta}\nTablas CSV: {salida}")


if __name__ == "__main__":
    main()
