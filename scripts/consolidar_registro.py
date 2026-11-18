"""Integra al registro central los registros parciales que dejan los notebooks y muestra el ranking.

Uso:
    python scripts/consolidar_registro.py                                   # solo muestra el ranking
    python scripts/consolidar_registro.py ~/Downloads/registro_experimentos_*.csv
    python scripts/consolidar_registro.py --entrega 2 --variante E10_espacial_simple \\
        --archivo entrega_AICOLOMBIA_E10_espacial_simple.csv --auc 0.77 --rmse 1.16 --winkler 2.68
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hackaton import registro  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Consolida el registro de experimentos y entregas.")
    parser.add_argument("archivos", nargs="*", type=Path, help="registros parciales descargados de Colab")
    parser.add_argument("--entrega", type=int, help="número de la entrega subida al formulario")
    parser.add_argument("--variante")
    parser.add_argument("--archivo")
    parser.add_argument("--auc", type=float)
    parser.add_argument("--rmse", type=float)
    parser.add_argument("--winkler", type=float)
    parser.add_argument("--notas", default="")
    args = parser.parse_args()

    for ruta in args.archivos:
        agregadas, total = registro.agregar_experimentos(pd.read_csv(ruta, dtype={"id_corrida": str}))
        print(f"{ruta.name}: {agregadas} filas nuevas (registro con {total})")

    if args.entrega is not None:
        experimentos = registro.ranking()
        fila = experimentos[experimentos["variante"] == args.variante]
        registro.agregar_entrega(fecha_hora=f"{datetime.now():%Y-%m-%d %H:%M}", entrega=args.entrega,
                                 variante=args.variante, archivo=args.archivo,
                                 puntaje_validacion=fila["puntos_prom"].iloc[0] if len(fila) else None,
                                 auc_tabla_vivo=args.auc, rmse_log_tabla_vivo=args.rmse,
                                 winkler_tabla_vivo=args.winkler, notas=args.notas)
        print(f"Entrega {args.entrega} registrada en {registro.REGISTRO_ENTREGAS.name}")

    pd.set_option("display.width", 220)
    print("\nRanking (datos reales; la corrida más reciente de cada variante):")
    print(registro.ranking(eventos=4977)[["ronda", "variante", "puntos_prom", "puntos_desv", "puntos_F1",
                                          "puntos_F2", "puntos_F3", "auc_prom", "fuente"]].round(3).to_string())
    if registro.REGISTRO_ENTREGAS.exists():
        print("\nEntregas:")
        print(pd.read_csv(registro.REGISTRO_ENTREGAS).to_string())


if __name__ == "__main__":
    main()
