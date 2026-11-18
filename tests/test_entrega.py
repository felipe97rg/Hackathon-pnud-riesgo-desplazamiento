import pandas as pd
import pytest

from hackaton.entrega import armar_entrega
from hackaton.evaluacion import validar_entrega


@pytest.fixture
def pruebas():
    pe = pd.DataFrame({"DIVIPOLA": ["05001", "05003"] * 2, "fecha": ["2022-10-01"] * 2 + ["2022-11-01"] * 2})
    po = pd.DataFrame({"DIVIPOLA": ["05001", "05003"], "fecha": ["2024-05-01"] * 2})
    return pe, po


def predicciones(n, q10=0.0, q90=5.0):
    return pd.DataFrame({"prob_evento": [0.2] * n, "personas_desplazadas_estimadas": [1.0] * n,
                         "personas_desplazadas_q10": [q10] * n, "personas_desplazadas_q90": [q90] * n})


def test_entrega_valida(pruebas):
    filas = pd.concat(pruebas, ignore_index=True)
    assert validar_entrega(armar_entrega(filas, predicciones(len(filas))), list(pruebas))


def test_armar_entrega_ordena_cuantiles_cruzados(pruebas):
    filas = pd.concat(pruebas, ignore_index=True)
    entrega = armar_entrega(filas, predicciones(len(filas), q10=9.0, q90=2.0))
    assert (entrega["personas_desplazadas_q10"] <= entrega["personas_desplazadas_q90"]).all()


def test_detecta_divipola_numerico(pruebas):
    filas = pd.concat(pruebas, ignore_index=True)
    entrega = armar_entrega(filas, predicciones(len(filas)))
    entrega["DIVIPOLA"] = entrega["DIVIPOLA"].astype(int)
    with pytest.raises(AssertionError, match="DIVIPOLA"):
        validar_entrega(entrega, list(pruebas))


def test_detecta_filas_faltantes(pruebas):
    filas = pd.concat(pruebas, ignore_index=True)
    entrega = armar_entrega(filas, predicciones(len(filas))).iloc[:-1]
    with pytest.raises(AssertionError, match="Filas"):
        validar_entrega(entrega, list(pruebas))
