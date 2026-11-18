"""Comprueba que los datos sintéticos respetan el esquema oficial (requiere generarlos antes)."""
import pytest

from hackaton import config
from hackaton.datos import cargar_reto, filas_a_predecir, panel_covariables

pytestmark = pytest.mark.skipif(not (config.DIR_SINTETICOS / "entrenamiento.csv").exists(),
                                reason="Correr antes: python scripts/generar_datos_sinteticos.py")


def test_esquema_y_filas():
    datos = cargar_reto(config.DIR_SINTETICOS)
    for nombre, df in datos.items():
        assert len(df) == config.FILAS_ESPERADAS[nombre]
        assert df["DIVIPOLA"].str.len().eq(5).all()
        assert df["DIVIPOLA"].nunique() == config.N_MUNICIPIOS
    assert len(filas_a_predecir(datos)) == 42_978
    assert set(config.HISTORIAL) <= set(datos["entrenamiento"].columns)


def test_panel_continuo():
    panel = panel_covariables(cargar_reto(config.DIR_SINTETICOS))
    assert panel.groupby("DIVIPOLA")["fecha"].nunique().eq(96).all()
