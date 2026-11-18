import numpy as np

from hackaton.evaluacion import diagnosticos, puntaje, winkler_log


def datos_simulados(n=5000, semilla=0):
    rng = np.random.default_rng(semilla)
    p = rng.beta(1, 6, n)
    evento = (rng.random(n) < p).astype(int)
    y = np.where(evento, np.round(np.expm1(rng.normal(3.5, 1.5, n)).clip(0)), 0)
    return p, evento, y


def test_prediccion_nula_vale_cero():
    _, evento, y = datos_simulados()
    cero = np.zeros_like(y)
    res = puntaje(evento, y, np.full(len(y), 0.5), cero, cero, cero)
    assert res["puntaje_validacion"] == 0


def test_modelo_informativo_supera_nulo():
    p, evento, y = datos_simulados()
    cero = np.zeros_like(y)
    res = puntaje(evento, y, p, np.expm1(p * 3.5), cero, np.where(p > 0.1, 60.0, 0.0))
    assert res["auc"] > 0.6 and res["puntaje_validacion"] > 0


def test_winkler_es_diez_veces_pinball():
    _, _, y = datos_simulados()
    z, lo, hi = np.log1p(y), 0.0, np.log1p(20.0)

    def pinball(tau, q):
        return np.mean(np.where(z >= q, tau * (z - q), (tau - 1) * (z - q)))

    assert np.isclose(winkler_log(y, 0.0, 20.0), 10 * (pinball(0.1, lo) + pinball(0.9, hi)))


def test_diagnosticos_cobertura_con_ceros():
    """Con muchos ceros, el intervalo [0, 0] en filas de bajo riesgo da cobertura mayor al 80 %."""
    p, evento, y = datos_simulados()
    cero = np.zeros_like(y)
    q90 = np.where(p > 0.3, 200.0, 0.0)
    d = diagnosticos(evento, y, p, np.expm1(p * 3.5), cero, q90)
    assert d["cobertura"] > 0.8
    assert 0 <= d["brier"] <= 1 and 0 < d["pr_auc"] <= 1
    assert d["rmse_log_sin_afectados"] < d["rmse_log_con_afectados"]
