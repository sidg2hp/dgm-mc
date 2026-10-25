import numpy as np
import pytest

from dgm_mc.estimators import fit_controls, pair_units, path_samples, rqmc, summarize, terminal_samples
from dgm_mc.market import Market, equicorrelated, geometric_log_gradient, geometric_price, payoff


def test_black_scholes_known_value():
    assert geometric_price(equicorrelated(1))[0] == pytest.approx(0.10450583572185557, abs=1e-14)


@pytest.mark.parametrize("d", [1, 2, 6, 10])
def test_geometric_mc_matches_closed_form(d):
    m = equicorrelated(d)
    y, _ = terminal_samples(m, 50000, 392, kind="geometric")
    stat = summarize(y)
    assert abs(stat["price"] - geometric_price(m)[0]) < 5 * stat["se"]


def test_deterministic_and_terminal():
    m = equicorrelated(2, sigma=0, spot=1.2)
    expected = 1.2 - np.exp(-0.05)
    assert geometric_price(m)[0] == pytest.approx(expected)
    m = equicorrelated(2, maturity=0, spot=1.2)
    assert geometric_price(m)[0] == pytest.approx(0.2)


@pytest.mark.parametrize("kwargs", [{"spot": 0}, {"sigma": -0.1}, {"maturity": -1}, {"strike": 0}, {"rate": float("nan")}, {"rho": -2}])
def test_invalid_market(kwargs):
    with pytest.raises(ValueError):
        equicorrelated(2, **kwargs)


def test_singular_correlation_is_supported():
    m = equicorrelated(3, rho=1.0)
    assert np.allclose(m.diffusion @ m.diffusion.T, m.covariance)


def test_antithetic_units_are_pairs():
    values = np.array([1., 3., 9., 5., 7., 11.])
    pairs = pair_units(values)
    assert np.array_equal(pairs, [3., 5., 10.])
    assert summarize(pairs)["se"] == pytest.approx(np.std(pairs, ddof=1) / np.sqrt(3))
    with pytest.raises(ValueError):
        terminal_samples(equicorrelated(1), 5, 1, antithetic=True)


@pytest.mark.parametrize("values", [[1], [1, float("nan")], [1, float("inf")]])
def test_reject_bad_statistics(values):
    with pytest.raises(ValueError):
        summarize(values)


def test_fit_degenerate_and_correlated_controls():
    c = np.arange(10.)
    controls = np.column_stack([c, c, np.zeros(10)])
    beta = fit_controls(3 * c + 7, controls)
    assert np.allclose(controls @ beta, 3 * c)
    assert beta[2] == 0


def test_zero_control_and_exact_paths():
    m = equicorrelated(3)
    y, controls = path_samples(m, 100, 4, 33, lambda t, x: np.zeros_like(x))
    assert np.array_equal(controls[:, 1], np.zeros(100))
    # Independent reimplementation of the same exact log dynamics.
    rng = np.random.default_rng(33)
    x = np.tile(m.log_spot, (100, 1))
    for _ in range(4):
        x += m.drift / 4 + rng.standard_normal(x.shape) / 2 @ m.diffusion.T
    assert np.allclose(y, np.exp(-m.rate) * payoff(x, m))


def test_left_endpoint_is_used():
    m = equicorrelated(1)
    seen = []
    def grad(t, x):
        seen.append((t, x.copy()))
        return np.ones_like(x)
    path_samples(m, 10, 4, 7, grad)
    assert [t for t, _ in seen] == [0, .25, .5, .75]
    assert np.array_equal(seen[0][1], np.zeros((10, 1)))


def test_oracle_delta_reduces_variance_and_is_centered():
    m = equicorrelated(1)
    y, c = path_samples(m, 12000, 32, 131, lambda t, x: geometric_log_gradient(m, t, x))
    stats = summarize(y - c[:, 1])
    assert stats["variance"] < np.var(y) * .05
    assert abs(stats["price"] - geometric_price(m)[0]) < 5 * stats["se"]


def test_inaccurate_integrand_does_not_shift_expectation():
    m = equicorrelated(2)
    _, c = path_samples(m, 40000, 4, 19, lambda t, x: 2 + np.tanh(x))
    stats = summarize(c[:, 1])
    assert abs(stats["price"]) < 5 * stats["se"]


def test_terminal_centering_counterexample():
    m = equicorrelated(1)
    truth = geometric_price(m)[0]
    y, _ = terminal_samples(m, 10000, 28)
    # Legacy logic even with EXACT terminal predictor and known coefficient:
    terminal_control = y * np.exp(m.rate)
    invalid = y - np.exp(-m.rate) * (terminal_control - truth)
    assert np.std(invalid) < 1e-15
    assert np.mean(invalid) == pytest.approx(np.exp(-m.rate) * truth)
    assert abs(np.mean(invalid) - truth) > .005


def test_rqmc_units_and_budget():
    result = rqmc(equicorrelated(1), 128, 4, 3)
    assert result["paths"] == 512 and result["units"] == 4
    assert result["se"] == pytest.approx(np.std(result["scramble_means"], ddof=1) / 2)
    with pytest.raises(ValueError):
        rqmc(equicorrelated(1), 127, 4, 3)
