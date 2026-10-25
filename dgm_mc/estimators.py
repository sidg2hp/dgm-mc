"""Sampling units and zero-mean controls are explicit in every estimator."""
from time import perf_counter

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc, t as student_t

from .market import geometric_price, payoff


def summarize(units):
    values = np.asarray(units, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite independent sampling units required")
    mean = float(values.mean())
    variance = float(values.var(ddof=1))
    se = np.sqrt(variance / len(values))
    half = student_t.ppf(0.975, len(values) - 1) * se
    return dict(price=mean, se=float(se), variance=variance, units=len(values),
                ci_low=float(mean - half), ci_high=float(mean + half))


def pair_units(values):
    values = np.asarray(values)
    if len(values) < 4 or len(values) % 2:
        raise ValueError("Antithetic sampling requires an even path count >= 4")
    half = len(values) // 2
    return (values[:half] + values[half:]) / 2


def terminal_samples(market, n_paths, seed, kind="arithmetic", antithetic=False):
    if n_paths < 2 or (antithetic and (n_paths < 4 or n_paths % 2)):
        raise ValueError("Invalid actual path budget")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths // 2 if antithetic else n_paths, market.d))
    if antithetic:
        z = np.concatenate([z, -z])
    x = market.log_spot + market.drift * market.maturity + np.sqrt(market.maturity) * (z @ market.diffusion.T)
    discount = np.exp(-market.rate * market.maturity)
    y = discount * payoff(x, market, kind)
    g = discount * payoff(x, market, "geometric") - geometric_price(market)[0]
    return (pair_units(y), pair_units(g)) if antithetic else (y, g)


def path_samples(market, n_paths, steps, seed, gradient, kind="arithmetic"):
    """Exact GBM endpoints; gradient(t,x) evaluated BEFORE each fresh increment.

    Training, pilot fitting and evaluation must use independent RNG streams.
    Arbitrary finite frozen gradients are admissible; accuracy is not assumed.
    """
    if n_paths < 2 or steps < 1:
        raise ValueError("Invalid path count/time grid")
    rng = np.random.default_rng(seed)
    x = np.tile(market.log_spot, (n_paths, 1))
    integral = np.zeros(n_paths)
    dt = market.maturity / steps
    diffusion = market.diffusion
    for j in range(steps):
        time = j * dt
        grad = np.asarray(gradient(time, x))
        if grad.shape != x.shape or not np.isfinite(grad).all():
            raise ValueError("Gradient must be finite with shape (paths, d)")
        dw = rng.standard_normal(x.shape) * np.sqrt(dt)
        noise = dw @ diffusion.T
        integral += np.exp(-market.rate * time) * (grad * noise).sum(axis=1)
        x = x + market.drift * dt + noise
    discount = np.exp(-market.rate * market.maturity)
    y = discount * payoff(x, market, kind)
    geo = discount * payoff(x, market, "geometric") - geometric_price(market)[0]
    return y, np.column_stack([geo, integral])


def fit_controls(y, controls):
    """Pilot OLS with SVD, standardized columns and no arbitrary ridge bias.

    Centering here estimates covariance, not the known control expectation.
    Evaluation MUST subtract the original zero-mean controls, not pilot means.
    Degenerate columns receive zero coefficient.
    """
    y, c = np.asarray(y), np.asarray(controls)
    if c.ndim != 2 or y.shape != (len(c),) or len(y) < 3:
        raise ValueError("Invalid pilot shapes")
    if not np.isfinite(y).all() or not np.isfinite(c).all():
        raise ValueError("Nonfinite pilot")
    centered = c - c.mean(axis=0)
    scale = centered.std(axis=0, ddof=1)
    active = scale > 1e-14
    beta = np.zeros(c.shape[1])
    if active.any():
        standardized = centered[:, active] / scale[active]
        weights = np.linalg.lstsq(standardized, y - y.mean(), rcond=1e-10)[0]
        beta[active] = weights / scale[active]
    return beta


def rqmc(market, paths_per_scramble, scrambles, seed, kind="arithmetic"):
    """SE across independent scrambles, NOT across dependent Sobol points."""
    if paths_per_scramble < 2 or paths_per_scramble & (paths_per_scramble - 1):
        raise ValueError("Sobol point count must be a power of two >= 2")
    if scrambles < 2:
        raise ValueError("At least two independent scrambles required")
    begin = perf_counter()
    means = []
    for child in np.random.SeedSequence(seed).spawn(scrambles):
        engine = qmc.Sobol(market.d, scramble=True, rng=np.random.default_rng(child))
        u = engine.random_base2(int(np.log2(paths_per_scramble)))
        z = ndtri(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps))
        x = market.log_spot + market.drift * market.maturity + np.sqrt(market.maturity) * (z @ market.diffusion.T)
        means.append(np.exp(-market.rate * market.maturity) * payoff(x, market, kind).mean())
    result = summarize(means)
    result.update(paths=paths_per_scramble * scrambles, seconds=perf_counter() - begin,
                  sampling_unit="independent_scramble", scramble_means=means)
    return result
