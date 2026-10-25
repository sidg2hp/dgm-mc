"""Exact lognormal dynamics and independently checkable basket formulas."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class Market:
    spot: tuple[float, ...]
    sigma: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    rate: float = 0.05
    maturity: float = 1.0
    strike: float = 1.0

    def __post_init__(self):
        s, v, c = map(np.asarray, (self.spot, self.sigma, self.correlation))
        if s.ndim != 1 or len(s) == 0 or v.shape != s.shape or c.shape != (len(s), len(s)):
            raise ValueError("Inconsistent market dimensions")
        if not all(np.isfinite(a).all() for a in (s, v, c)):
            raise ValueError("Nonfinite market inputs")
        if np.any(s <= 0) or np.any(v < 0):
            raise ValueError("Spots must be positive and volatilities nonnegative")
        if not np.isfinite([self.rate, self.maturity, self.strike]).all():
            raise ValueError("Nonfinite market scalars")
        if self.maturity < 0 or self.strike <= 0:
            raise ValueError("Invalid maturity or strike")
        if not np.allclose(c, c.T, atol=1e-12, rtol=0) or not np.allclose(np.diag(c), 1):
            raise ValueError("Correlation must be symmetric with unit diagonal")
        if np.linalg.eigvalsh(c).min() < -1e-12:
            raise ValueError("Correlation must be positive semidefinite")

    @property
    def d(self):
        return len(self.spot)

    @property
    def covariance(self):
        return np.outer(self.sigma, self.sigma) * np.asarray(self.correlation)

    @property
    def diffusion(self):
        # Eigenfactor also supports perfectly correlated/deterministic assets.
        values, vectors = np.linalg.eigh(self.covariance)
        return vectors * np.sqrt(np.maximum(values, 0))[None, :]

    @property
    def drift(self):
        return self.rate - 0.5 * np.asarray(self.sigma) ** 2

    @property
    def log_spot(self):
        return np.log(np.asarray(self.spot) / self.strike)

    def to_dict(self):
        return asdict(self)


def equicorrelated(d, spot=1.0, sigma=0.2, rho=0.3, **kwargs):
    if d < 1:
        raise ValueError("d must be positive")
    corr = np.full((d, d), rho)
    np.fill_diagonal(corr, 1.0)
    spots = np.broadcast_to(spot, (d,))
    vols = np.broadcast_to(sigma, (d,))
    return Market(tuple(spots), tuple(vols), tuple(map(tuple, corr)), **kwargs)


def payoff(log_prices, market, kind="arithmetic"):
    x = np.asarray(log_prices)
    if x.ndim != 2 or x.shape[1] != market.d or not np.isfinite(x).all():
        raise ValueError("Expected finite (paths, d) log prices")
    if kind == "arithmetic":
        underlying = np.exp(x).mean(axis=1)
    elif kind == "geometric":
        underlying = np.exp(x.mean(axis=1))
    elif kind == "max":
        underlying = np.exp(x).max(axis=1)
    else:
        raise ValueError(f"Unsupported payoff: {kind}")
    return market.strike * np.maximum(underlying - 1, 0)


def geometric_price(market, log_prices=None, tau=None):
    """Equal-weight geometric basket price; includes zero variance and maturity."""
    x = market.log_spot[None, :] if log_prices is None else np.atleast_2d(log_prices)
    tau = market.maturity if tau is None else tau
    if tau < 0 or x.shape[1] != market.d:
        raise ValueError("Invalid geometric price arguments")
    mean = x.mean(axis=1) + market.drift.mean() * tau
    variance = max(float(market.covariance.mean()) * tau, 0.0)
    discount = np.exp(-market.rate * tau)
    if variance < 1e-30:
        return market.strike * discount * np.maximum(np.exp(mean) - 1, 0)
    root = np.sqrt(variance)
    return market.strike * discount * (
        np.exp(mean + variance / 2) * ndtr((mean + variance) / root) - ndtr(mean / root)
    )


def geometric_log_gradient(market, t, x):
    """Analytical du/dx, used only as an explicitly labelled oracle control."""
    tau = market.maturity - t
    mean = x.mean(axis=1) + market.drift.mean() * tau
    variance = max(float(market.covariance.mean()) * tau, 0.0)
    cdf = (mean > 0).astype(float) if variance < 1e-30 else ndtr((mean + variance) / np.sqrt(variance))
    value = market.strike * np.exp(-market.rate * tau + mean + variance / 2) * cdf / market.d
    return np.repeat(value[:, None], market.d, axis=1)
