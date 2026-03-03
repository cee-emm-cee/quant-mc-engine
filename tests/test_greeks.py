import sys
sys.path.append('src')

import numpy as np
from scipy.stats import norm
from greeks import delta, gamma, vega, theta, rho, _bs_pricer


def bs_analytical_delta(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def bs_analytical_gamma(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S0 * sigma * np.sqrt(T))


def bs_analytical_vega(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * norm.pdf(d1) * np.sqrt(T)


def bs_analytical_rho(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * T * np.exp(-r * T) * norm.cdf(d2)


def test_delta_matches_analytical():
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    numerical = delta(_bs_pricer, S0, K, r, sigma, T)
    analytical = bs_analytical_delta(S0, K, r, sigma, T)
    assert abs(numerical - analytical) < 0.001


def test_gamma_matches_analytical():
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    numerical = gamma(_bs_pricer, S0, K, r, sigma, T)
    analytical = bs_analytical_gamma(S0, K, r, sigma, T)
    assert abs(numerical - analytical) < 0.0005


def test_vega_matches_analytical():
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    numerical = vega(_bs_pricer, S0, K, r, sigma, T)
    analytical = bs_analytical_vega(S0, K, r, sigma, T)
    assert abs(numerical - analytical) < 0.05


def test_rho_matches_analytical():
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    numerical = rho(_bs_pricer, S0, K, r, sigma, T)
    analytical = bs_analytical_rho(S0, K, r, sigma, T)
    assert abs(numerical - analytical) < 0.05


def test_delta_bounded():
    """Call delta must be between 0 and 1."""
    for S0 in [70, 85, 100, 115, 130]:
        d = delta(_bs_pricer, S0, 100, 0.05, 0.2, 1.0)
        assert 0 <= d <= 1


def test_gamma_positive():
    """Call gamma must be positive."""
    for S0 in [70, 85, 100, 115, 130]:
        g = gamma(_bs_pricer, S0, 100, 0.05, 0.2, 1.0)
        assert g > 0


def test_delta_increases_with_spot():
    """Delta must increase as spot increases."""
    deltas = [delta(_bs_pricer, S0, 100, 0.05, 0.2, 1.0) for S0 in [80, 90, 100, 110, 120]]
    for i in range(len(deltas) - 1):
        assert deltas[i] < deltas[i + 1]