import sys
sys.path.append('src')

import numpy as np
from black_scholes import black_scholes_call
from heston import heston_european_call


def test_heston_collapses_to_bs():
    """With xi=0, Heston should match BS."""
    S0, K, r, T = 100, 100, 0.05, 1.0
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    xi = 0.0
    rho = -0.7

    bs = black_scholes_call(S0, K, r, 0.2, T)
    heston = heston_european_call(S0, K, r, v0, kappa, theta, xi, rho, T)
    assert abs(heston['price'] - bs) < 3 * heston['stderr']


def test_heston_price_positive():
    """Heston call price must be positive."""
    result = heston_european_call(100, 100, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0)
    assert result['price'] > 0


def test_heston_price_less_than_spot():
    """Call price must be less than spot."""
    result = heston_european_call(100, 100, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0)
    assert result['price'] < 100


def test_heston_higher_vol_of_vol():
    """Higher xi should change the price (not necessarily higher or lower)."""
    base = heston_european_call(100, 100, 0.05, 0.04, 2.0, 0.04, 0.1, -0.7, 1.0)
    high_xi = heston_european_call(100, 100, 0.05, 0.04, 2.0, 0.04, 0.5, -0.7, 1.0)
    assert abs(base['price'] - high_xi['price']) > 0.01


def test_heston_correlation_effect():
    """Negative rho should produce different price than positive rho."""
    neg_rho = heston_european_call(100, 100, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0)
    pos_rho = heston_european_call(100, 100, 0.05, 0.04, 2.0, 0.04, 0.3, 0.7, 1.0)
    assert abs(neg_rho['price'] - pos_rho['price']) > 0.01


def test_heston_stderr_reasonable():
    """Standard error should be small relative to price."""
    result = heston_european_call(100, 100, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0)
    assert result['stderr'] / result['price'] < 0.01