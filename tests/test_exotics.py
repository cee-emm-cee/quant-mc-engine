import sys
sys.path.append('src')

import numpy as np
from black_scholes import black_scholes_call
from exotics import asian_call_fixed_strike, up_and_out_call


def test_asian_less_than_european():
    """Asian call price must be less than European call price."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    bs = black_scholes_call(S0, K, r, sigma, T)
    asian = asian_call_fixed_strike(S0, K, r, sigma, T)
    assert asian['price'] < bs


def test_asian_one_step_matches_european():
    """Asian call with 1 monitoring date should match European call."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    bs = black_scholes_call(S0, K, r, sigma, T)
    asian = asian_call_fixed_strike(S0, K, r, sigma, T, n_steps=1)
    assert abs(asian['price'] - bs) < 3 * asian['stderr']


def test_asian_price_decreases_with_steps():
    """More monitoring dates should reduce Asian call price."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    prices = []
    for n in [1, 10, 50, 252]:
        result = asian_call_fixed_strike(S0, K, r, sigma, T, n_steps=n)
        prices.append(result['price'])
    for i in range(len(prices) - 1):
        assert prices[i] > prices[i + 1]


def test_barrier_converges_to_european():
    """Up-and-out call with very high barrier should match European call."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    bs = black_scholes_call(S0, K, r, sigma, T)
    uo = up_and_out_call(S0, K, 1000, r, sigma, T)
    assert abs(uo['price'] - bs) < 3 * uo['stderr']


def test_barrier_near_spot_is_worthless():
    """Up-and-out call with barrier just above S0 should be near zero."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    uo = up_and_out_call(S0, K, 101, r, sigma, T)
    assert uo['price'] < 0.5
    assert uo['knockout_pct'] > 0.9


def test_barrier_price_increases_with_barrier():
    """Higher barrier means higher up-and-out price."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    prices = []
    for B in [110, 120, 130, 150, 200]:
        result = up_and_out_call(S0, K, B, r, sigma, T)
        prices.append(result['price'])
    for i in range(len(prices) - 1):
        assert prices[i] < prices[i + 1]