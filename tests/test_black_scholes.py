import sys
sys.path.append('src')

import numpy as np
from black_scholes import black_scholes_call, black_scholes_put


def test_put_call_parity():
    """C - P = S0 - K*exp(-rT) for any parameter set."""
    test_cases = [
        (100, 100, 0.05, 0.2, 1.0),
        (100, 110, 0.03, 0.3, 0.5),
        (50, 40, 0.08, 0.15, 2.0),
        (200, 250, 0.02, 0.4, 0.25),
        (100, 100, 0.10, 0.5, 3.0),
    ]
    for S0, K, r, sigma, T in test_cases:
        call = black_scholes_call(S0, K, r, sigma, T)
        put = black_scholes_put(S0, K, r, sigma, T)
        lhs = call - put
        rhs = S0 - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10, f"Parity failed for S0={S0}, K={K}"


def test_call_bounds():
    """Call price must be between max(S0 - K*exp(-rT), 0) and S0."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    call = black_scholes_call(S0, K, r, sigma, T)
    lower = max(S0 - K * np.exp(-r * T), 0)
    assert lower <= call <= S0


def test_put_bounds():
    """Put price must be between max(K*exp(-rT) - S0, 0) and K*exp(-rT)."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    put = black_scholes_put(S0, K, r, sigma, T)
    lower = max(K * np.exp(-r * T) - S0, 0)
    assert lower <= put <= K * np.exp(-r * T)


def test_call_increases_with_spot():
    """Call price is monotonically increasing in S0."""
    prices = [black_scholes_call(S0, 100, 0.05, 0.2, 1.0) for S0 in [80, 90, 100, 110, 120]]
    for i in range(len(prices) - 1):
        assert prices[i] < prices[i + 1]


def test_call_increases_with_vol():
    """Call price is monotonically increasing in sigma for ATM."""
    prices = [black_scholes_call(100, 100, 0.05, sigma, 1.0) for sigma in [0.1, 0.2, 0.3, 0.4, 0.5]]
    for i in range(len(prices) - 1):
        assert prices[i] < prices[i + 1]