import sys
sys.path.append('src')

import numpy as np
from black_scholes import black_scholes_call
from implied_vol import implied_vol


def test_round_trip_atm():
    """Recover known vol for ATM option."""
    S0, K, r, T = 100, 100, 0.05, 1.0
    true_vol = 0.2
    price = black_scholes_call(S0, K, r, true_vol, T)
    recovered = implied_vol(price, S0, K, r, T)
    assert abs(recovered - true_vol) < 1e-8


def test_round_trip_itm():
    """Recover known vol for ITM option."""
    S0, K, r, T = 100, 85, 0.05, 1.0
    true_vol = 0.3
    price = black_scholes_call(S0, K, r, true_vol, T)
    recovered = implied_vol(price, S0, K, r, T)
    assert abs(recovered - true_vol) < 1e-8


def test_round_trip_otm():
    """Recover known vol for OTM option."""
    S0, K, r, T = 100, 115, 0.05, 1.0
    true_vol = 0.25
    price = black_scholes_call(S0, K, r, true_vol, T)
    recovered = implied_vol(price, S0, K, r, T)
    assert abs(recovered - true_vol) < 1e-8


def test_round_trip_high_vol():
    """Recover known vol for high volatility option."""
    S0, K, r, T = 100, 100, 0.05, 1.0
    true_vol = 0.8
    price = black_scholes_call(S0, K, r, true_vol, T)
    recovered = implied_vol(price, S0, K, r, T)
    assert abs(recovered - true_vol) < 1e-8


def test_round_trip_short_maturity():
    """Recover known vol for short-dated option."""
    S0, K, r, T = 100, 100, 0.05, 0.1
    true_vol = 0.2
    price = black_scholes_call(S0, K, r, true_vol, T)
    recovered = implied_vol(price, S0, K, r, T)
    assert abs(recovered - true_vol) < 1e-8


def test_round_trip_long_maturity():
    """Recover known vol for long-dated option."""
    S0, K, r, T = 100, 100, 0.05, 5.0
    true_vol = 0.2
    price = black_scholes_call(S0, K, r, true_vol, T)
    recovered = implied_vol(price, S0, K, r, T)
    assert abs(recovered - true_vol) < 1e-8


def test_multiple_vols():
    """Round-trip across a range of volatilities."""
    S0, K, r, T = 100, 100, 0.05, 1.0
    for true_vol in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        price = black_scholes_call(S0, K, r, true_vol, T)
        recovered = implied_vol(price, S0, K, r, T)
        assert abs(recovered - true_vol) < 1e-8, f"Failed for vol={true_vol}"


def test_implied_vol_positive():
    """Recovered implied vol must be positive."""
    S0, K, r, T = 100, 100, 0.05, 1.0
    price = black_scholes_call(S0, K, r, 0.3, T)
    recovered = implied_vol(price, S0, K, r, T)
    assert recovered > 0