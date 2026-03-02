import sys
sys.path.append('src')

import numpy as np
from black_scholes import black_scholes_call
from monte_carlo import european_call_option
from variance_reduction import antithetic_european_call, control_variate_european_call


def test_antithetic_converges_to_bs():
    """Antithetic pricer agrees with BS within 2 standard errors."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    bs = black_scholes_call(S0, K, r, sigma, T)
    av = antithetic_european_call(S0, K, r, sigma, T, number_of_sims=200000)
    assert abs(av['price'] - bs) < 2 * av['stderr']


def test_control_variate_converges_to_bs():
    """Control variate pricer agrees with BS within 2 standard errors."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    bs = black_scholes_call(S0, K, r, sigma, T)
    cv = control_variate_european_call(S0, K, r, sigma, T, number_of_sims=200000)
    assert abs(cv['price'] - bs) < 2 * cv['stderr']


def test_antithetic_reduces_variance():
    """Antithetic SE must be lower than naive MC SE."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    mc = european_call_option(S0, K, r, sigma, T, number_of_sims=200000)
    av = antithetic_european_call(S0, K, r, sigma, T, number_of_sims=200000)
    assert av['stderr'] < mc['stderr']


def test_control_variate_reduces_variance():
    """Control variate SE must be lower than naive MC SE."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    mc = european_call_option(S0, K, r, sigma, T, number_of_sims=200000)
    cv = control_variate_european_call(S0, K, r, sigma, T, number_of_sims=200000)
    assert cv['stderr'] < mc['stderr']


def test_control_variate_beats_antithetic():
    """Control variate SE must be lower than antithetic SE for ATM call."""
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    av = antithetic_european_call(S0, K, r, sigma, T, number_of_sims=200000)
    cv = control_variate_european_call(S0, K, r, sigma, T, number_of_sims=200000)
    assert cv['stderr'] < av['stderr']


def test_multiple_parameter_sets():
    """Both methods converge to BS across different parameter sets."""
    cases = [
        (100, 90, 0.05, 0.2, 1.0),
        (100, 110, 0.03, 0.3, 0.5),
        (50, 50, 0.08, 0.25, 2.0),
    ]
    for S0, K, r, sigma, T in cases:
        bs = black_scholes_call(S0, K, r, sigma, T)
        av = antithetic_european_call(S0, K, r, sigma, T, number_of_sims=200000)
        cv = control_variate_european_call(S0, K, r, sigma, T, number_of_sims=200000)
        assert abs(av['price'] - bs) < 3 * av['stderr'], f"AV failed for K={K}"
        assert abs(cv['price'] - bs) < 3 * cv['stderr'], f"CV failed for K={K}"