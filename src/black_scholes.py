import numpy as np
from scipy.stats import norm


def black_scholes_call(S0, K, r, sigma, T):
    """
    Black-Scholes closed-form price for a European call.

    Parameters
    ----------
    S0 : float
        Current stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate (annualized).
    sigma : float
        Volatility (annualized).
    T : float
        Time to maturity in years.

    Returns
    -------
    float
        Call option price.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S0, K, r, sigma, T):
    """
    Black-Scholes closed-form price for a European put.

    Parameters
    ----------
    S0 : float
        Current stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate (annualized).
    sigma : float
        Volatility (annualized).
    T : float
        Time to maturity in years.

    Returns
    -------
    float
        Put option price.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


if __name__ == "__main__":
    # Test parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    call_price = black_scholes_call(S0, K, r, sigma, T)
    put_price = black_scholes_put(S0, K, r, sigma, T)

    # Put-call parity check: C - P = S0 - K*exp(-rT)
    parity_lhs = call_price - put_price
    parity_rhs = S0 - K * np.exp(-r * T)

    print(f"BS Call Price:  {call_price:.6f}")
    print(f"BS Put Price:   {put_price:.6f}")
    print(f"Put-Call Parity LHS (C - P):       {parity_lhs:.10f}")
    print(f"Put-Call Parity RHS (S0 - Ke^-rT): {parity_rhs:.10f}")
    print(f"Parity Error: {abs(parity_lhs - parity_rhs):.2e}")
