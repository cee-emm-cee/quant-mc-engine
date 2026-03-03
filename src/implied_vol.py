import numpy as np
from scipy.stats import norm


def bs_price(S0, K, r, sigma, T):
    """BS call price for internal use."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S0, K, r, sigma, T):
    """BS vega for Newton-Raphson Jacobian."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * norm.pdf(d1) * np.sqrt(T)


def implied_vol(market_price, S0, K, r, T, tol=1e-10, max_iter=100):
    """
    Implied volatility via Newton-Raphson.

    Parameters
    ----------
    market_price : float
        Observed option price.
    S0 : float
        Current stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    float
        Implied volatility, or NaN if convergence fails.
    """
    sigma = 0.2

    for i in range(max_iter):
        price = bs_price(S0, K, r, sigma, T)
        v = bs_vega(S0, K, r, sigma, T)

        if v < 1e-12:
            return float('nan')

        sigma = sigma - (price - market_price) / v

        if sigma < 1e-6:
            sigma = 1e-6

        if abs(price - market_price) < tol:
            return sigma

    return float('nan')


if __name__ == "__main__":
    from black_scholes import black_scholes_call

    S0 = 100.0
    K = 100.0
    r = 0.05
    T = 1.0

    test_vols = [0.10, 0.20, 0.30, 0.40, 0.50]

    print("Round-trip test: BS price -> implied vol -> compare to original")
    print("=" * 65)
    print(f"{'True Vol':>10} {'BS Price':>10} {'Recovered Vol':>14} {'Error':>12}")
    print("=" * 65)

    for true_vol in test_vols:
        price = black_scholes_call(S0, K, r, true_vol, T)
        recovered = implied_vol(price, S0, K, r, T)
        error = abs(recovered - true_vol)
        print(f"{true_vol:>10.4f} {price:>10.6f} {recovered:>14.10f} {error:>12.2e}")