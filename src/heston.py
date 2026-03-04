import numpy as np


def heston_european_call(S0, K, r, v0, kappa, theta, xi, rho, T,
                         n_steps=252, number_of_sims=200000, seed=24):
    """
    Monte Carlo price for a European call under the Heston stochastic volatility model.

    Parameters
    ----------
    S0 : float
        Current stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    v0 : float
        Initial variance.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run variance.
    xi : float
        Vol of vol.
    rho : float
        Correlation between stock and variance processes.
    T : float
        Time to maturity.
    n_steps : int
        Number of time steps.
    number_of_sims : int
        Number of simulations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        price, stderr, ci95
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    S = np.full(number_of_sims, S0)
    v = np.full(number_of_sims, v0)

    for _ in range(n_steps):
        Z_v = rng.standard_normal(number_of_sims)
        Z_perp = rng.standard_normal(number_of_sims)
        Z_S = rho * Z_v + np.sqrt(1 - rho**2) * Z_perp

        v_pos = np.maximum(v, 0)
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * np.sqrt(dt) * Z_v
        S = S * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * np.sqrt(dt) * Z_S)

    payoffs = np.maximum(S - K, 0.0)
    discounted = np.exp(-r * T) * payoffs

    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(number_of_sims)
    ci95 = (price - 1.96 * stderr, price + 1.96 * stderr)

    return {"price": price, "stderr": stderr, "ci95": ci95}


if __name__ == "__main__":
    from black_scholes import black_scholes_call

    S0 = 100.0
    K = 100.0
    r = 0.05
    T = 1.0

    # Heston parameters
    v0 = 0.04       # initial variance (vol = 0.2)
    kappa = 2.0     # mean reversion speed
    theta = 0.04    # long-run variance (vol = 0.2)
    xi = 0.0        # vol of vol = 0 (should collapse to BS)
    rho = -0.7

    bs = black_scholes_call(S0, K, r, 0.2, T)
    heston_flat = heston_european_call(S0, K, r, v0, kappa, theta, xi, rho, T)

    print("Heston with xi=0 should match BS (vol=0.2)")
    print(f"BS Price:      {bs:.6f}")
    print(f"Heston Price:  {heston_flat['price']:.6f}  (SE: {heston_flat['stderr']:.6f})")
    print(f"Match:         {abs(heston_flat['price'] - bs) < 3 * heston_flat['stderr']}")

    # Now with real Heston parameters
    xi = 0.3
    heston_real = heston_european_call(S0, K, r, v0, kappa, theta, xi, rho, T)
    print(f"\nHeston (xi=0.3, rho=-0.7): {heston_real['price']:.6f}  (SE: {heston_real['stderr']:.6f})")