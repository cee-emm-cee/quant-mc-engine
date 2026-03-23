import numpy as np

def european_call_option(S0, K, r, sigma, T, number_of_sims=200000, seed=24):
    """
    Monte Carlo price for a European call under risk-neutral GBM.

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

    Z = rng.standard_normal(number_of_sims)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST = S0 * np.exp(drift + diffusion)

    payoffs = np.maximum(ST - K, 0.0)
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
    sigma = 0.2
    T = 1.0

    mc_result = european_call_option(S0, K, r, sigma, T)
    bs_price = black_scholes_call(S0, K, r, sigma, T)

    print(f"MC Price:       {mc_result['price']:.6f}")
    print(f"BS Price:       {bs_price:.6f}")
    print(f"Difference:     {abs(mc_result['price'] - bs_price):.6f}")
    print(f"MC Std Error:   {mc_result['stderr']:.6f}")
    print(f"MC 95% CI:      ({mc_result['ci95'][0]:.6f}, {mc_result['ci95'][1]:.6f})")
    print(f"BS within CI:   {mc_result['ci95'][0] <= bs_price <= mc_result['ci95'][1]}")