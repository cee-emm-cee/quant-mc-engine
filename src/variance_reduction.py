import numpy as np


def antithetic_european_call(S0, K, r, sigma, T, number_of_sims=200000, seed=24):
    """
    Monte Carlo price for a European call using antithetic variates.

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
        Total number of simulations (must be even).
    seed : int
        Random seed.

    Returns
    -------
    dict
        price, stderr, ci95
    """
    rng = np.random.default_rng(seed)
    half_n = number_of_sims // 2

    Z = rng.standard_normal(half_n)

    drift = (r - 0.5 * sigma**2) * T
    diff = sigma * np.sqrt(T)

    ST_pos = S0 * np.exp(drift + diff * Z)
    ST_neg = S0 * np.exp(drift + diff * (-Z))

    payoff_pos = np.maximum(ST_pos - K, 0.0)
    payoff_neg = np.maximum(ST_neg - K, 0.0)

    paired_avg = 0.5 * (payoff_pos + payoff_neg)
    discounted = np.exp(-r * T) * paired_avg

    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(half_n)
    ci95 = (price - 1.96 * stderr, price + 1.96 * stderr)

    return {"price": price, "stderr": stderr, "ci95": ci95}


if __name__ == "__main__":
    from black_scholes import black_scholes_call
    from monte_carlo import european_call_option

    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    bs = black_scholes_call(S0, K, r, sigma, T)
    mc = european_call_option(S0, K, r, sigma, T)
    av = antithetic_european_call(S0, K, r, sigma, T)

    print(f"BS Price:              {bs:.6f}")
    print(f"Naive MC Price:        {mc['price']:.6f}  (SE: {mc['stderr']:.6f})")
    print(f"Antithetic MC Price:   {av['price']:.6f}  (SE: {av['stderr']:.6f})")
    print(f"Variance Reduction:    {(1 - (av['stderr'] / mc['stderr'])**2) * 100:.1f}%")
    print(f"BS within AV CI:       {av['ci95'][0] <= bs <= av['ci95'][1]}")