import numpy as np


def asian_call_fixed_strike(S0, K, r, sigma, T, n_steps=252, number_of_sims=200000, seed=24):
    """
    Monte Carlo price for an arithmetic Asian call with fixed strike.

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
    n_steps : int
        Number of monitoring dates.
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
    running_sum = np.zeros(number_of_sims)

    for _ in range(n_steps):
        Z = rng.standard_normal(number_of_sims)
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        running_sum += S

    S_avg = running_sum / n_steps
    payoffs = np.maximum(S_avg - K, 0.0)
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

    bs = black_scholes_call(S0, K, r, sigma, T)
    asian = asian_call_fixed_strike(S0, K, r, sigma, T)
    asian_1 = asian_call_fixed_strike(S0, K, r, sigma, T, n_steps=1)

    print(f"BS European Call:          {bs:.6f}")
    print(f"Asian Call (252 steps):     {asian['price']:.6f}  (SE: {asian['stderr']:.6f})")
    print(f"Asian Call (1 step):        {asian_1['price']:.6f}  (SE: {asian_1['stderr']:.6f})")
    print(f"Asian < European:           {asian['price'] < bs}")
    print(f"Asian(1 step) ~ European:   {abs(asian_1['price'] - bs) < 3 * asian_1['stderr']}")