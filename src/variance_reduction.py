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

def control_variate_european_call(S0, K, r, sigma, T, number_of_sims=200000, seed=24):
    """
    Monte Carlo price for a European call using control variates.
    Control variable: S_T with known expectation S0 * exp(rT).

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

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoffs

    # Control: S_T with known expectation
    expected_ST = S0 * np.exp(r * T)

    # Estimate optimal beta
    cov = np.cov(discounted, ST)[0, 1]
    var = np.var(ST, ddof=1)
    beta = cov / var

    # Apply correction
    adjusted = discounted - beta * (ST - expected_ST)

    price = adjusted.mean()
    stderr = adjusted.std(ddof=1) / np.sqrt(number_of_sims)
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
    cv = control_variate_european_call(S0, K, r, sigma, T)

    print(f"BS Price:              {bs:.6f}")
    print(f"Naive MC Price:        {mc['price']:.6f}  (SE: {mc['stderr']:.6f})")
    print(f"Antithetic MC Price:   {av['price']:.6f}  (SE: {av['stderr']:.6f})")
    print(f"Control Var MC Price:  {cv['price']:.6f}  (SE: {cv['stderr']:.6f})")
    print(f"AV Variance Reduction: {(1 - (av['stderr'] / mc['stderr'])**2) * 100:.1f}%")
    print(f"CV Variance Reduction: {(1 - (cv['stderr'] / mc['stderr'])**2) * 100:.1f}%")
    print(f"BS within CV CI:       {cv['ci95'][0] <= bs <= cv['ci95'][1]}")