import numpy as np

def european_call_option(S0, K, r, sigma, T, number_of_sims=200000, seed=24):
    """
    Monte Carlo price for a European call under risk-neutral GBM using the exact terminal distribution.

    Returns:
      - price: Monte Carlo estimate
      - stderr: standard error of the discounted payoff estimator
      - ci95: 95% confidence interval (approx)
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