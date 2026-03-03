import numpy as np


def delta(pricer, S0, K, r, sigma, T, h=None, **kwargs):
    """
    Delta via central finite difference.
    Bumps S0.
    """
    if h is None:
        h = S0 * 0.01
    price_up = pricer(S0 + h, K, r, sigma, T, **kwargs)['price']
    price_down = pricer(S0 - h, K, r, sigma, T, **kwargs)['price']
    return (price_up - price_down) / (2 * h)


def gamma(pricer, S0, K, r, sigma, T, h=None, **kwargs):
    """
    Gamma via central finite difference.
    Second derivative with respect to S0.
    """
    if h is None:
        h = S0 * 0.01
    price_up = pricer(S0 + h, K, r, sigma, T, **kwargs)['price']
    price_mid = pricer(S0, K, r, sigma, T, **kwargs)['price']
    price_down = pricer(S0 - h, K, r, sigma, T, **kwargs)['price']
    return (price_up - 2 * price_mid + price_down) / (h ** 2)


def vega(pricer, S0, K, r, sigma, T, h=None, **kwargs):
    """
    Vega via central finite difference.
    Bumps sigma.
    """
    if h is None:
        h = sigma * 0.01
    price_up = pricer(S0, K, r, sigma + h, T, **kwargs)['price']
    price_down = pricer(S0, K, r, sigma - h, T, **kwargs)['price']
    return (price_up - price_down) / (2 * h)


def theta(pricer, S0, K, r, sigma, T, h=None, **kwargs):
    """
    Theta via forward finite difference.
    Bumps T downward (less time remaining = lower price for calls).
    """
    if h is None:
        h = T * 0.01
    price_now = pricer(S0, K, r, sigma, T, **kwargs)['price']
    price_later = pricer(S0, K, r, sigma, T - h, **kwargs)['price']
    return (price_later - price_now) / h


def rho(pricer, S0, K, r, sigma, T, h=None, **kwargs):
    """
    Rho via central finite difference.
    Bumps r.
    """
    if h is None:
        h = r * 0.01 if r != 0 else 0.0001
    price_up = pricer(S0, K, r + h, sigma, T, **kwargs)['price']
    price_down = pricer(S0, K, r - h, sigma, T, **kwargs)['price']
    return (price_up - price_down) / (2 * h)
def _bs_pricer(S0, K, r, sigma, T, **kwargs):
    """Wrapper to make BS compatible with Greek functions."""
    from black_scholes import black_scholes_call
    return {"price": black_scholes_call(S0, K, r, sigma, T)}


if __name__ == "__main__":
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    print("Greeks via bump-and-reprice (BS pricer)")
    print("=" * 45)
    print(f"Delta:  {delta(_bs_pricer, S0, K, r, sigma, T):.6f}")
    print(f"Gamma:  {gamma(_bs_pricer, S0, K, r, sigma, T):.6f}")
    print(f"Vega:   {vega(_bs_pricer, S0, K, r, sigma, T):.6f}")
    print(f"Theta:  {theta(_bs_pricer, S0, K, r, sigma, T):.6f}")
    print(f"Rho:    {rho(_bs_pricer, S0, K, r, sigma, T):.6f}")

    from monte_carlo import european_call_option

    print("\nGreeks via bump-and-reprice (MC pricer)")
    print("=" * 45)
    print(f"Delta:  {delta(european_call_option, S0, K, r, sigma, T):.6f}")
    print(f"Gamma:  {gamma(european_call_option, S0, K, r, sigma, T):.6f}")
    print(f"Vega:   {vega(european_call_option, S0, K, r, sigma, T):.6f}")
    print(f"Theta:  {theta(european_call_option, S0, K, r, sigma, T):.6f}")
    print(f"Rho:    {rho(european_call_option, S0, K, r, sigma, T):.6f}")