# Quant Monte Carlo Engine

A Monte Carlo pricing engine for vanilla and exotic derivatives, built from scratch in Python. Includes variance reduction methods, Greeks computation, implied volatility surface construction, and stochastic volatility modeling via the Heston model.

## Repository Structure

```
quant-mc-engine/
├── src/
│   ├── black_scholes.py        # Closed-form European call and put pricing
│   ├── monte_carlo.py          # Naive Monte Carlo pricer (risk-neutral GBM)
│   ├── variance_reduction.py   # Antithetic variates and control variate pricers
│   ├── exotics.py              # Arithmetic Asian call and up-and-out barrier call
│   ├── greeks.py               # Delta, Gamma, Vega, Theta, Rho via bump-and-reprice
│   ├── implied_vol.py          # Newton-Raphson implied volatility solver
│   └── heston.py               # Heston stochastic volatility Monte Carlo pricer
├── tests/
│   ├── test_black_scholes.py
│   ├── test_variance_reduction.py
│   ├── test_exotics.py
│   ├── test_greeks.py
│   ├── test_implied_vol.py
│   └── test_heston.py
├── notebooks/
│   ├── 01_convergence_analysis.ipynb
│   ├── 02_exotics_pricing.ipynb
│   ├── 03_greeks_surfaces.ipynb
│   ├── 04_implied_vol_surface.ipynb
│   └── 05_heston_model.ipynb
├── figures/
├── requirements.txt
└── README.md
```

## Features

### Pricing Models
- **Black-Scholes**: Closed-form European call and put with put-call parity validation
- **Monte Carlo (Naive)**: Risk-neutral GBM simulation with standard error and confidence intervals
- **Antithetic Variates**: ~50% variance reduction via negatively correlated path pairs
- **Control Variates**: ~85% variance reduction using S_T as control with optimal beta estimation

### Exotic Options
- **Arithmetic Asian Call**: Path-dependent pricing with configurable monitoring dates. No closed-form solution exists.
- **Up-and-Out Barrier Call**: Discrete barrier monitoring with knockout probability tracking

### Risk Sensitivities
- **Greeks via Bump-and-Reprice**: Delta, Gamma, Vega, Theta, Rho computed through central finite differences
- **Pricer-Agnostic**: Greek functions accept any pricing function as input (Black-Scholes, Monte Carlo, exotics, Heston)
- **3D Surface Plots**: Delta, Gamma, and Vega surfaces across spot price and time to maturity

### Implied Volatility
- **Newton-Raphson Solver**: Quadratic convergence, machine-precision recovery of known volatilities
- **Volatility Surface Construction**: Smile cross-sections and full 3D implied vol surface from synthetic market data

### Stochastic Volatility
- **Heston Model**: Full Euler discretization with log-Euler stock process and full truncation variance scheme
- **Correlated Brownian Motions**: Cholesky decomposition for stock-variance correlation
- **Implied Vol Smile**: Heston naturally produces the volatility skew observed in equity markets
- **Rho Sensitivity Analysis**: Visualization of how stock-variance correlation drives smile shape

## Results

| Method | Price | Std Error | Variance Reduction |
|--------|-------|-----------|-------------------|
| Black-Scholes | 10.4506 | -- | -- |
| Naive MC (200k sims) | 10.4595 | 0.0329 | -- |
| Antithetic MC | 10.4655 | 0.0234 | 49.6% |
| Control Variate MC | 10.4682 | 0.0126 | 85.4% |

Parameters: S0=100, K=100, r=0.05, sigma=0.2, T=1.0

## Installation

```bash
git clone https://github.com/cee-emm-cee/quant-mc-engine.git
cd quant-mc-engine
pip install -r requirements.txt
```

## Usage

```python
from src.black_scholes import black_scholes_call
from src.monte_carlo import european_call_option
from src.variance_reduction import control_variate_european_call
from src.greeks import delta, gamma, vega

# Price a European call
bs_price = black_scholes_call(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)

# Monte Carlo with control variates
mc_result = control_variate_european_call(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)

# Compute Delta using any pricer
from src.greeks import _bs_pricer
d = delta(_bs_pricer, S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
```

## Testing

```bash
python3 -m pytest tests/ -v
```

38 tests covering put-call parity, convergence validation, variance reduction verification, Greeks accuracy against analytical solutions, implied vol round-trip recovery, and Heston model boundary conditions.

## Mathematical Foundation

### Risk-Neutral Pricing

All simulations use the risk-neutral measure where the stock evolves as:

$$dS = rS \, dt + \sigma S \, dW$$

The option price is the discounted expected payoff under this measure.

### Heston Stochastic Volatility

The Heston model replaces constant volatility with a mean-reverting stochastic variance process:

$$dS = rS \, dt + \sqrt{v} \, S \, dW_S$$

$$dv = \kappa(\theta - v) \, dt + \xi \sqrt{v} \, dW_v$$

$$\text{Corr}(dW_S, dW_v) = \rho$$

Negative rho produces the left skew observed in equity implied volatility surfaces.

## Built With
- Python 3.14
- NumPy
- SciPy
- Matplotlib
- pytest
