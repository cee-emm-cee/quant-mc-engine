import numpy as np
import matplotlib.pyplot as plt

def plot_gbm_paths(S0=100, r=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=10, seed=24):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)

    plt.figure(figsize=(10, 6))

    for _ in range(n_paths):
        Z = rng.standard_normal(n_steps)
        S = np.zeros(n_steps + 1)
        S[0] = S0
        for t in range(n_steps):
            S[t + 1] = S[t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t])
        plt.plot(times, S, alpha=0.7)

    plt.axhline(y=S0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Time (Years)')
    plt.ylabel('Stock Price ($)')
    plt.title('Geometric Brownian Motion: Simulated Stock Price Paths')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../figures/gbm_paths.png', dpi=300)
    plt.show()

plot_gbm_paths()