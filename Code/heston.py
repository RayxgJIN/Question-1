import numpy as np
import matplotlib.pyplot as plt

# Heston model parameters
S0 = 100              # Initial stock price
v0 = 0.2              # Initial volatility
K = 100               # Strike price
T = 1                 # Time to maturity (1 year)
r = 0.05              # Risk-free rate
theta = 0.2           # Long-term mean of volatility
kappa = 0.075         # Mean reversion rate
xi = 0.1              # Volatility of volatility
n_steps = 252         # Number of time steps (trading days)
dt = T / n_steps      # Time increment

# HMC parameters
epsilon = 0.01  # Step size
L = 20          # Number of leapfrog steps

def heston_dynamics(S, v):
    """Calculate the gradients for Heston model dynamics."""
    dS = r * S + np.sqrt(v) * S * np.random.normal()
    dv = kappa * (theta - v) * dt + xi * np.sqrt(v) * np.random.normal()
    return dS, dv

def leapfrog(S, v, pS, pv, epsilon, L):
    """Leapfrog integration for HMC."""
    for _ in range(L):
        # Half step for momentum
        pS -= 0.5 * epsilon * (S * pS)
        S += epsilon * (pS + r * S)
        v += epsilon * pv
        pv -= 0.5 * epsilon * (kappa * (theta - v))
    # Final momentum update
    pS -= 0.5 * epsilon * (S * pS)
    return S, v, pS, pv

def V(S, v):
    """Potential energy function."""
    return 0.5 * (S - K) ** 2 + 0.5 * (v - theta) ** 2

def KineticEnergy(pS, pv):
    """Kinetic energy function."""
    return 0.5 * pS ** 2 + 0.5 * pv ** 2

def hmc(S0, v0, n_samples, epsilon, L):
    """Hamiltonian Monte Carlo sampling."""
    samples = []
    S, v = S0, v0
    pS = np.random.normal(0, 1)  # Initial momentum for stock price
    pv = np.random.normal(0, 1)   # Initial momentum for volatility

    for _ in range(n_samples):
        S_new, v_new, pS_new, pv_new = leapfrog(S, v, pS, pv, epsilon, L)
        H_old = V(S, v) + KineticEnergy(pS, pv)
        H_new = V(S_new, v_new) + KineticEnergy(pS_new, pv_new)

        # Metropolis acceptance step
        if np.random.rand() < np.exp(H_old - H_new):
            S, v, pS, pv = S_new, v_new, pS_new, pv_new
        
        samples.append((S, v))

    return samples

# Run the HMC simulation
samples = hmc(S0, v0, n_steps, epsilon, L)

# Extract stock prices and volatilities
S_samples, v_samples = zip(*samples)

# Plotting
plt.figure(figsize=(12, 8))

# Top panel: Stock price
plt.subplot(2, 1, 1)
plt.plot(S_samples, label='Stock Price $S_t$', color='blue')
plt.title('Heston Model Dynamics via HMC')
plt.ylabel('Stock Price')
plt.legend()

# Bottom panel: Instantaneous volatility
plt.subplot(2, 1, 2)
plt.plot(v_samples, label='Instantaneous Volatility $ν_t$', color='orange')
plt.xlabel('Days')
plt.ylabel('Volatility')
plt.legend()

plt.tight_layout()
plt.show()

# Print summary statistics
print("Mean Final Stock Price:", S_samples[-1])
print("Mean Instantaneous Volatility:", np.mean(v_samples))