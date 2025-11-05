"""
Example: Monte Carlo Option Pricing

Demonstrates option pricing using Monte Carlo simulation.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qf.models import monte_carlo, black_scholes


def main():
    print("=" * 60)
    print("Monte Carlo Option Pricing Example")
    print("=" * 60)
    
    # Option parameters
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 1.0  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    print("\nOption Parameters:")
    print(f"  Stock Price (S):     ${S:.2f}")
    print(f"  Strike Price (K):    ${K:.2f}")
    print(f"  Time to Maturity:    {T} year(s)")
    print(f"  Risk-free Rate:      {r*100:.1f}%")
    print(f"  Volatility:          {sigma*100:.1f}%")
    
    # European Call Option
    print("\n" + "-" * 60)
    print("European Call Option")
    print("-" * 60)
    
    mc_call = monte_carlo.european_option(
        S, K, T, r, sigma, 
        option_type='call', 
        n_simulations=100000,
        seed=42
    )
    
    bs_call = black_scholes.call_option(S, K, T, r, sigma)
    
    print(f"Monte Carlo Price:      ${mc_call['price']:.4f}")
    print(f"Standard Error:         ${mc_call['std_error']:.4f}")
    print(f"95% Confidence:         ${mc_call['confidence_interval'][0]:.4f} - ${mc_call['confidence_interval'][1]:.4f}")
    print(f"Black-Scholes Price:    ${bs_call:.4f}")
    print(f"Difference:             ${abs(mc_call['price'] - bs_call):.4f}")
    
    # European Put Option
    print("\n" + "-" * 60)
    print("European Put Option")
    print("-" * 60)
    
    mc_put = monte_carlo.european_option(
        S, K, T, r, sigma,
        option_type='put',
        n_simulations=100000,
        seed=42
    )
    
    bs_put = black_scholes.put_option(S, K, T, r, sigma)
    
    print(f"Monte Carlo Price:      ${mc_put['price']:.4f}")
    print(f"Standard Error:         ${mc_put['std_error']:.4f}")
    print(f"95% Confidence:         ${mc_put['confidence_interval'][0]:.4f} - ${mc_put['confidence_interval'][1]:.4f}")
    print(f"Black-Scholes Price:    ${bs_put:.4f}")
    print(f"Difference:             ${abs(mc_put['price'] - bs_put):.4f}")
    
    # Asian Call Option
    print("\n" + "-" * 60)
    print("Asian Call Option (Average Price)")
    print("-" * 60)
    
    asian_call = monte_carlo.asian_option(
        S, K, T, r, sigma,
        option_type='call',
        n_simulations=50000,
        n_steps=252,
        seed=42
    )
    
    print(f"Monte Carlo Price:      ${asian_call['price']:.4f}")
    print(f"Standard Error:         ${asian_call['std_error']:.4f}")
    
    # Barrier Option
    print("\n" + "-" * 60)
    print("Barrier Option (Up-and-Out Call)")
    print("-" * 60)
    
    B = 120  # Barrier level
    print(f"  Barrier Level:        ${B:.2f}")
    
    barrier_call = monte_carlo.barrier_option(
        S, K, B, T, r, sigma,
        barrier_type='up-and-out',
        option_type='call',
        n_simulations=50000,
        n_steps=252,
        seed=42
    )
    
    print(f"Monte Carlo Price:      ${barrier_call['price']:.4f}")
    print(f"Standard Error:         ${barrier_call['std_error']:.4f}")
    print(f"European Call Price:    ${bs_call:.4f}")
    print(f"Barrier Discount:       {(1 - barrier_call['price']/bs_call)*100:.2f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
