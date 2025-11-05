"""
Example: Black-Scholes Option Pricing

Demonstrates European option pricing using the Black-Scholes model.
"""

import sys
sys.path.insert(0, '/home/runner/work/d0m1ng0/d0m1ng0')

from qf.models import black_scholes


def main():
    print("=" * 60)
    print("Black-Scholes Option Pricing Example")
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
    
    # Calculate call option price
    call_price = black_scholes.call_option(S, K, T, r, sigma)
    print(f"\n{'Call Option Price:':<25} ${call_price:.4f}")
    
    # Calculate put option price
    put_price = black_scholes.put_option(S, K, T, r, sigma)
    print(f"{'Put Option Price:':<25} ${put_price:.4f}")
    
    # Verify put-call parity: C - P = S - K*e^(-rT)
    parity_diff = call_price - put_price - (S - K * (1 / (1 + r) ** T))
    print(f"\n{'Put-Call Parity Check:':<25} {abs(parity_diff):.8f} (should be ~0)")
    
    # Calculate Greeks for call option
    print("\nCall Option Greeks:")
    greeks = black_scholes.greeks(S, K, T, r, sigma, option_type='call')
    for greek, value in greeks.items():
        print(f"  {greek.capitalize():<10} {value:>12.6f}")
    
    # Calculate Greeks for put option
    print("\nPut Option Greeks:")
    greeks = black_scholes.greeks(S, K, T, r, sigma, option_type='put')
    for greek, value in greeks.items():
        print(f"  {greek.capitalize():<10} {value:>12.6f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
