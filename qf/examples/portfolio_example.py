"""
Example: Portfolio Optimization

Demonstrates portfolio optimization using Markowitz mean-variance theory.
"""

import sys
sys.path.insert(0, '/home/runner/work/d0m1ng0/d0m1ng0')

import numpy as np
from qf.utils import portfolio


def main():
    print("=" * 60)
    print("Portfolio Optimization Example")
    print("=" * 60)
    
    # Asset parameters (3 assets)
    asset_names = ['Stock A', 'Stock B', 'Stock C']
    expected_returns = np.array([0.12, 0.10, 0.08])  # 12%, 10%, 8% annual returns
    
    # Covariance matrix (annual)
    cov_matrix = np.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.03, 0.008],
        [0.005, 0.008, 0.02]
    ])
    
    risk_free_rate = 0.03  # 3% risk-free rate
    
    print("\nAsset Expected Returns:")
    for name, ret in zip(asset_names, expected_returns):
        print(f"  {name:<12} {ret*100:.2f}%")
    
    print("\nCovariance Matrix:")
    for i, name in enumerate(asset_names):
        row_str = f"  {name:<12}"
        for j in range(len(asset_names)):
            row_str += f" {cov_matrix[i, j]:>8.4f}"
        print(row_str)
    
    print(f"\nRisk-free Rate:      {risk_free_rate*100:.2f}%")
    
    # Equal-weight portfolio
    print("\n" + "-" * 60)
    print("Equal-Weight Portfolio")
    print("-" * 60)
    
    equal_weights = portfolio.equal_weight_portfolio(len(asset_names))
    eq_return = portfolio.portfolio_return(equal_weights, expected_returns)
    eq_variance = portfolio.portfolio_variance(equal_weights, cov_matrix)
    eq_std = np.sqrt(eq_variance)
    eq_sharpe = portfolio.sharpe_ratio(eq_return, risk_free_rate, eq_std)
    
    print("Weights:")
    for name, weight in zip(asset_names, equal_weights):
        print(f"  {name:<12} {weight*100:.2f}%")
    print(f"\nExpected Return:     {eq_return*100:.2f}%")
    print(f"Volatility:          {eq_std*100:.2f}%")
    print(f"Sharpe Ratio:        {eq_sharpe:.4f}")
    
    # Minimum variance portfolio
    print("\n" + "-" * 60)
    print("Minimum Variance Portfolio")
    print("-" * 60)
    
    min_var = portfolio.minimum_variance_portfolio(cov_matrix)
    mv_return = portfolio.portfolio_return(min_var['weights'], expected_returns)
    mv_sharpe = portfolio.sharpe_ratio(mv_return, risk_free_rate, min_var['volatility'])
    
    print("Weights:")
    for name, weight in zip(asset_names, min_var['weights']):
        print(f"  {name:<12} {weight*100:.2f}%")
    print(f"\nExpected Return:     {mv_return*100:.2f}%")
    print(f"Volatility:          {min_var['volatility']*100:.2f}%")
    print(f"Sharpe Ratio:        {mv_sharpe:.4f}")
    
    # Optimal portfolio (maximum Sharpe ratio)
    print("\n" + "-" * 60)
    print("Optimal Portfolio (Maximum Sharpe Ratio)")
    print("-" * 60)
    
    optimal = portfolio.markowitz_optimization(expected_returns, cov_matrix, risk_free_rate)
    
    print("Weights:")
    for name, weight in zip(asset_names, optimal['weights']):
        print(f"  {name:<12} {weight*100:.2f}%")
    print(f"\nExpected Return:     {optimal['return']*100:.2f}%")
    print(f"Volatility:          {optimal['volatility']*100:.2f}%")
    print(f"Sharpe Ratio:        {optimal['sharpe_ratio']:.4f}")
    
    # Comparison
    print("\n" + "-" * 60)
    print("Portfolio Comparison")
    print("-" * 60)
    
    print(f"{'Portfolio':<30} {'Return':<12} {'Volatility':<12} {'Sharpe':<10}")
    print("-" * 64)
    print(f"{'Equal-Weight':<30} {eq_return*100:>10.2f}% {eq_std*100:>10.2f}% {eq_sharpe:>10.4f}")
    print(f"{'Minimum Variance':<30} {mv_return*100:>10.2f}% {min_var['volatility']*100:>10.2f}% {mv_sharpe:>10.4f}")
    print(f"{'Optimal (Max Sharpe)':<30} {optimal['return']*100:>10.2f}% {optimal['volatility']*100:>10.2f}% {optimal['sharpe_ratio']:>10.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
