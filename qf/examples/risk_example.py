"""
Example: Risk Management

Demonstrates various risk metrics and measures.
"""

import sys
sys.path.insert(0, '/home/runner/work/d0m1ng0/d0m1ng0')

import numpy as np
from qf.utils import risk


def main():
    print("=" * 60)
    print("Risk Management Example")
    print("=" * 60)
    
    # Generate sample returns (simulating daily returns for 1 year)
    np.random.seed(42)
    n_days = 252
    daily_returns = np.random.normal(0.0005, 0.02, n_days)  # ~12.6% annual return, ~32% volatility
    
    # Generate sample prices
    initial_price = 100
    prices = [initial_price]
    for ret in daily_returns:
        prices.append(prices[-1] * (1 + ret))
    prices = np.array(prices)
    
    print(f"\nPortfolio Statistics:")
    print(f"  Number of Days:      {n_days}")
    print(f"  Mean Daily Return:   {np.mean(daily_returns)*100:.4f}%")
    print(f"  Daily Volatility:    {np.std(daily_returns)*100:.4f}%")
    print(f"  Annual Return:       {np.mean(daily_returns)*252*100:.2f}%")
    print(f"  Annual Volatility:   {np.std(daily_returns)*np.sqrt(252)*100:.2f}%")
    
    # Value at Risk
    print("\n" + "-" * 60)
    print("Value at Risk (VaR)")
    print("-" * 60)
    
    var_95_hist = risk.value_at_risk(daily_returns, confidence_level=0.95, method='historical')
    var_99_hist = risk.value_at_risk(daily_returns, confidence_level=0.99, method='historical')
    var_95_para = risk.value_at_risk(daily_returns, confidence_level=0.95, method='parametric')
    var_99_para = risk.value_at_risk(daily_returns, confidence_level=0.99, method='parametric')
    
    print(f"95% VaR (Historical):    {var_95_hist*100:.4f}%")
    print(f"99% VaR (Historical):    {var_99_hist*100:.4f}%")
    print(f"95% VaR (Parametric):    {var_95_para*100:.4f}%")
    print(f"99% VaR (Parametric):    {var_99_para*100:.4f}%")
    
    # Conditional Value at Risk
    print("\n" + "-" * 60)
    print("Conditional Value at Risk (CVaR / Expected Shortfall)")
    print("-" * 60)
    
    cvar_95 = risk.conditional_value_at_risk(daily_returns, confidence_level=0.95)
    cvar_99 = risk.conditional_value_at_risk(daily_returns, confidence_level=0.99)
    
    print(f"95% CVaR:                {cvar_95*100:.4f}%")
    print(f"99% CVaR:                {cvar_99*100:.4f}%")
    
    # Maximum Drawdown
    print("\n" + "-" * 60)
    print("Maximum Drawdown")
    print("-" * 60)
    
    mdd = risk.maximum_drawdown(prices)
    
    print(f"Max Drawdown:            {mdd['max_drawdown']*100:.2f}%")
    print(f"Peak Index (Day):        {mdd['peak_index']}")
    print(f"Trough Index (Day):      {mdd['trough_index']}")
    print(f"Peak Price:              ${prices[mdd['peak_index']]:.2f}")
    print(f"Trough Price:            ${prices[mdd['trough_index']]:.2f}")
    
    # Beta and Alpha (relative to market)
    print("\n" + "-" * 60)
    print("Market Risk Metrics (Beta & Alpha)")
    print("-" * 60)
    
    # Simulate market returns
    market_returns = np.random.normal(0.0004, 0.015, n_days)  # Market: ~10% return, ~24% volatility
    
    beta_value = risk.beta(daily_returns, market_returns)
    alpha_value = risk.alpha(daily_returns, market_returns, risk_free_rate=0.03/252)
    
    print(f"Beta:                    {beta_value:.4f}")
    print(f"Alpha (daily):           {alpha_value*100:.4f}%")
    print(f"Alpha (annualized):      {alpha_value*252*100:.2f}%")
    
    # Tracking Error and Information Ratio
    print("\n" + "-" * 60)
    print("Tracking Error & Information Ratio")
    print("-" * 60)
    
    te = risk.tracking_error(daily_returns, market_returns)
    ir = risk.information_ratio(daily_returns, market_returns)
    
    print(f"Tracking Error:          {te*100:.2f}%")
    print(f"Information Ratio:       {ir:.4f}")
    
    # Sortino Ratio
    print("\n" + "-" * 60)
    print("Sortino Ratio (Downside Risk)")
    print("-" * 60)
    
    sortino = risk.sortino_ratio(daily_returns, risk_free_rate=0.03/252)
    
    print(f"Sortino Ratio:           {sortino:.4f}")
    
    # Stress Testing
    print("\n" + "-" * 60)
    print("Stress Testing Scenarios")
    print("-" * 60)
    
    scenarios = [-0.10, -0.20, -0.30, -0.40]  # -10%, -20%, -30%, -40% shocks
    stress_results = risk.stress_test(daily_returns, scenarios)
    
    print(f"{'Scenario':<20} {'Final Value':<15} {'Loss':<15} {'Loss %':<10}")
    print("-" * 60)
    for scenario_name, result in stress_results.items():
        print(f"{scenario_name:<20} ${result['final_value']:>12.4f} ${result['loss']:>12.4f} {result['loss_pct']*100:>8.2f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
