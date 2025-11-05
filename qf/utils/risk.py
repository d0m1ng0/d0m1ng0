"""
Risk Management Tools

Measure and manage financial risk.
"""

import numpy as np
from scipy import stats


def value_at_risk(returns, confidence_level=0.95, method='historical'):
    """
    Calculate Value at Risk (VaR).
    
    Parameters:
    -----------
    returns : array-like
        Historical returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    method : str
        'historical' or 'parametric'
    
    Returns:
    --------
    float
        Value at Risk (negative value represents loss)
    """
    if method == 'historical':
        return np.percentile(returns, (1 - confidence_level) * 100)
    elif method == 'parametric':
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence_level)
        return mean + z_score * std
    else:
        raise ValueError("Method must be 'historical' or 'parametric'")


def conditional_value_at_risk(returns, confidence_level=0.95):
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    CVaR is the expected loss given that the loss exceeds VaR.
    
    Parameters:
    -----------
    returns : array-like
        Historical returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    
    Returns:
    --------
    float
        Conditional Value at Risk
    """
    var = value_at_risk(returns, confidence_level, method='historical')
    return np.mean(returns[returns <= var])


def maximum_drawdown(prices):
    """
    Calculate maximum drawdown from peak.
    
    Parameters:
    -----------
    prices : array-like
        Price series
    
    Returns:
    --------
    dict
        Dictionary containing max drawdown, peak index, and trough index
    """
    prices = np.array(prices)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(prices)
    
    # Calculate drawdown
    drawdown = (prices - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    
    # Find peak before maximum drawdown
    peak_idx = np.argmax(prices[:max_dd_idx + 1])
    
    return {
        'max_drawdown': max_dd,
        'peak_index': peak_idx,
        'trough_index': max_dd_idx
    }


def beta(asset_returns, market_returns):
    """
    Calculate beta (systematic risk) of an asset.
    
    Parameters:
    -----------
    asset_returns : array-like
        Asset returns
    market_returns : array-like
        Market returns
    
    Returns:
    --------
    float
        Beta coefficient
    """
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    return covariance / market_variance


def alpha(asset_returns, market_returns, risk_free_rate=0.0):
    """
    Calculate Jensen's alpha (excess return over CAPM prediction).
    
    Parameters:
    -----------
    asset_returns : array-like
        Asset returns
    market_returns : array-like
        Market returns
    risk_free_rate : float
        Risk-free rate
    
    Returns:
    --------
    float
        Jensen's alpha
    """
    asset_mean = np.mean(asset_returns)
    market_mean = np.mean(market_returns)
    beta_value = beta(asset_returns, market_returns)
    
    expected_return = risk_free_rate + beta_value * (market_mean - risk_free_rate)
    return asset_mean - expected_return


def tracking_error(portfolio_returns, benchmark_returns):
    """
    Calculate tracking error (volatility of excess returns).
    
    Parameters:
    -----------
    portfolio_returns : array-like
        Portfolio returns
    benchmark_returns : array-like
        Benchmark returns
    
    Returns:
    --------
    float
        Tracking error (annualized)
    """
    excess_returns = np.array(portfolio_returns) - np.array(benchmark_returns)
    return np.std(excess_returns) * np.sqrt(252)  # Annualized


def information_ratio(portfolio_returns, benchmark_returns):
    """
    Calculate information ratio (risk-adjusted excess return).
    
    Parameters:
    -----------
    portfolio_returns : array-like
        Portfolio returns
    benchmark_returns : array-like
        Benchmark returns
    
    Returns:
    --------
    float
        Information ratio
    """
    excess_returns = np.array(portfolio_returns) - np.array(benchmark_returns)
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0):
    """
    Calculate Sortino ratio (downside risk-adjusted return).
    
    Parameters:
    -----------
    returns : array-like
        Returns
    risk_free_rate : float
        Risk-free rate
    target_return : float
        Target return threshold
    
    Returns:
    --------
    float
        Sortino ratio
    """
    excess_returns = np.array(returns) - risk_free_rate
    downside_returns = excess_returns[excess_returns < target_return]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    return np.mean(excess_returns) / downside_deviation


def stress_test(returns, scenario_shocks):
    """
    Perform stress testing with given scenario shocks.
    
    Parameters:
    -----------
    returns : array-like
        Current returns distribution
    scenario_shocks : list of float
        List of shock magnitudes to test
    
    Returns:
    --------
    dict
        Dictionary with shock scenarios and resulting portfolio values
    """
    current_value = 1.0  # Normalized starting value
    results = {}
    
    for shock in scenario_shocks:
        shocked_value = current_value * (1 + shock)
        results[f'shock_{shock:.2%}'] = {
            'final_value': shocked_value,
            'loss': shocked_value - current_value,
            'loss_pct': shock
        }
    
    return results
