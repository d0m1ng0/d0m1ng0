"""
Portfolio Optimization and Management

Tools for portfolio construction and optimization.
"""

import numpy as np


def expected_return(returns):
    """
    Calculate expected return of an asset or portfolio.
    
    Parameters:
    -----------
    returns : array-like
        Historical returns
    
    Returns:
    --------
    float
        Expected return (mean)
    """
    return np.mean(returns)


def portfolio_variance(weights, cov_matrix):
    """
    Calculate portfolio variance.
    
    Parameters:
    -----------
    weights : array-like
        Portfolio weights (must sum to 1)
    cov_matrix : array-like
        Covariance matrix of asset returns
    
    Returns:
    --------
    float
        Portfolio variance
    """
    return np.dot(weights, np.dot(cov_matrix, weights))


def portfolio_return(weights, expected_returns):
    """
    Calculate expected portfolio return.
    
    Parameters:
    -----------
    weights : array-like
        Portfolio weights (must sum to 1)
    expected_returns : array-like
        Expected returns of each asset
    
    Returns:
    --------
    float
        Expected portfolio return
    """
    return np.dot(weights, expected_returns)


def sharpe_ratio(portfolio_return, risk_free_rate, portfolio_std):
    """
    Calculate Sharpe ratio (risk-adjusted return).
    
    Parameters:
    -----------
    portfolio_return : float
        Portfolio return
    risk_free_rate : float
        Risk-free rate
    portfolio_std : float
        Portfolio standard deviation
    
    Returns:
    --------
    float
        Sharpe ratio
    """
    return (portfolio_return - risk_free_rate) / portfolio_std


def markowitz_optimization(expected_returns, cov_matrix, risk_free_rate=0.0):
    """
    Find optimal portfolio weights using Markowitz mean-variance optimization.
    
    This implementation finds the tangency portfolio (maximum Sharpe ratio).
    
    Parameters:
    -----------
    expected_returns : array-like
        Expected returns of each asset
    cov_matrix : array-like
        Covariance matrix of asset returns
    risk_free_rate : float
        Risk-free rate
    
    Returns:
    --------
    dict
        Dictionary containing optimal weights, return, volatility, and Sharpe ratio
    """
    n_assets = len(expected_returns)
    
    # Inverse of covariance matrix
    cov_inv = np.linalg.inv(cov_matrix)
    
    # Excess returns
    excess_returns = expected_returns - risk_free_rate
    
    # Optimal weights (tangency portfolio)
    weights = np.dot(cov_inv, excess_returns)
    weights = weights / np.sum(weights)
    
    # Portfolio metrics
    port_return = portfolio_return(weights, expected_returns)
    port_variance = portfolio_variance(weights, cov_matrix)
    port_std = np.sqrt(port_variance)
    sharpe = sharpe_ratio(port_return, risk_free_rate, port_std)
    
    return {
        'weights': weights,
        'return': port_return,
        'volatility': port_std,
        'sharpe_ratio': sharpe
    }


def efficient_frontier(expected_returns, cov_matrix, n_points=50):
    """
    Calculate the efficient frontier.
    
    Parameters:
    -----------
    expected_returns : array-like
        Expected returns of each asset
    cov_matrix : array-like
        Covariance matrix of asset returns
    n_points : int
        Number of points on the frontier
    
    Returns:
    --------
    tuple
        (returns, volatilities) arrays for the efficient frontier
    """
    n_assets = len(expected_returns)
    
    # Generate random portfolios
    returns = []
    volatilities = []
    
    for _ in range(n_points * 100):
        # Random weights
        w = np.random.random(n_assets)
        w = w / np.sum(w)
        
        # Portfolio metrics
        ret = portfolio_return(w, expected_returns)
        vol = np.sqrt(portfolio_variance(w, cov_matrix))
        
        returns.append(ret)
        volatilities.append(vol)
    
    # Find efficient frontier (max return for each volatility level)
    returns = np.array(returns)
    volatilities = np.array(volatilities)
    
    # Sort by volatility
    sorted_idx = np.argsort(volatilities)
    volatilities = volatilities[sorted_idx]
    returns = returns[sorted_idx]
    
    # Keep only efficient points (higher return for same or lower volatility)
    efficient_returns = []
    efficient_vols = []
    max_return = -np.inf
    
    for vol, ret in zip(volatilities, returns):
        if ret > max_return:
            max_return = ret
            efficient_vols.append(vol)
            efficient_returns.append(ret)
    
    return np.array(efficient_returns), np.array(efficient_vols)


def equal_weight_portfolio(n_assets):
    """
    Generate equal-weighted portfolio.
    
    Parameters:
    -----------
    n_assets : int
        Number of assets
    
    Returns:
    --------
    array
        Equal weights for each asset
    """
    return np.ones(n_assets) / n_assets


def minimum_variance_portfolio(cov_matrix):
    """
    Find minimum variance portfolio weights.
    
    Parameters:
    -----------
    cov_matrix : array-like
        Covariance matrix of asset returns
    
    Returns:
    --------
    dict
        Dictionary containing weights and portfolio volatility
    """
    n_assets = len(cov_matrix)
    
    # Inverse of covariance matrix
    cov_inv = np.linalg.inv(cov_matrix)
    
    # Minimum variance weights
    ones = np.ones(n_assets)
    weights = np.dot(cov_inv, ones)
    weights = weights / np.sum(weights)
    
    # Portfolio volatility
    port_variance = portfolio_variance(weights, cov_matrix)
    port_std = np.sqrt(port_variance)
    
    return {
        'weights': weights,
        'volatility': port_std
    }
