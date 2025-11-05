"""
Monte Carlo Simulation for Option Pricing

Numerical method for pricing options and derivatives.
"""

import numpy as np


def european_option(S, K, T, r, sigma, option_type='call', n_simulations=100000, seed=None):
    """
    Price European option using Monte Carlo simulation.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility (annual)
    option_type : str
        'call' or 'put'
    n_simulations : int
        Number of Monte Carlo simulations
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing option price, standard error, and confidence interval
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random price paths
    Z = np.random.standard_normal(n_simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:  # put
        payoffs = np.maximum(K - ST, 0)
    
    # Discount to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n_simulations)
    
    # 95% confidence interval
    confidence_interval = (
        option_price - 1.96 * std_error,
        option_price + 1.96 * std_error
    )
    
    return {
        'price': option_price,
        'std_error': std_error,
        'confidence_interval': confidence_interval
    }


def asian_option(S, K, T, r, sigma, option_type='call', n_simulations=100000, 
                 n_steps=252, seed=None):
    """
    Price Asian (average) option using Monte Carlo simulation.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility (annual)
    option_type : str
        'call' or 'put'
    n_simulations : int
        Number of Monte Carlo simulations
    n_steps : int
        Number of time steps for averaging
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing option price and standard error
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    
    # Initialize price paths
    prices = np.zeros((n_simulations, n_steps + 1))
    prices[:, 0] = S
    
    # Generate price paths
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_simulations)
        prices[:, t] = prices[:, t-1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        )
    
    # Calculate average prices
    avg_prices = np.mean(prices, axis=1)
    
    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(avg_prices - K, 0)
    else:  # put
        payoffs = np.maximum(K - avg_prices, 0)
    
    # Discount to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n_simulations)
    
    return {
        'price': option_price,
        'std_error': std_error
    }


def barrier_option(S, K, B, T, r, sigma, barrier_type='up-and-out', 
                   option_type='call', n_simulations=100000, n_steps=252, seed=None):
    """
    Price barrier option using Monte Carlo simulation.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    B : float
        Barrier level
    T : float
        Time to maturity (years)
    r : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility (annual)
    barrier_type : str
        'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
    option_type : str
        'call' or 'put'
    n_simulations : int
        Number of Monte Carlo simulations
    n_steps : int
        Number of time steps
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing option price and standard error
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    
    # Initialize price paths
    prices = np.zeros((n_simulations, n_steps + 1))
    prices[:, 0] = S
    
    # Generate price paths
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_simulations)
        prices[:, t] = prices[:, t-1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        )
    
    # Check barrier conditions
    if 'up' in barrier_type:
        barrier_crossed = np.any(prices >= B, axis=1)
    else:  # down
        barrier_crossed = np.any(prices <= B, axis=1)
    
    # Calculate payoffs at maturity
    if option_type == 'call':
        payoffs = np.maximum(prices[:, -1] - K, 0)
    else:  # put
        payoffs = np.maximum(K - prices[:, -1], 0)
    
    # Apply barrier condition
    if 'out' in barrier_type:
        payoffs[barrier_crossed] = 0  # knocked out
    else:  # in
        payoffs[~barrier_crossed] = 0  # only active if knocked in
    
    # Discount to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n_simulations)
    
    return {
        'price': option_price,
        'std_error': std_error
    }
