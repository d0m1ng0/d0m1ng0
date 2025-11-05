"""
Quantum Finance (QF) - Foundations Module

A comprehensive quantum finance library providing fundamental tools for:
- Option pricing (Black-Scholes, Monte Carlo)
- Portfolio optimization
- Risk management
- Financial derivatives valuation
"""

__version__ = "0.1.0"
__author__ = "d0m1ng0"

from .models import black_scholes, monte_carlo
from .utils import portfolio, risk

__all__ = [
    "black_scholes",
    "monte_carlo",
    "portfolio",
    "risk",
]
