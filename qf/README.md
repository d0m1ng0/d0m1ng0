# Quantum Finance (QF)

A comprehensive Python library for quantitative finance, providing foundational tools for option pricing, portfolio optimization, and risk management.

## Features

### 📊 Option Pricing Models
- **Black-Scholes Model**: Classical European option pricing with Greeks calculation
- **Monte Carlo Simulation**: 
  - European options (calls and puts)
  - Asian options (average price)
  - Barrier options (up-and-out, up-and-in, down-and-out, down-and-in)

### 📈 Portfolio Management
- **Portfolio Optimization**:
  - Markowitz mean-variance optimization
  - Minimum variance portfolio
  - Efficient frontier calculation
  - Equal-weight portfolio
- **Portfolio Metrics**:
  - Expected return and variance
  - Sharpe ratio

### ⚠️ Risk Management
- **Value at Risk (VaR)**:
  - Historical method
  - Parametric method
- **Conditional Value at Risk (CVaR)** / Expected Shortfall
- **Maximum Drawdown** analysis
- **Market Risk Metrics**:
  - Beta (systematic risk)
  - Jensen's Alpha
- **Performance Metrics**:
  - Tracking error
  - Information ratio
  - Sortino ratio
- **Stress Testing** scenarios

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Black-Scholes Option Pricing

```python
from qf.models import black_scholes

# Option parameters
S = 100    # Current stock price
K = 105    # Strike price
T = 1.0    # Time to maturity (years)
r = 0.05   # Risk-free rate
sigma = 0.2  # Volatility

# Price call option
call_price = black_scholes.call_option(S, K, T, r, sigma)
print(f"Call Price: ${call_price:.2f}")

# Calculate Greeks
greeks = black_scholes.greeks(S, K, T, r, sigma, option_type='call')
print(f"Delta: {greeks['delta']:.4f}")
```

### Monte Carlo Simulation

```python
from qf.models import monte_carlo

# Price European call option
result = monte_carlo.european_option(
    S=100, K=105, T=1.0, r=0.05, sigma=0.2,
    option_type='call',
    n_simulations=100000
)

print(f"Option Price: ${result['price']:.2f}")
print(f"Std Error: ${result['std_error']:.2f}")
```

### Portfolio Optimization

```python
import numpy as np
from qf.utils import portfolio

# Asset expected returns and covariance
expected_returns = np.array([0.12, 0.10, 0.08])
cov_matrix = np.array([
    [0.04, 0.01, 0.005],
    [0.01, 0.03, 0.008],
    [0.005, 0.008, 0.02]
])

# Find optimal portfolio (max Sharpe ratio)
optimal = portfolio.markowitz_optimization(
    expected_returns, 
    cov_matrix, 
    risk_free_rate=0.03
)

print(f"Optimal Weights: {optimal['weights']}")
print(f"Expected Return: {optimal['return']*100:.2f}%")
print(f"Sharpe Ratio: {optimal['sharpe_ratio']:.4f}")
```

### Risk Management

```python
import numpy as np
from qf.utils import risk

# Sample returns
returns = np.random.normal(0.0005, 0.02, 252)

# Calculate Value at Risk
var_95 = risk.value_at_risk(returns, confidence_level=0.95)
print(f"95% VaR: {var_95*100:.2f}%")

# Calculate Conditional VaR
cvar_95 = risk.conditional_value_at_risk(returns, confidence_level=0.95)
print(f"95% CVaR: {cvar_95*100:.2f}%")

# Maximum Drawdown
prices = [100 * (1 + r) for r in np.cumsum(returns)]
mdd = risk.maximum_drawdown(prices)
print(f"Max Drawdown: {mdd['max_drawdown']*100:.2f}%")
```

## Examples

Run the included examples to see the library in action:

```bash
# Black-Scholes example
python qf/examples/black_scholes_example.py

# Monte Carlo simulation example
python qf/examples/monte_carlo_example.py

# Portfolio optimization example
python qf/examples/portfolio_example.py

# Risk management example
python qf/examples/risk_example.py
```

## Module Structure

```
qf/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── black_scholes.py    # Black-Scholes model
│   └── monte_carlo.py       # Monte Carlo simulations
├── utils/
│   ├── __init__.py
│   ├── portfolio.py         # Portfolio optimization
│   └── risk.py              # Risk management tools
└── examples/
    ├── black_scholes_example.py
    ├── monte_carlo_example.py
    ├── portfolio_example.py
    └── risk_example.py
```

## Mathematical Background

### Black-Scholes Formula

For a European call option:

```
C = S₀N(d₁) - Ke⁻ʳᵀN(d₂)
```

where:
- `d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)`
- `d₂ = d₁ - σ√T`
- `N(·)` is the cumulative standard normal distribution

### Monte Carlo Method

Simulates stock price paths using geometric Brownian motion:

```
Sᴛ = S₀ exp((r - σ²/2)T + σ√T·Z)
```

where `Z ~ N(0,1)` is a standard normal random variable.

### Portfolio Optimization

Finds weights `w` that maximize the Sharpe ratio:

```
max SR = (wᵀμ - rₓ) / √(wᵀΣw)
```

where:
- `μ` is the vector of expected returns
- `Σ` is the covariance matrix
- `rₓ` is the risk-free rate

## Dependencies

- NumPy: Numerical computing
- SciPy: Scientific computing and statistical functions

## License

This project is open source and available for educational and research purposes.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Author

d0m1ng0 - Cadet at 42, Paris

## Acknowledgments

Built with foundational concepts from:
- Hull, J. C. (2018). Options, Futures, and Other Derivatives
- Markowitz, H. (1952). Portfolio Selection
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities
