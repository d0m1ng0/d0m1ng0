## Hi there 👋
### Cadet at 42, Paris

---

## 🚀 Projects

### 📊 Quantum Finance (QF)
A comprehensive Python library for quantitative finance, providing foundational tools for:
- **Option Pricing**: Black-Scholes model, Monte Carlo simulations (European, Asian, Barrier options)
- **Portfolio Optimization**: Markowitz mean-variance optimization, efficient frontier
- **Risk Management**: VaR, CVaR, maximum drawdown, beta, alpha, and more

[📖 View QF Documentation](./qf/README.md)

#### Quick Example
```python
from qf.models import black_scholes

# Price a European call option
call_price = black_scholes.call_option(
    S=100, K=105, T=1.0, r=0.05, sigma=0.2
)
print(f"Call Option Price: ${call_price:.2f}")
```

---

<!--
**d0m1ng0/d0m1ng0** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.
-->
