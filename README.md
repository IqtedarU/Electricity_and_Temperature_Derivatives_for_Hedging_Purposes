# Electricity and Temperature Derivatives for Hedging Risk

This project explores the use of electricity forward contracts and Cooling Degree Day (CDD) weather derivatives to hedge the financial risk faced by a power company with uncertain future demand and electricity prices. The project simulates prices, temperature, and load using stochastic models, and evaluates unhedged and hedged earnings under different strategies.

This is a class project for *Energy Commodities Analytics and Trading* and applies quantitative finance techniques such as:

- Geometric Brownian Motion (GBM) for simulating electricity prices  
- Mean-reverting processes for modeling temperature and load  
- Monte Carlo simulation for portfolio performance  
- Portfolio optimization for selecting hedge positions  
- Value at Risk (VaR) and percentile-based risk metrics  

Presentation:  
[Link to slides](https://docs.google.com/presentation/d/1kmWw5BDqw3V5UMxtpKgjCgu0DhnZt9aOFTpuiXCUXYo/edit#slide=id.g2ad6f70761f_0_923)

## Simulation Setup

- Initial Stock Price (S₀): 61.98  
- Annual Return (μ): 3%  
- Volatility (σ): 21%  
- Simulated Days: 92 (from July 1 to September 30)  
- Simulations: 100,000 Monte Carlo paths  
- Time Steps: Daily  
- Load & Temperature: Simulated with seasonal autocorrelation and covariance structure  
- Contracts:
  - Electricity Forward at $67
  - CDD Contract Strike = 684, Unit = $20 per unit

## Earnings Summary

### Unhedged Earnings

- Mean: $1,275,802  
- Median: $1,323,122  
- 5th Percentile: -$346,449  
- 95% Value at Risk: $1,622,251  
- 1st Percentile: -$1,234,943  
- 99% Value at Risk: $2,510,745  

## Optimal Hedging Strategies

### Electricity Forward Hedge Only

- Optimal Position: 6 contracts  
- Mean Earnings: $666,284  
- Median: $683,707  
- 5th Percentile: -$87,162  
- 1st Percentile: -$520,470  
- 95% Value at Risk: $753,446  
- 99% Value at Risk: $1,186,754  

### CDD Hedge Only

- Optimal Position: 399 contracts  
- Mean Earnings: $1,157,935  
- Median: $1,198,761  
- 5th Percentile: -$359,209  
- 1st Percentile: -$1,148,415  
- 95% Value at Risk: $1,517,144  
- 99% Value at Risk: $2,306,350  

### Combined Hedge (Electricity + CDD)

- Best Position for Electricity Contracts: 6  
- Best Position for CDD Contracts: 78  
- Mean Earnings: $643,242  
- Median: $658,676  
- 5th Percentile: -$104,488  
- 1st Percentile: -$511,535  
- 95% Value at Risk: $747,731  
- 99% Value at Risk: $1,154,778  

## Key Insights

- CDD hedging provides slightly reduced risk and profit compared to unhedged, and may suit risk-neutral companies.  
- Electricity forwards offer better downside protection with lower volatility in earnings.  
- The combined hedge slightly reduces profit but also improves worst-case outcomes and smooths returns.  
- CDD hedging is most effective when temperature risk strongly correlates with load.

## Concepts Applied

- Chebyshev Inequality  
- Correlation and Cholesky Decomposition  
- AR(1) temporal correlation  
- Monte Carlo simulation  
- Quantile and percentile-based risk analysis  
- Hedging under price and demand uncertainty

## Files

- `simulation.py`: Full simulation of price, temperature, load, and earnings  
- `MSF568_2023cFall_AnalyticTermProject.csv`: Historical inputs for means and standard deviations  
- `notebooks/`: Visualization of simulation results and hedging performance

This project demonstrates practical energy risk management using quantitative tools and can be extended to real-world scenarios by calibrating with market and weather data.
