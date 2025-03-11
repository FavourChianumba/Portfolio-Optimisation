# Portfolio Optimization Dashboard Data Sources

    Generated: 2025-03-10 14:33:16

    This directory contains all data files required for the Portfolio Optimization Tableau dashboard.
    
## Available Data Files

* **portfolio_weights**
  - File: portfolio_weights.csv
  - Description: Portfolio Weights
  - Created: 2025-03-10 14:33:11

* **volatility_metrics_viz**
  - File: volatility_metrics_viz.csv
  - Description: Volatility Metrics Viz
  - Created: 2025-03-10 14:33:11

* **backtest_drawdowns**
  - File: backtest_drawdowns.csv
  - Description: Backtest Drawdowns
  - Created: 2025-03-10 14:33:11

* **key_metrics_dashboard**
  - File: key_metrics_dashboard.csv
  - Description: Key Metrics Dashboard
  - Created: 2025-03-10 14:33:13

* **efficient_frontier_viz**
  - File: efficient_frontier_viz.csv
  - Description: Efficient Frontier Viz
  - Created: 2025-03-10 14:33:11

* **backtest_values**
  - File: backtest_values.csv
  - Description: Backtest Values
  - Created: 2025-03-10 14:33:11

* **correlation_heatmap**
  - File: correlation_heatmap.csv
  - Description: Correlation Heatmap
  - Created: 2025-03-10 14:33:11

* **max_drawdowns**
  - File: max_drawdowns.csv
  - Description: Max Drawdowns
  - Created: 2025-03-10 14:33:13

* **monte_carlo_distributions**
  - File: monte_carlo_distributions.csv
  - Description: Monte Carlo Distributions
  - Created: 2025-03-10 14:33:15

* **optimal_portfolios**
  - File: optimal_portfolios.csv
  - Description: Optimal Portfolios
  - Created: 2025-03-10 14:33:10

* **stress_test_summary**
  - File: stress_test_summary.csv
  - Description: Stress Test Summary
  - Created: 2025-03-10 14:33:15

* **tail_risk_metrics_viz**
  - File: tail_risk_metrics_viz.csv
  - Description: Tail Risk Metrics Viz
  - Created: 2025-03-10 14:33:11

* **backtest_growth**
  - File: backtest_growth.csv
  - Description: Backtest Growth
  - Created: 2025-03-10 14:33:11

* **monte_carlo_stats**
  - File: monte_carlo_stats.csv
  - Description: Monte Carlo Stats
  - Created: 2025-03-10 14:33:14

* **monte_carlo_percentiles**
  - File: monte_carlo_percentiles.csv
  - Description: Monte Carlo Percentiles
  - Created: 2025-03-10 14:33:14

* **backtest_performance**
  - File: backtest_performance.csv
  - Description: Backtest Performance
  - Created: 2025-03-10 14:33:11


    ## Connecting in Tableau

    1. Use "Connect to Data" -> "Text File" and select the CSV files you need
    2. Join related tables as needed (e.g., portfolio weights can be joined with performance metrics)
    3. Create calculated fields as needed for interactive features

    ## Key Visualizations

    1. **Efficient Frontier**: Use efficient_frontier_viz.csv with Volatility on x-axis and Return on y-axis
    2. **Portfolio Weights**: Use portfolio_weights.csv with a stacked bar or pie chart
    3. **Backtest Performance**: Use backtest_performance.csv for comparing strategies
    4. **Risk Metrics**: Use key_metrics_dashboard.csv for risk comparison
    5. **Monte Carlo Projections**: Use monte_carlo_distributions.csv for probability distributions

    ## Parameters for Interactivity

    Consider adding these parameters in Tableau:
    - Risk Tolerance (Low/Medium/High)
    - Investment Horizon (Years)
    - Initial Investment Amount
    - Rebalancing Frequency

    