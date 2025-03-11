# Portfolio Optimization Tableau Dashboard Setup

## Dashboard Structure
1. Overview Dashboard
   - Key performance metrics
   - Asset allocation summary
   - Risk-return scatter plot

2. Efficient Frontier Dashboard
   - Interactive efficient frontier visualization
   - Optimal portfolio highlights
   - Weight allocation controls

3. Backtest Analysis Dashboard
   - Strategy comparison
   - Rolling performance metrics
   - Drawdown analysis

4. Risk Analysis Dashboard
   - Risk metrics heatmap
   - Correlation matrix
   - Tail risk visualization

5. Monte Carlo Dashboard
   - Projection distributions
   - Probability of meeting return targets
   - Stress test scenarios

## Implementation Steps
1. Connect to all CSV files in the data directory
2. Create calculated fields for interactivity
3. Set up parameters for user inputs
4. Create individual visualizations
5. Combine into dashboards with actions

## Parameters to Create
- Risk Tolerance: [Low, Medium, High]
- Investment Horizon: [1-30 years]
- Initial Investment: [Dollar amount]
- Rebalance Frequency: [Monthly, Quarterly, Annually]

## Example Calculated Fields
```
// Expected Return based on selected risk tolerance
IF [Risk Tolerance] = 'Low' THEN
    [Min Vol Return]
ELSEIF [Risk Tolerance] = 'Medium' THEN
    [Balanced Return]
ELSE
    [Max Sharpe Return]
END

// Probability of Meeting Target
[Probability Above Target]

// Custom Risk Score
([Volatility] * 0.4) + ([Max Drawdown] * 0.4) + ([Downside Deviation] * 0.2)
```

## Key Visualizations
1. Efficient Frontier: Scatter plot with Volatility (x) vs Return (y)
2. Portfolio Weights: Stacked bar chart or pie chart
3. Performance Comparison: Line chart of cumulative returns
4. Risk Heatmap: Highlight table of risk metrics
5. Monte Carlo: Area chart with percentile bands
