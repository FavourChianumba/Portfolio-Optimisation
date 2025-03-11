import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
import os
from datetime import datetime
from config.settings import settings
from utils.logger import setup_logger

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Setup directories
data_processed_dir = project_root / settings.DATA_PROCESSED_DIR
dashboard_dir = project_root / "dashboard"
dashboard_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)

class PortfolioDashboard:
    """Generate dashboard visualizations and exports for Tableau"""
    
    def __init__(self):
        """Initialize dashboard generator"""
        self.optimization_dir = data_processed_dir / "optimization"
        self.backtest_dir = data_processed_dir / "backtest"
        self.monte_carlo_dir = data_processed_dir / "monte_carlo"
        self.risk_dir = data_processed_dir / "risk_metrics"
        
    def generate_efficient_frontier_data(self):
        """Prepare efficient frontier data for dashboard"""
        try:
            logger.info("Preparing efficient frontier data for dashboard...")
            
            # Load efficient frontier data
            ef_path = self.optimization_dir / "efficient_frontier.csv"
            ef_curve_path = self.optimization_dir / "efficient_frontier_curve.csv"
            
            if not ef_path.exists() and not ef_curve_path.exists():
                logger.warning("No efficient frontier data found")
                return None
            
            # Combine data from random portfolios and optimized curve
            if ef_path.exists():
                ef_data = pd.read_csv(ef_path)
                
                # Load optimal portfolios
                max_sharpe_path = self.optimization_dir / "max_sharpe_optimized.json"
                min_vol_path = self.optimization_dir / "min_vol_optimized.json"
                
                if max_sharpe_path.exists():
                    with open(max_sharpe_path, 'r') as f:
                        max_sharpe = json.load(f)
                else:
                    max_sharpe = None
                
                if min_vol_path.exists():
                    with open(min_vol_path, 'r') as f:
                        min_vol = json.load(f)
                else:
                    min_vol = None
                
                # Create a CSV file with optimal portfolio markers
                optimal_portfolios = []
                
                if max_sharpe:
                    optimal_portfolios.append({
                        'Name': 'Max Sharpe',
                        'Return': max_sharpe['Return'],
                        'Volatility': max_sharpe['Volatility'],
                        'Sharpe': max_sharpe['Sharpe']
                    })
                
                if min_vol:
                    optimal_portfolios.append({
                        'Name': 'Min Volatility',
                        'Return': min_vol['Return'],
                        'Volatility': min_vol['Volatility'],
                        'Sharpe': min_vol['Sharpe']
                    })
                
                if optimal_portfolios:
                    optimal_df = pd.DataFrame(optimal_portfolios)
                    optimal_df.to_csv(dashboard_dir / "optimal_portfolios.csv", index=False)
                
                # Generate sample portfolios along the efficient frontier
                if ef_curve_path.exists():
                    ef_curve = pd.read_csv(ef_curve_path)
                    
                    # Combine curve with random portfolios for better visualization
                    # Mark them with different types
                    ef_data['Type'] = 'Random Portfolio'
                    ef_curve['Type'] = 'Efficient Frontier'
                    
                    # Select a subset of random portfolios for cleaner visualization
                    sample_size = min(1000, len(ef_data))
                    ef_data_sample = ef_data.sample(sample_size)
                    
                    # Combine datasets
                    combined_ef = pd.concat([ef_data_sample, ef_curve])
                    combined_ef.to_csv(dashboard_dir / "efficient_frontier_viz.csv", index=False)
                else:
                    # Add type column if only random portfolios exist
                    ef_data['Type'] = 'Random Portfolio'
                    ef_data.to_csv(dashboard_dir / "efficient_frontier_viz.csv", index=False)
            
            # Load asset weights for optimal portfolios
            weights_data = []
            
            # Check for max Sharpe weights
            max_sharpe_weights_path = self.optimization_dir / "max_sharpe_optimized_weights.csv"
            if max_sharpe_weights_path.exists():
                max_sharpe_weights = pd.read_csv(max_sharpe_weights_path)
                max_sharpe_weights['Portfolio'] = 'Max Sharpe'
                weights_data.append(max_sharpe_weights)
            
            # Check for min vol weights
            min_vol_weights_path = self.optimization_dir / "min_vol_optimized_weights.csv"
            if min_vol_weights_path.exists():
                min_vol_weights = pd.read_csv(min_vol_weights_path)
                min_vol_weights['Portfolio'] = 'Min Volatility'
                weights_data.append(min_vol_weights)
            
            # Check for risk parity weights
            risk_parity_path = self.optimization_dir / "risk_parity_data.csv"
            if risk_parity_path.exists():
                risk_parity_weights = pd.read_csv(risk_parity_path)
                # Filter to just weights (not risk contributions)
                if 'Weight' in risk_parity_weights.columns:
                    risk_parity_weights = risk_parity_weights[['Asset', 'Weight']]
                    risk_parity_weights['Portfolio'] = 'Risk Parity'
                    weights_data.append(risk_parity_weights)
            
            # Create equal weight portfolio for comparison
            if weights_data:
                # Get assets from first portfolio
                assets = weights_data[0]['Asset'].unique()
                equal_weights = pd.DataFrame({
                    'Asset': assets,
                    'Weight': 1.0 / len(assets),
                    'Portfolio': 'Equal Weight'
                })
                weights_data.append(equal_weights)
            
            # Combine all weights
            if weights_data:
                all_weights = pd.concat(weights_data)
                all_weights.to_csv(dashboard_dir / "portfolio_weights.csv", index=False)
            
            logger.info("Efficient frontier data prepared for dashboard")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing efficient frontier data: {str(e)}", exc_info=True)
            return False
    
    def generate_backtest_comparison(self):
        """Prepare backtest comparison data for dashboard"""
        try:
            logger.info("Preparing backtest comparison data for dashboard...")
            
            # Look for comparison backtest results
            comparison_files = list(self.backtest_dir.glob("comparison_*_results.json"))
            
            if not comparison_files:
                logger.warning("No backtest comparison data found")
                return None
            
            # Use the most recent comparison file
            latest_comparison = max(comparison_files, key=os.path.getctime)
            
            with open(latest_comparison, 'r') as f:
                comparison_results = json.load(f)
            
            # Extract performance metrics
            performance_data = []
            for strategy_name, results in comparison_results.items():
                performance_data.append({
                    'Strategy': strategy_name,
                    'Total Return': results['total_return'],
                    'Annualized Return': results['annualized_return'],
                    'Volatility': results['annualized_volatility'],
                    'Sharpe Ratio': results['sharpe_ratio'],
                    'Max Drawdown': results['max_drawdown'],
                    'Final Value': results['final_value']
                })
            
            performance_df = pd.DataFrame(performance_data)
            performance_df.to_csv(dashboard_dir / "backtest_performance.csv", index=False)
            
            # Check for portfolio values time series
            comparison_id = latest_comparison.stem.replace("_results", "")
            values_path = self.backtest_dir / f"{comparison_id}_portfolio_values.csv"
            
            if values_path.exists():
                values_df = pd.read_csv(values_path, index_col=0, parse_dates=True)
                values_df.to_csv(dashboard_dir / "backtest_values.csv")
                
                # Calculate growth of $1 for relative comparison
                growth_df = values_df.div(values_df.iloc[0]) 
                growth_df.to_csv(dashboard_dir / "backtest_growth.csv")
            
            # Check for drawdowns
            drawdown_files = list(self.backtest_dir.glob("*_drawdowns.csv"))
            
            if drawdown_files:
                # Consolidate drawdowns from various backtests
                drawdown_data = []
                
                for file in drawdown_files:
                    # Extract strategy name from filename
                    strategy_name = file.stem.split('_')[0]
                    
                    if 'static' in strategy_name:
                        rebalance = file.stem.split('_')[1]
                        strategy_name = f"Static ({rebalance})"
                    elif 'dynamic' in strategy_name:
                        strategy_name = "Dynamic Allocation"
                    elif 'macro' in strategy_name:
                        strategy_name = "Macro Factors"
                    
                    # Load drawdown data
                    df = pd.read_csv(file, index_col=0, parse_dates=True)
                    
                    # Rename column
                    df.columns = [strategy_name]
                    
                    drawdown_data.append(df)
                
                # Combine all drawdowns
                if drawdown_data:
                    combined_drawdowns = pd.concat(drawdown_data, axis=1)
                    combined_drawdowns.to_csv(dashboard_dir / "backtest_drawdowns.csv")
            
            logger.info("Backtest comparison data prepared for dashboard")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing backtest comparison data: {str(e)}", exc_info=True)
            return False
        
    def generate_risk_metrics_dashboard(self):
        """Prepare risk metrics data for dashboard"""
        try:
            logger.info("Preparing risk metrics data for dashboard...")
            
            # Check for asset risk metrics
            risk_metrics_path = self.risk_dir / "asset_risk_metrics.csv"
            
            if not risk_metrics_path.exists():
                logger.warning("No risk metrics data found")
                return None
            
            # Load risk metrics
            risk_metrics = pd.read_csv(risk_metrics_path)
            
            # Create correlation matrix visualization data
            correlation_path = data_processed_dir / "correlation_matrix.csv"
            
            if correlation_path.exists():
                corr_matrix = pd.read_csv(correlation_path, index_col=0)
                
                # Convert correlation matrix to long format for heatmap
                corr_long = corr_matrix.reset_index().melt(id_vars='index', 
                                                        var_name='Asset2', 
                                                        value_name='Correlation')
                corr_long.columns = ['Asset1', 'Asset2', 'Correlation']
                
                corr_long.to_csv(dashboard_dir / "correlation_heatmap.csv", index=False)
            
            # Process volatility metrics specifically for risk dashboard
            vol_path = self.risk_dir / "volatility_metrics.csv"
            if vol_path.exists():
                vol_metrics = pd.read_csv(vol_path)
                
                # Create a melted version for easier visualization
                vol_metrics_long = vol_metrics.reset_index().melt(id_vars='index', 
                                                              var_name='Metric', 
                                                              value_name='Value')
                vol_metrics_long.columns = ['Asset', 'Metric', 'Value']
                
                # Filter to relevant metrics for dashboard
                key_metrics = ['annual_volatility', 'upside_volatility', 'downside_volatility',
                              'volatility_ratio', 'volatility_of_volatility']
                
                vol_metrics_filtered = vol_metrics_long[vol_metrics_long['Metric'].isin(key_metrics)]
                vol_metrics_filtered.to_csv(dashboard_dir / "volatility_metrics_viz.csv", index=False)
            
            # Process tail risk metrics
            tail_path = self.risk_dir / "tail_risk_metrics.csv"
            if tail_path.exists():
                tail_metrics = pd.read_csv(tail_path)
                
                # Create a melted version for visualization
                tail_metrics_long = tail_metrics.reset_index().melt(id_vars='index', 
                                                               var_name='Metric', 
                                                               value_name='Value')
                tail_metrics_long.columns = ['Asset', 'Metric', 'Value']
                
                # Filter to relevant metrics for dashboard
                key_metrics = ['var_95_daily', 'cvar_95_daily', 'skewness', 'excess_kurtosis']
                
                tail_metrics_filtered = tail_metrics_long[tail_metrics_long['Metric'].isin(key_metrics)]
                tail_metrics_filtered.to_csv(dashboard_dir / "tail_risk_metrics_viz.csv", index=False)
            
            # Process drawdown information
            drawdown_files = list(self.risk_dir.glob("*_drawdowns.csv"))
            
            if drawdown_files:
                # Create a summary of max drawdowns
                drawdown_summary = []
                
                for file in drawdown_files:
                    asset = file.stem.replace("_drawdowns", "")
                    
                    # Load drawdown data
                    df = pd.read_csv(file, index_col=0, parse_dates=True)
                    
                    # Find max drawdown and its date
                    max_dd = df.min().iloc[0]
                    max_dd_date = df.idxmin().iloc[0]
                    
                    drawdown_summary.append({
                        'Asset': asset,
                        'Max Drawdown': max_dd,
                        'Max Drawdown Date': max_dd_date
                    })
                
                drawdown_df = pd.DataFrame(drawdown_summary)
                drawdown_df.to_csv(dashboard_dir / "max_drawdowns.csv", index=False)
            
            # Combine risk and performance metrics
            perf_path = self.risk_dir / "performance_metrics.csv"
            if perf_path.exists() and risk_metrics_path.exists():
                perf_metrics = pd.read_csv(perf_path)
                
                # Join performance and risk metrics
                combined_metrics = risk_metrics.join(perf_metrics, rsuffix='_perf')
                
                # Select key metrics for dashboard
                key_cols = ['annual_volatility', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                          'max_drawdown', 'annualized_return', 'var_95_daily', 'skewness']
                
                available_cols = [col for col in key_cols if col in combined_metrics.columns]
                
                if available_cols:
                    metrics_dashboard = combined_metrics[available_cols]
                    metrics_dashboard.to_csv(dashboard_dir / "key_metrics_dashboard.csv")
            
            logger.info("Risk metrics data prepared for dashboard")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing risk metrics data: {str(e)}", exc_info=True)
            return False
    
    def generate_monte_carlo_dashboard(self):
        """Prepare Monte Carlo simulation data for dashboard"""
        try:
            logger.info("Preparing Monte Carlo simulation data for dashboard...")
            
            # Check for Monte Carlo stats
            mc_stats_files = list(self.monte_carlo_dir.glob("mc_stats_*.json"))
            
            if not mc_stats_files:
                logger.warning("No Monte Carlo simulation data found")
                return None
            
            # Process stats for each simulation method
            mc_stats = []
            
            for file in mc_stats_files:
                # Extract simulation method from filename
                method = file.stem.replace("mc_stats_", "")
                
                with open(file, 'r') as f:
                    stats = json.load(f)
                
                # Add method to stats
                stats['method'] = method
                mc_stats.append(stats)
            
            # Create a DataFrame with simulation stats
            stats_df = pd.DataFrame(mc_stats)
            stats_df.to_csv(dashboard_dir / "monte_carlo_stats.csv", index=False)
            
            # Create percentile comparison data
            percentile_data = []
            
            for stats in mc_stats:
                method = stats['method']
                
                for key, value in stats.items():
                    if key.startswith('percentile_'):
                        percentile = key.replace('percentile_', '')
                        percentile_data.append({
                            'Method': method,
                            'Percentile': int(percentile),
                            'Value': value
                        })
            
            if percentile_data:
                percentile_df = pd.DataFrame(percentile_data)
                percentile_df.to_csv(dashboard_dir / "monte_carlo_percentiles.csv", index=False)
            
            # Check for final value distributions
            final_value_files = list(self.monte_carlo_dir.glob("mc_final_values_*.csv"))
            
            if final_value_files:
                final_values = []
                
                for file in final_value_files:
                    method = file.stem.replace("mc_final_values_", "")
                    
                    df = pd.read_csv(file)
                    df['Method'] = method
                    
                    final_values.append(df)
                
                if final_values:
                    combined_final_values = pd.concat(final_values)
                    combined_final_values.to_csv(dashboard_dir / "monte_carlo_distributions.csv", index=False)
            
            # Process stress test results
            stress_test_path = self.monte_carlo_dir / "stress_test_results.json"
            
            if stress_test_path.exists():
                with open(stress_test_path, 'r') as f:
                    stress_results = json.load(f)
                
                # Extract key metrics for each scenario
                stress_summary = []
                
                for scenario, results in stress_results.items():
                    stress_summary.append({
                        'Scenario': results.get('description', scenario),
                        'Initial Value': results.get('initial_value', 0),
                        'Final Value': results.get('final_value', 0),
                        'Total Return': results.get('total_return', 0),
                        'Max Drawdown': results.get('max_drawdown', 0),
                        'Duration (Days)': results.get('duration_days', 0)
                    })
                
                stress_df = pd.DataFrame(stress_summary)
                stress_df.to_csv(dashboard_dir / "stress_test_summary.csv", index=False)
            
            logger.info("Monte Carlo simulation data prepared for dashboard")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing Monte Carlo data: {str(e)}", exc_info=True)
            return False
    
    def generate_tableau_data_extract(self):
        """Prepare a consolidated data extract for Tableau"""
        try:
            logger.info("Preparing consolidated data extract for Tableau...")
            
            # Create a metadata file with data sources
            data_sources = {}
            
            # Check and add all dashboard files
            for file in dashboard_dir.glob("*.csv"):
                data_sources[file.stem] = {
                    'file': file.name,
                    'description': file.stem.replace('_', ' ').title(),
                    'created': datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                }
            
            # Save metadata
            with open(dashboard_dir / "data_sources.json", 'w') as f:
                json.dump(data_sources, f, indent=4)
            
            # Create a README file for Tableau users
            try:
                import tabulate
                # Original code that uses tabulate
                readme = f"""# Portfolio Optimization Dashboard Data Sources

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    This directory contains all data files required for the Portfolio Optimization Tableau dashboard.

    ## Data Files

    {pd.DataFrame(data_sources).T.to_markdown()}

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

    """
            except ImportError:
                logger.warning("Tabulate package not installed. Using alternative formatting.")
                # Alternative approach without tabulate
                table_str = "\n## Available Data Files\n\n"
                for name, info in data_sources.items():
                    table_str += f"* **{name}**\n"
                    table_str += f"  - File: {info['file']}\n"
                    table_str += f"  - Description: {info['description']}\n"
                    table_str += f"  - Created: {info['created']}\n\n"
                
                readme = f"""# Portfolio Optimization Dashboard Data Sources

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    This directory contains all data files required for the Portfolio Optimization Tableau dashboard.
    {table_str}
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

    """
            
            with open(dashboard_dir / "README.md", 'w') as f:
                f.write(readme)
            
            # Create a Tableau packaged data source if possible
            try:
                # This requires tableauserverclient or similar library
                # For now, just indicate this is where it would happen
                logger.info("Creating Tableau packaged data source would be implemented here")
                pass
            except:
                logger.warning("Could not create Tableau packaged data source (not implemented)")
            
            logger.info("Consolidated data extract prepared for Tableau")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing Tableau data extract: {str(e)}", exc_info=True)
            return False
    
    def create_sample_tableau_dashboard(self):
        """Generate a sample Tableau workbook template (XML-based .twb file)"""
        try:
            logger.info("This method would generate a sample Tableau workbook template")
            logger.info("However, creating a .twb file requires TableauServerClient or similar tools")
            logger.info("For a proper implementation, consider using TabPy or Tableau API")
            
            # In a real implementation, this would create a .twb file with pre-configured
            # visualizations, data connections, and parameters
            
            # Create documentation for manual dashboard setup instead
            dashboard_docs = """# Portfolio Optimization Tableau Dashboard Setup

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
"""
            
            with open(dashboard_dir / "dashboard_setup_guide.md", 'w') as f:
                f.write(dashboard_docs)
            
            logger.info("Dashboard setup guide created")
            return True
            
        except Exception as e:
            logger.error(f"Error creating sample dashboard: {str(e)}", exc_info=True)
            return False
    
    def generate_all_dashboard_data(self):
        """Generate all dashboard data and visualizations"""
        try:
            logger.info("Generating all dashboard data and visualizations...")
            
            # Run all data preparation methods
            self.generate_efficient_frontier_data()
            self.generate_backtest_comparison()
            self.generate_risk_metrics_dashboard()
            self.generate_monte_carlo_dashboard()
            self.generate_tableau_data_extract()
            self.create_sample_tableau_dashboard()
            
            logger.info("All dashboard data generated successfully")
            
            # Generate an index.html file to browse the data
            self._generate_data_browser()
            
            logger.info(f"Dashboard data available at: {dashboard_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {str(e)}", exc_info=True)
            return False
    
    def _generate_data_browser(self):
        """Generate a simple HTML page to browse the dashboard data"""
        try:
            files = list(dashboard_dir.glob("*.csv")) + list(dashboard_dir.glob("*.json")) + list(dashboard_dir.glob("*.md"))
            
            file_list = []
            for file in sorted(files):
                size_kb = file.stat().st_size / 1024
                modified = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                file_list.append({
                    'name': file.name,
                    'size': f"{size_kb:.1f} KB",
                    'modified': modified
                })
            
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Optimization Dashboard Data</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Portfolio Optimization Dashboard Data</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Available Data Files</h2>
        <table>
            <tr>
                <th>Filename</th>
                <th>Size</th>
                <th>Last Modified</th>
            </tr>
            {"".join(f"<tr><td>{f['name']}</td><td>{f['size']}</td><td>{f['modified']}</td></tr>" for f in file_list)}
        </table>
        
        <h2>Next Steps</h2>
        <p>Use these files to create visualizations in Tableau or other BI tools.</p>
        <p>See the dashboard_setup_guide.md file for detailed instructions.</p>
    </div>
</body>
</html>
"""
            
            with open(dashboard_dir / "index.html", 'w') as f:
                f.write(html)
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating data browser: {str(e)}", exc_info=True)
            return False

if __name__ == "__main__":
    dashboard = PortfolioDashboard()
    dashboard.generate_all_dashboard_data()