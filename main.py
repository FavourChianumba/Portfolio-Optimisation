import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import project modules
from config.settings import settings
from utils.logger import setup_logger
from src import data_cleaning, data_collection, data_validation
from src import risk_metrics, optimization, monte_carlo, backtesting, dashboard

# Set up main logger
logger = setup_logger("main")

def run_pipeline():
    """Run the complete portfolio optimization pipeline"""
    start_time = time.time()
    
    logger.info("Starting portfolio optimization pipeline")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies first
    if not check_dependencies():
        return {'status': 'failed', 'error': 'Missing required dependencies'}
    
    try:
        # Step 1: Data Collection
        logger.info("Step 1: Data Collection")
        asset_data, fred_data, yahoo_macro = data_collection.collect_data()
        
        # Step 2: Data Cleaning
        logger.info("Step 2: Data Cleaning")
        cleaned_asset_data, returns, monthly_returns = data_cleaning.clean_asset_data(asset_data)
        macro_data, macro_changes = data_cleaning.clean_macro_data()
        
        # Log data characteristics
        logger.info(f"Cleaned {len(cleaned_asset_data.columns)} assets with {len(cleaned_asset_data)} data points")
        logger.info(f"Processed {len(macro_data.columns)} macro factors")
        
        # Step 3: Data Validation
        logger.info("Step 3: Data Validation")
        asset_validation = data_validation.validate_asset_data()
        macro_validation = data_validation.validate_macro_data()
        
        # Check validation results
        if asset_validation and asset_validation.get('passed_validation', False):
            logger.info("Asset data passed validation")
        else:
            logger.warning("Asset data has validation warnings - check validation results")
        
        if macro_validation and macro_validation.get('passed_validation', False):
            logger.info("Macro data passed validation")
        else:
            logger.warning("Macro data has validation warnings - check validation results")
        
        # Step 4: Risk Metrics
        logger.info("Step 4: Risk Metrics Calculation")
        risk_calculator = risk_metrics.RiskMetrics(returns_data=returns, prices_data=cleaned_asset_data)
        risk_metrics_data = risk_calculator.calculate_all_metrics()
        
        # Log key risk findings
        high_vol_assets = []
        for asset, metrics in risk_metrics_data.items():
            if metrics.get('annual_volatility', 0) > 0.25:  # 25% annual volatility threshold
                high_vol_assets.append(asset)
        
        if high_vol_assets:
            logger.info(f"High volatility assets identified: {', '.join(high_vol_assets)}")
        
        # Step 5: Portfolio Optimization
        logger.info("Step 5: Portfolio Optimization")
        optimizer = optimization.PortfolioOptimizer(returns_data=returns)
        
        # First generate the efficient frontier
        ef_results, min_vol_portfolio, max_sharpe_portfolio = optimizer.generate_efficient_frontier()
        
        # Log optimal portfolios
        logger.info(f"Max Sharpe portfolio - Return: {max_sharpe_portfolio['Return']:.4f}, "
                   f"Volatility: {max_sharpe_portfolio['Volatility']:.4f}, "
                   f"Sharpe: {max_sharpe_portfolio['Sharpe']:.4f}")
        
        logger.info(f"Min Volatility portfolio - Return: {min_vol_portfolio['Return']:.4f}, "
                   f"Volatility: {min_vol_portfolio['Volatility']:.4f}, "
                   f"Sharpe: {min_vol_portfolio['Sharpe']:.4f}")
        
        # Run optimized calculations
        max_sharpe_optimized = optimizer.optimize_sharpe_ratio()
        min_vol_optimized = optimizer.optimize_minimum_volatility()
        risk_parity_portfolio = optimizer.optimize_risk_parity()
        
        # Generate efficient frontier curve
        ef_curve, ef_portfolios = optimizer.generate_efficient_frontier_curve()

        # Optional: Run Factor-Based Optimization 
        try:
            logger.info("Step 5b: Factor-Based Optimization")
            from src.factor_optimization import FactorOptimizer
            factor_opt = FactorOptimizer(returns_data=returns)
            factor_portfolio = factor_opt.optimize_factor_portfolio(objective='sharpe')
            logger.info(f"Factor-Based portfolio - Return: {factor_portfolio['return']:.4f}, "
                    f"Volatility: {factor_portfolio['volatility']:.4f}, "
                    f"Sharpe: {factor_portfolio['sharpe_ratio']:.4f}")
        except Exception as e:
            logger.warning(f"Factor-Based Optimization skipped: {str(e)}")

        # Optional: Run Black-Litterman Optimization
        try:
            logger.info("Step 5c: Black-Litterman Optimization")
            from src.black_litterman import BlackLittermanOptimizer
            bl_opt = BlackLittermanOptimizer(returns_data=returns)
            bl_portfolio = bl_opt.optimize_portfolio(objective='sharpe')
            logger.info(f"Black-Litterman portfolio - Return: {bl_portfolio['return']:.4f}, "
                    f"Volatility: {bl_portfolio['volatility']:.4f}, "
                    f"Sharpe: {bl_portfolio['sharpe_ratio']:.4f}")
        except Exception as e:
            logger.warning(f"Black-Litterman Optimization skipped: {str(e)}")
                
        # Step 6: Monte Carlo Simulation
        logger.info("Step 6: Monte Carlo Simulation")
        
        # Use max Sharpe weights for simulation
        max_sharpe_weights = max_sharpe_optimized.get('Weights', {})
        simulator = monte_carlo.MonteCarloSimulator(returns_data=returns, portfolio_weights=max_sharpe_weights)
        
        # Run simulations with different methods
        param_sim, param_stats = simulator.run_simulation(return_method='parametric')
        hist_sim, hist_stats = simulator.run_simulation(return_method='historical')
        bootstrap_sim, bootstrap_stats = simulator.run_simulation(return_method='bootstrap')
        
        # Log simulation results
        logger.info(f"Parametric simulation - Mean final value: ${param_stats['mean']:,.2f}, "
                   f"Median: ${param_stats['median']:,.2f}")
        
        logger.info(f"Historical simulation - Mean final value: ${hist_stats['mean']:,.2f}, "
                   f"Median: ${hist_stats['median']:,.2f}")
        
        logger.info(f"Bootstrap simulation - Mean final value: ${bootstrap_stats['mean']:,.2f}, "
                   f"Median: ${bootstrap_stats['median']:,.2f}")
        
        # Run stress tests
        stress_results = simulator.run_stress_test()
        
        # Log worst stress test scenario
        worst_scenario = min(stress_results.items(), key=lambda x: x[1]['total_return'])
        logger.info(f"Worst stress scenario: {worst_scenario[0]} - "
                   f"Return: {worst_scenario[1]['total_return']:.2%}, "
                   f"Max Drawdown: {worst_scenario[1]['max_drawdown']:.2%}")
        
        # Step 7: Portfolio Backtesting
        logger.info("Step 7: Portfolio Backtesting")
        backtester = backtesting.PortfolioBacktester(prices_data=cleaned_asset_data, returns_data=returns)
        
        # Define strategies for comparison
        strategies = [
            {
                'name': 'Equal Weight',
                'weights': {asset: 1.0/len(cleaned_asset_data.columns) for asset in cleaned_asset_data.columns},
                'rebalance_frequency': 'M'
            },
            {
                'name': 'Max Sharpe',
                'weights': max_sharpe_optimized.get('Weights', {}),
                'rebalance_frequency': 'M'
            },
            {
                'name': 'Min Volatility',
                'weights': min_vol_optimized.get('Weights', {}),
                'rebalance_frequency': 'M'
            },
            {
                'name': 'Risk Parity',
                'weights': risk_parity_portfolio.get('Weights', {}),
                'rebalance_frequency': 'M'
            }
        ]
        
        # Run backtest comparison
        backtest_results, portfolio_values, portfolio_returns = backtester.backtest_strategy_comparison(
            strategies=strategies,
            start_date=cleaned_asset_data.index[0],
            end_date=cleaned_asset_data.index[-1]
        )
        
        # Log backtest results
        for strategy, result in backtest_results.items():
            logger.info(f"{strategy} backtest - Return: {result['annualized_return']:.2%}, "
                      f"Volatility: {result['annualized_volatility']:.2%}, "
                      f"Sharpe: {result['sharpe_ratio']:.2f}")
        
        # Run macro-based backtest if macro data is available
        try:
            if macro_data is not None:
                logger.info("Running macro backtesting...")
                macro_backtest_result, macro_values, macro_returns, macro_weights = backtester.backtest_with_macro_factors(
                    weights=max_sharpe_optimized.get('Weights', {}),
                    start_date=cleaned_asset_data.index[0],
                    end_date=cleaned_asset_data.index[-1]
                )
                
                logger.info(f"Macro-adjusted backtest - Return: {macro_backtest_result['annualized_return']:.2%}, "
                        f"Volatility: {macro_backtest_result['annualized_volatility']:.2%}, "
                        f"Sharpe: {macro_backtest_result['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.warning(f"Macro factor backtesting skipped: {str(e)}")

        # Step 8: Dashboard Generation
        logger.info("Step 8: Dashboard Generation")
        dashboard_generator = dashboard.PortfolioDashboard()
        dashboard_generator.generate_all_dashboard_data()
        
        # Final report generation
        logger.info("Generating final performance report")
        
        # Create a summary of the best performing strategies
        best_strategy = max(backtest_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        logger.info(f"Best performing strategy: {best_strategy[0]} with Sharpe ratio {best_strategy[1]['sharpe_ratio']:.2f}")
        
        # Compare optimized portfolio to market benchmark (if SPY is included)
        if 'SPY' in cleaned_asset_data.columns:
            spy_return = (cleaned_asset_data['SPY'].iloc[-1] / cleaned_asset_data['SPY'].iloc[0] - 1)
            max_sharpe_backtest = backtest_results.get('Max Sharpe', {})
            if max_sharpe_backtest:
                max_sharpe_return = max_sharpe_backtest.get('total_return', 0)
                outperformance = max_sharpe_return - spy_return
                logger.info(f"Optimized portfolio vs. SPY: {outperformance:.2%} outperformance")
        
        # Calculate and log risk-adjusted performance metrics
        for strategy, result in backtest_results.items():
            calmar = result.get('annualized_return', 0) / abs(result.get('max_drawdown', 1))
            logger.info(f"{strategy} - Calmar Ratio: {calmar:.2f}")
        
        # Pipeline completed
        elapsed_time = time.time() - start_time
        logger.info(f"Portfolio optimization pipeline completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"All results saved to {project_root}/data/processed and {project_root}/dashboard directories")
        
        # Return success status and key results
        return {
            'status': 'success',
            'elapsed_time': elapsed_time,
            'best_strategy': best_strategy[0],
            'max_sharpe': {
                'return': max_sharpe_optimized.get('Return', 0),
                'volatility': max_sharpe_optimized.get('Volatility', 0),
                'sharpe': max_sharpe_optimized.get('Sharpe', 0),
                'weights': max_sharpe_optimized.get('Weights', {})
            },
            'monte_carlo': {
                'mean_final_value': param_stats['mean'],
                'median_final_value': param_stats['median'],
                'p95_final_value': param_stats.get('percentile_95', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline execution stopped after {elapsed_time:.2f} seconds")
        return {
            'status': 'failed',
            'error': str(e),
            'elapsed_time': elapsed_time
        }

def check_dependencies():
    """Check for required dependencies and warn about missing ones"""
    missing_dependencies = []
    optional_dependencies = []
    
    # Core dependencies - required
    try:
        import pandas
        import numpy
        import matplotlib
    except ImportError as e:
        missing_dependencies.append(str(e).split("'")[1])
    
    # Optional dependencies - warn but continue
    try:
        import statsmodels
    except ImportError:
        optional_dependencies.append("statsmodels")
    
    try:
        import tabulate
    except ImportError:
        optional_dependencies.append("tabulate")
    
    # Log results
    if missing_dependencies:
        logger.error(f"Missing required dependencies: {', '.join(missing_dependencies)}")
        logger.error("Please install them with: pip install " + " ".join(missing_dependencies))
        return False
    
    if optional_dependencies:
        logger.warning(f"Missing optional dependencies: {', '.join(optional_dependencies)}")
        logger.warning("For full functionality, install them with: pip install " + " ".join(optional_dependencies))
    
    return True

if __name__ == "__main__":
    results = run_pipeline()
    
    if results.get('status') == 'success':
        print("\n==== Portfolio Optimization Completed Successfully ====")
        print(f"Elapsed time: {results['elapsed_time']:.2f} seconds")
        print(f"Best strategy: {results['best_strategy']}")
        
        # Print Max Sharpe portfolio details
        print("\nMax Sharpe Portfolio:")
        print(f"  Return: {results['max_sharpe']['return']:.2%}")
        print(f"  Volatility: {results['max_sharpe']['volatility']:.2%}")
        print(f"  Sharpe Ratio: {results['max_sharpe']['sharpe']:.2f}")
        
        # Print top 5 allocations
        print("\nTop 5 allocations:")
        sorted_weights = sorted(results['max_sharpe']['weights'].items(), key=lambda x: x[1], reverse=True)[:5]
        for asset, weight in sorted_weights:
            print(f"  {asset}: {weight:.2%}")
        
        # Print Monte Carlo projections
        print("\nMonte Carlo Projections (from $100,000 initial investment):")
        print(f"  Mean final value: ${results['monte_carlo']['mean_final_value']:,.2f}")
        print(f"  Median final value: ${results['monte_carlo']['median_final_value']:,.2f}")
        print(f"  95th percentile: ${results['monte_carlo']['p95_final_value']:,.2f}")
        
        print("\nComplete results available in the dashboard directory.")
    else:
        print("\n==== Portfolio Optimization Failed ====")
        print(f"Error: {results.get('error', 'Unknown error')}")
        print(f"Check logs for details.")