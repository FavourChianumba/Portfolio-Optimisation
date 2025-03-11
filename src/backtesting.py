import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta
from config.settings import settings
from utils.logger import setup_logger
from src.risk_metrics import RiskMetrics

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Setup directories
data_processed_dir = project_root / settings.DATA_PROCESSED_DIR
backtest_dir = data_processed_dir / "backtest"
backtest_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)

class PortfolioBacktester:
    """Backtest portfolio strategies across historical data"""
    
    def __init__(self, prices_data=None, returns_data=None):
        """Initialize with price and returns data"""
        self.prices = prices_data
        self.returns = returns_data
        self.risk_free_rate = settings.RISK_FREE_RATE
        
        # Load data if not provided
        if self.prices is None:
            prices_path = data_processed_dir / "cleaned_asset_prices.csv"
            self.prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        
        if self.returns is None:
            returns_path = data_processed_dir / "daily_returns.csv"
            self.returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        # Initialize risk metrics
        self.risk_metrics = RiskMetrics(returns_data=self.returns, prices_data=self.prices)
    
    def backtest_static_allocation(self, weights, start_date=None, end_date=None, 
                                initial_investment=100000, rebalance_frequency='M', 
                                save_results=True):
        try:
            logger.info(f"Starting backtest of static allocation with {rebalance_frequency} rebalancing...")
            
            # Filter for assets in the allocation that exist in price data
            assets = [asset for asset in list(weights.keys()) if asset in self.prices.columns]
            if not assets:
                logger.error("No assets in weights found in price data")
                return None
                
            weights_array = np.array([weights[asset] for asset in assets])
            
            # Normalize weights to sum to 1
            weights_array = weights_array / np.sum(weights_array)
            
            # Update weights dictionary with normalized values
            weights = {asset: float(weight) for asset, weight in zip(assets, weights_array)}
            
            # Get price data for the specified assets and date range
            portfolio_prices = self.prices[assets].copy()
            
            if start_date:
                portfolio_prices = portfolio_prices[portfolio_prices.index >= start_date]
            
            if end_date:
                portfolio_prices = portfolio_prices[portfolio_prices.index <= end_date]
            
            # Fill any missing values to avoid NaN issues
            portfolio_prices = portfolio_prices.fillna(method='ffill').fillna(method='bfill')
            
            # Ensure we have data
            if len(portfolio_prices) == 0:
                logger.error("No data available for specified date range")
                return None
                
            # Check for NaN values in price data
            if portfolio_prices.isna().any().any():
                logger.warning("NaN values found in price data even after filling")
                # Remove assets with NaN values
                good_assets = portfolio_prices.columns[~portfolio_prices.isna().any()]
                portfolio_prices = portfolio_prices[good_assets]
                
                # Recalculate weights with only good assets
                good_weights = {asset: weights[asset] for asset in good_assets if asset in weights}
                weights_sum = sum(good_weights.values())
                if weights_sum == 0:
                    logger.error("No valid assets with weights remain")
                    return None
                    
                weights = {asset: weight/weights_sum for asset, weight in good_weights.items()}
                assets = list(weights.keys())
                weights_array = np.array([weights[asset] for asset in assets])
            
            # Initialize results
            dates = portfolio_prices.index
            portfolio_values = np.zeros(len(dates))
            asset_units = np.zeros(len(assets))
            asset_values = np.zeros((len(dates), len(assets)))
            
            # Set initial allocation
            portfolio_values[0] = initial_investment
            day0_prices = portfolio_prices.iloc[0].values
            
            for i, asset in enumerate(assets):
                if day0_prices[i] > 0:  # Add check to avoid division by zero
                    asset_units[i] = (initial_investment * weights_array[i]) / day0_prices[i]
                    asset_values[0, i] = asset_units[i] * day0_prices[i]
                else:
                    logger.warning(f"Zero price for {asset} on first day, setting units to 0")
            
            # Determine rebalance dates
            if rebalance_frequency:
                # Convert pandas frequency string to DateOffset
                try:
                    rebalance_dates = pd.date_range(start=dates[0], end=dates[-1], freq=rebalance_frequency)
                    # Ensure dates exist in the data (trading days)
                    rebalance_dates = [date for date in rebalance_dates if date in dates]
                except Exception as e:
                    logger.warning(f"Error creating rebalance dates: {e}. Using monthly rebalancing.")
                    rebalance_dates = pd.date_range(start=dates[0], end=dates[-1], freq='M')
                    rebalance_dates = [date for date in rebalance_dates if date in dates]
            else:
                # No rebalancing (buy-and-hold)
                rebalance_dates = [dates[0]]
            
            # Track the last rebalance date
            last_rebalance = dates[0]
            
            # Run the backtest
            for t in range(1, len(dates)):
                current_date = dates[t]
                current_prices = portfolio_prices.iloc[t].values
                
                # Update asset values based on price changes
                for i in range(len(assets)):
                    if not np.isnan(current_prices[i]) and current_prices[i] > 0:
                        asset_values[t, i] = asset_units[i] * current_prices[i]
                
                # Calculate total portfolio value with error checking
                current_portfolio_value = np.sum(asset_values[t])
                portfolio_values[t] = current_portfolio_value
                
                # Check if portfolio value becomes too small or zero (avoid division issues)
                if current_portfolio_value <= 0.01 * initial_investment:
                    logger.warning(f"Portfolio value dropped to {current_portfolio_value} at {current_date}. Ending backtest.")
                    break
                
                # Check if rebalancing is needed
                if rebalance_frequency and current_date in rebalance_dates:
                    logger.debug(f"Rebalancing on {current_date}")
                    last_rebalance = current_date
                    
                    # Rebalance portfolio with error checking
                    for i, asset in enumerate(assets):
                        if current_prices[i] > 0:  # Avoid division by zero
                            target_value = portfolio_values[t] * weights_array[i]
                            asset_units[i] = target_value / current_prices[i]
                            asset_values[t, i] = asset_units[i] * current_prices[i]
            
            # Calculate portfolio returns with error checking
            portfolio_returns = np.zeros(len(dates))
            for i in range(1, len(dates)):
                if portfolio_values[i-1] > 0:  # Avoid division by zero
                    portfolio_returns[i] = (portfolio_values[i] / portfolio_values[i-1]) - 1
            
            # Convert to Series with error handling
            portfolio_returns_series = pd.Series(portfolio_returns, index=dates)
            portfolio_values_series = pd.Series(portfolio_values, index=dates)
            
            # Create asset values DataFrame
            asset_values_df = pd.DataFrame(asset_values, index=dates, columns=assets)
            
            # Calculate key metrics with robust error handling
            try:
                total_return = (portfolio_values[-1] / initial_investment) - 1
            except:
                total_return = np.nan
                
            # Calculate drawdowns with error handling
            try:
                portfolio_cum_returns = (1 + portfolio_returns_series).cumprod()
                portfolio_cum_max = portfolio_cum_returns.cummax()
                drawdowns = (portfolio_cum_returns / portfolio_cum_max) - 1
                max_drawdown = drawdowns.min()
            except:
                drawdowns = pd.Series(np.zeros(len(dates)), index=dates)
                max_drawdown = 0
            
            # Annualized metrics with error handling
            days = (dates[-1] - dates[0]).days
            years = max(days / 365.25, 0.1)  # Avoid very small year values
            
            try:
                annualized_return = (1 + total_return) ** (1 / years) - 1
            except:
                annualized_return = np.nan
                
            try:
                annualized_vol = portfolio_returns_series.std() * np.sqrt(252)
            except:
                annualized_vol = np.nan
                
            try:
                sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_vol
            except:
                sharpe_ratio = np.nan
            
            # Calculate rolling metrics
            rolling_returns = portfolio_returns_series.rolling(window=252).mean() * 252
            rolling_vol = portfolio_returns_series.rolling(window=252).std() * np.sqrt(252)
            rolling_sharpe = rolling_returns.sub(self.risk_free_rate).div(rolling_vol)
            
            # Prepare results
            backtest_results = {
                'initial_investment': initial_investment,
                'start_date': dates[0].strftime('%Y-%m-%d'),
                'end_date': dates[-1].strftime('%Y-%m-%d'),
                'rebalance_frequency': rebalance_frequency,
                'weights': weights,
                'total_return': float(total_return),
                'annualized_return': float(annualized_return),
                'annualized_volatility': float(annualized_vol),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'final_value': float(portfolio_values[-1])
            }
            
            # Save results
            if save_results:
                # Create a unique ID for this backtest
                backtest_id = f"static_{rebalance_frequency or 'buyhold'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Save backtest parameters and results
                with open(backtest_dir / f"{backtest_id}_results.json", 'w') as f:
                    json.dump(backtest_results, f, indent=4)
                
                # Save time series data
                portfolio_values_series.to_csv(backtest_dir / f"{backtest_id}_portfolio_values.csv", 
                                              header=['Portfolio_Value'])
                
                portfolio_returns_series.to_csv(backtest_dir / f"{backtest_id}_portfolio_returns.csv",
                                              header=['Portfolio_Return'])
                
                drawdowns.to_csv(backtest_dir / f"{backtest_id}_drawdowns.csv", header=['Drawdown'])
                
                asset_values_df.to_csv(backtest_dir / f"{backtest_id}_asset_values.csv")
                
                # Save rolling metrics
                rolling_metrics = pd.DataFrame({
                    'Rolling_Return': rolling_returns,
                    'Rolling_Volatility': rolling_vol,
                    'Rolling_Sharpe': rolling_sharpe
                })
                rolling_metrics.to_csv(backtest_dir / f"{backtest_id}_rolling_metrics.csv")
                
                # Create visualizations
                # 1. Portfolio value over time
                plt.figure(figsize=(12, 8))
                plt.plot(dates, portfolio_values)
                plt.title('Portfolio Value Over Time')
                plt.xlabel('Date')
                plt.ylabel('Value ($)')
                plt.grid(True)
                plt.savefig(backtest_dir / f"{backtest_id}_portfolio_value.png", dpi=300)
                plt.close()
                
                # 2. Asset allocation over time
                plt.figure(figsize=(12, 8))
                asset_allocation = asset_values_df.div(asset_values_df.sum(axis=1), axis=0)
                asset_allocation.plot.area(figsize=(12, 8))
                plt.title('Portfolio Allocation Over Time')
                plt.xlabel('Date')
                plt.ylabel('Allocation (%)')
                plt.grid(True)
                plt.savefig(backtest_dir / f"{backtest_id}_allocation.png", dpi=300)
                plt.close()
                
                # 3. Drawdown chart
                plt.figure(figsize=(12, 8))
                drawdowns.plot(figsize=(12, 8), color='red')
                plt.title('Portfolio Drawdown')
                plt.xlabel('Date')
                plt.ylabel('Drawdown (%)')
                plt.grid(True)
                plt.fill_between(drawdowns.index, 0, drawdowns.values, color='red', alpha=0.3)
                plt.savefig(backtest_dir / f"{backtest_id}_drawdown.png", dpi=300)
                plt.close()
                
                # 4. Rolling metrics
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
                
                rolling_returns.plot(ax=ax1)
                ax1.set_title('Rolling 1-Year Return')
                ax1.set_ylabel('Annualized Return')
                ax1.grid(True)
                
                rolling_vol.plot(ax=ax2)
                ax2.set_title('Rolling 1-Year Volatility')
                ax2.set_ylabel('Annualized Volatility')
                ax2.grid(True)
                
                rolling_sharpe.plot(ax=ax3)
                ax3.set_title('Rolling 1-Year Sharpe Ratio')
                ax3.set_ylabel('Sharpe Ratio')
                ax3.grid(True)
                
                plt.tight_layout()
                plt.savefig(backtest_dir / f"{backtest_id}_rolling_metrics.png", dpi=300)
                plt.close()
            
            logger.info(f"Backtest completed. Total return: {total_return:.2%}, Sharpe ratio: {sharpe_ratio:.2f}")
            return backtest_results, portfolio_values_series, portfolio_returns_series, asset_values_df
        
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}", exc_info=True)
            raise
    
    def backtest_strategy_comparison(self, strategies, start_date=None, end_date=None, 
                                initial_investment=100000, save_results=True):
        """
        Compare multiple portfolio strategies in a single backtest
        
        Parameters:
        - strategies: List of dictionaries, each with 'name', 'weights', and 'rebalance_frequency'
        - start_date: Backtest start date (or None for earliest date)
        - end_date: Backtest end date (or None for latest date)
        - initial_investment: Initial portfolio value
        - save_results: Whether to save backtest results
        """
        try:
            logger.info(f"Starting comparison backtest of {len(strategies)} strategies...")
            
            # Run backtest for each strategy
            results = {}
            portfolio_values = {}
            portfolio_returns = {}
            
            # Counter for successful backtests
            successful_backtests = 0
            
            for strategy in strategies:
                name = strategy['name']
                weights = strategy['weights']
                rebalance_freq = strategy.get('rebalance_frequency', 'M')
                
                logger.info(f"Backtesting strategy: {name}")
                
                # Filter out assets that don't exist in price data
                filtered_weights = {asset: weight for asset, weight in weights.items() 
                                if asset in self.prices.columns}
                
                # Check if we have any valid weights left
                if not filtered_weights:
                    logger.warning(f"Strategy {name} has no valid assets. Skipping.")
                    continue
                    
                # Normalize weights
                total_weight = sum(filtered_weights.values())
                if total_weight <= 0:
                    logger.warning(f"Strategy {name} has zero or negative total weight. Skipping.")
                    continue
                    
                normalized_weights = {asset: weight/total_weight for asset, weight in filtered_weights.items()}
                
                # Run backtest with no saving of individual results
                backtest_result = self.backtest_static_allocation(
                    weights=normalized_weights,
                    start_date=start_date,
                    end_date=end_date,
                    initial_investment=initial_investment,
                    rebalance_frequency=rebalance_freq,
                    save_results=False
                )
                
                # Check if backtest returned valid results
                if backtest_result is None:
                    logger.warning(f"Backtest for {name} failed to return results. Skipping.")
                    continue
                    
                backtest_result, values, returns, _ = backtest_result
                    
                # Validate results before including
                if (backtest_result is not None and 
                    values is not None and 
                    returns is not None and
                    not pd.isna(backtest_result.get('sharpe_ratio', np.nan))):
                    
                    # Store results
                    results[name] = backtest_result
                    portfolio_values[name] = values
                    portfolio_returns[name] = returns
                    successful_backtests += 1
                else:
                    logger.warning(f"Strategy {name} produced invalid results. Skipping.")
                    
            if successful_backtests == 0:
                logger.warning("No strategies produced valid backtest results.")
                return {}, {}, {}
            
            # Save comparison results
            if save_results and results:
                # Create a unique ID for this comparison
                comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Save comparison parameters and results
                with open(backtest_dir / f"{comparison_id}_results.json", 'w') as f:
                    json.dump(results, f, indent=4)
                
                # Combine portfolio values for plotting
                values_df = pd.DataFrame(portfolio_values)
                values_df.to_csv(backtest_dir / f"{comparison_id}_portfolio_values.csv")
                
                # Combine returns for statistical comparison
                returns_df = pd.DataFrame(portfolio_returns)
                returns_df.to_csv(backtest_dir / f"{comparison_id}_portfolio_returns.csv")
                
                # Calculate cumulative performance for each strategy
                cum_returns = pd.DataFrame()
                for name, returns in portfolio_returns.items():
                    cum_returns[name] = (1 + returns).cumprod()
                    
                cum_returns.to_csv(backtest_dir / f"{comparison_id}_cumulative_returns.csv")
                
                # Create comparison visualizations (only if we have valid data)
                if len(cum_returns) > 0:
                    # 1. Cumulative performance
                    plt.figure(figsize=(12, 8))
                    cum_returns.plot(figsize=(12, 8))
                    plt.title('Strategy Comparison - Cumulative Performance')
                    plt.xlabel('Date')
                    plt.ylabel('Growth of $1')
                    plt.grid(True)
                    plt.savefig(backtest_dir / f"{comparison_id}_performance.png", dpi=300)
                    plt.close()
                    
                    # 2. Performance metrics comparison
                    metrics = ['total_return', 'annualized_return', 'annualized_volatility', 
                            'sharpe_ratio', 'max_drawdown']
                    metrics_data = {}
                    
                    for metric in metrics:
                        metrics_data[metric] = {name: result.get(metric, np.nan) for name, result in results.items()}
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df.to_csv(backtest_dir / f"{comparison_id}_metrics.csv")
                    
                    # Create bar charts for key metrics
                    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
                    
                    for i, metric in enumerate(metrics):
                        ax = axes[i]
                        metrics_df[metric].plot(kind='bar', ax=ax)
                        ax.set_title(f'Strategy Comparison - {metric.replace("_", " ").title()}')
                        ax.grid(True, axis='y')
                        
                        # Format percentages
                        if metric != 'sharpe_ratio':
                            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
                    
                    plt.tight_layout()
                    plt.savefig(backtest_dir / f"{comparison_id}_metrics_comparison.png", dpi=300)
                    plt.close()
                    
                    # 3. Rolling metrics comparison
                    # Calculate rolling Sharpe ratios
                    window = 252  # 1 year
                    rolling_sharpes = {}
                    
                    for name in returns_df.columns:
                        # Check for enough data points before calculating
                        if len(returns_df[name].dropna()) >= window:
                            rolling_return = returns_df[name].rolling(window=window).mean() * 252
                            rolling_vol = returns_df[name].rolling(window=window).std() * np.sqrt(252)
                            # Add error checking for division by zero
                            with np.errstate(divide='ignore', invalid='ignore'):
                                rs = rolling_return.sub(self.risk_free_rate).div(rolling_vol)
                                rs = rs.replace([np.inf, -np.inf], np.nan)
                            rolling_sharpes[name] = rs
                    
                    if rolling_sharpes:
                        rolling_sharpes_df = pd.DataFrame(rolling_sharpes)
                        rolling_sharpes_df.to_csv(backtest_dir / f"{comparison_id}_rolling_sharpes.csv")
                        
                        plt.figure(figsize=(12, 8))
                        rolling_sharpes_df.plot(figsize=(12, 8))
                        plt.title('Strategy Comparison - Rolling 1-Year Sharpe Ratio')
                        plt.xlabel('Date')
                        plt.ylabel('Sharpe Ratio')
                        plt.grid(True)
                        plt.savefig(backtest_dir / f"{comparison_id}_rolling_sharpes.png", dpi=300)
                        plt.close()
            
            logger.info(f"Strategy comparison completed for {successful_backtests} strategies.")
            return results, portfolio_values, portfolio_returns
            
        except Exception as e:
            logger.error(f"Strategy comparison failed: {str(e)}", exc_info=True)
            raise
    
    def backtest_dynamic_allocation(self, allocation_function, start_date=None, end_date=None,
                                initial_investment=100000, lookback_window=252, 
                                rebalance_frequency='M', save_results=True):
        """
        Backtest a dynamic asset allocation strategy that changes weights over time
        
        Parameters:
        - allocation_function: Function that takes (historical_returns, current_date) and returns weights dict
        - start_date: Backtest start date (or None for earliest date)
        - end_date: Backtest end date (or None for latest date)
        - initial_investment: Initial portfolio value
        - lookback_window: Number of days to use for historical data in allocation_function
        - rebalance_frequency: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
        - save_results: Whether to save backtest results
        """
        try:
            logger.info(f"Starting dynamic allocation backtest with {rebalance_frequency} rebalancing...")
            
            # Filter date range
            if start_date:
                prices = self.prices[self.prices.index >= start_date].copy()
                returns = self.returns[self.returns.index >= start_date].copy()
            else:
                prices = self.prices.copy()
                returns = self.returns.copy()
            
            if end_date:
                prices = prices[prices.index <= end_date]
                returns = returns[returns.index <= end_date]
            
            # Fill any missing values to avoid NaN issues
            prices = prices.fillna(method='ffill').fillna(method='bfill')
            returns = returns.fillna(method='ffill').fillna(method='bfill')
            
            # Ensure we have enough historical data
            required_history = lookback_window
            
            if len(returns) < required_history:
                logger.error(f"Not enough historical data. Need at least {required_history} days.")
                return None
            
            # Determine start date for the backtest (after lookback window)
            backtest_start_date = returns.index[lookback_window]
            
            # Filter data for the actual backtest period
            backtest_prices = prices[prices.index >= backtest_start_date]
            all_dates = backtest_prices.index
            all_assets = backtest_prices.columns
            
            if len(all_dates) == 0:
                logger.error("No dates available for backtest after applying lookback window")
                return None
            
            # Determine rebalance dates
            try:
                if rebalance_frequency:
                    # Generate rebalance dates
                    rebalance_dates = pd.date_range(start=backtest_start_date, end=all_dates[-1], 
                                                freq=rebalance_frequency)
                    # Ensure dates exist in the data (trading days) by taking the nearest available date
                    rebalance_dates = [all_dates[all_dates.searchsorted(date, side='right') - 1] for date in rebalance_dates 
                                    if date >= all_dates[0] and date <= all_dates[-1]]
                    # Remove duplicates that might occur due to searchsorted
                    rebalance_dates = list(dict.fromkeys(rebalance_dates))
                else:
                    # No rebalancing (buy-and-hold)
                    rebalance_dates = [all_dates[0]]
            except Exception as e:
                logger.warning(f"Error creating rebalance dates: {e}. Using monthly rebalancing.")
                # Fallback to simple first-of-month dates
                rebalance_dates = [all_dates[0]]
                for i in range(1, len(all_dates)):
                    if all_dates[i].month != all_dates[i-1].month:
                        rebalance_dates.append(all_dates[i])
            
            # Initialize results
            portfolio_values = pd.Series(0.0, index=all_dates)
            # Set initial investment
            portfolio_values.iloc[0] = initial_investment
            
            asset_units = pd.Series(0, index=all_assets)
            weights_history = pd.DataFrame(index=rebalance_dates, columns=all_assets, data=0.0)
            
            # Initial allocation
            current_date = all_dates[0]
            lookback_returns = returns[returns.index < current_date].iloc[-lookback_window:]
            
            # Get initial weights from the allocation function
            try:
                initial_weights = allocation_function(lookback_returns, current_date)
            except Exception as e:
                logger.warning(f"Error in allocation function on initial date: {str(e)}")
                # Fallback to equal weights
                initial_weights = {asset: 1.0/len(all_assets) for asset in all_assets}
            
            # Filter for assets in price data and normalize weights
            initial_weights = {asset: weight for asset, weight in initial_weights.items() 
                            if asset in all_assets}
            total_weight = sum(initial_weights.values())
            if total_weight <= 0:
                # Fallback to equal weights
                initial_weights = {asset: 1.0/len(all_assets) for asset in all_assets}
            else:
                # Normalize weights to sum to 1
                initial_weights = {asset: weight/total_weight for asset, weight in initial_weights.items()}
            
            # Record initial weights
            for asset, weight in initial_weights.items():
                weights_history.loc[current_date, asset] = weight
            
            # Calculate initial units with error checking
            for asset, weight in initial_weights.items():
                if asset in backtest_prices.columns:
                    try:
                        price = backtest_prices.loc[current_date, asset]
                        if not pd.isna(price) and price > 0:
                            asset_units[asset] = (initial_investment * weight) / price
                    except Exception as e:
                        logger.warning(f"Error setting initial units for {asset}: {str(e)}")
            
            # Run the backtest
            for i, date in enumerate(all_dates[1:], 1):
                current_date = date
                
                try:
                    # Calculate current portfolio value
                    current_portfolio_value = 0
                    for asset in asset_units.index:
                        if asset in backtest_prices.columns:
                            try:
                                price = backtest_prices.loc[current_date, asset]
                                if not pd.isna(price) and price > 0:
                                    current_portfolio_value += asset_units[asset] * price
                            except KeyError:
                                # Handle case where the date might not be in the price data for this asset
                                logger.debug(f"Price data not available for {asset} on {current_date}")
                    
                    # Update portfolio value
                    if current_portfolio_value > 0:
                        portfolio_values[date] = current_portfolio_value
                    else:
                        # Use previous value if calculation gave invalid result
                        portfolio_values[date] = portfolio_values.iloc[i-1]
                    
                    # Check if rebalancing is needed
                    if date in rebalance_dates:
                        # Get historical data for the allocation function
                        lookback_returns = returns[returns.index < date].iloc[-lookback_window:]
                        
                        # Get new weights
                        try:
                            new_weights = allocation_function(lookback_returns, date)
                            
                            # Filter and normalize weights
                            new_weights = {asset: weight for asset, weight in new_weights.items() 
                                        if asset in all_assets}
                            total_weight = sum(new_weights.values())
                            
                            if total_weight <= 0:
                                logger.warning(f"Invalid weights on {date}. Using previous weights.")
                                # Use weights from last rebalance
                                last_rebalance = max([d for d in rebalance_dates if d < date], default=all_dates[0])
                                new_weights = {asset: weights_history.loc[last_rebalance, asset] 
                                            for asset in all_assets}
                            else:
                                # Normalize weights
                                new_weights = {asset: weight/total_weight for asset, weight in new_weights.items()}
                        except Exception as e:
                            logger.warning(f"Error in allocation function on {date}: {str(e)}")
                            # Use weights from last rebalance
                            last_rebalance = max([d for d in rebalance_dates if d < date], default=all_dates[0])
                            new_weights = {asset: weights_history.loc[last_rebalance, asset] 
                                        for asset in all_assets}
                        
                        # Record weights
                        for asset, weight in new_weights.items():
                            weights_history.loc[date, asset] = weight
                        
                        # Rebalance portfolio
                        for asset, weight in new_weights.items():
                            if asset in backtest_prices.columns:
                                try:
                                    price = backtest_prices.loc[date, asset]
                                    if not pd.isna(price) and price > 0:
                                        asset_units[asset] = (portfolio_values[date] * weight) / price
                                except KeyError:
                                    logger.warning(f"Price not available for {asset} on rebalance date {date}")
                except Exception as e:
                    logger.warning(f"Error processing date {date}: {str(e)}")
                    # Use previous value if processing failed
                    portfolio_values[date] = portfolio_values.iloc[i-1]
            
            # Calculate portfolio returns
            portfolio_returns = portfolio_values.pct_change().fillna(0)
            
            # Calculate metrics
            total_return = (portfolio_values[-1] / initial_investment) - 1
            
            # Calculate drawdowns
            portfolio_cum_returns = (1 + portfolio_returns).cumprod()
            portfolio_cum_max = portfolio_cum_returns.cummax()
            drawdowns = (portfolio_cum_returns / portfolio_cum_max) - 1
            max_drawdown = drawdowns.min()
            
            # Annualized metrics
            days = (all_dates[-1] - all_dates[0]).days
            years = days / 365.25
            
            annualized_return = (1 + total_return) ** (1 / years) - 1
            annualized_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_vol
            
            # Create asset values DataFrame
            asset_values = pd.DataFrame(index=all_dates, columns=all_assets)
            for date in all_dates:
                for asset in all_assets:
                    if asset in backtest_prices.columns:
                        asset_values.loc[date, asset] = asset_units[asset] * backtest_prices.loc[date, asset]
            
            # Fill weights history for all dates (forward fill)
            weights_all_dates = pd.DataFrame(index=all_dates, columns=all_assets)
            for date in all_dates:
                last_rebalance = max([d for d in rebalance_dates if d <= date])
                weights_all_dates.loc[date] = weights_history.loc[last_rebalance]
            
            # Prepare results
            backtest_results = {
                'initial_investment': initial_investment,
                'start_date': all_dates[0].strftime('%Y-%m-%d'),
                'end_date': all_dates[-1].strftime('%Y-%m-%d'),
                'rebalance_frequency': rebalance_frequency,
                'lookback_window': lookback_window,
                'total_return': float(total_return),
                'annualized_return': float(annualized_return),
                'annualized_volatility': float(annualized_vol),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'final_value': float(portfolio_values[-1])
            }
            
            # Save results
            if save_results:
                # Create a unique ID for this backtest
                backtest_id = f"dynamic_{rebalance_frequency}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Save backtest parameters and results
                with open(backtest_dir / f"{backtest_id}_results.json", 'w') as f:
                    json.dump(backtest_results, f, indent=4)
                
                # Save time series data
                portfolio_values.to_csv(backtest_dir / f"{backtest_id}_portfolio_values.csv", 
                                       header=['Portfolio_Value'])
                
                portfolio_returns.to_csv(backtest_dir / f"{backtest_id}_portfolio_returns.csv",
                                        header=['Portfolio_Return'])
                
                drawdowns.to_csv(backtest_dir / f"{backtest_id}_drawdowns.csv", header=['Drawdown'])
                
                asset_values.to_csv(backtest_dir / f"{backtest_id}_asset_values.csv")
                
                weights_history.to_csv(backtest_dir / f"{backtest_id}_weights_history.csv")
                weights_all_dates.to_csv(backtest_dir / f"{backtest_id}_weights_all_dates.csv")
                
                # Create visualizations
                # 1. Portfolio value over time
                plt.figure(figsize=(12, 8))
                portfolio_values.plot(figsize=(12, 8))
                plt.title('Dynamic Portfolio Value Over Time')
                plt.xlabel('Date')
                plt.ylabel('Value ($)')
                plt.grid(True)
                plt.savefig(backtest_dir / f"{backtest_id}_portfolio_value.png", dpi=300)
                plt.close()
                
                # 2. Asset allocation over time
                plt.figure(figsize=(12, 8))
                # Sample weights at regular intervals for clearer visualization
                sample_dates = pd.date_range(start=all_dates[0], end=all_dates[-1], freq='3M')
                sample_dates = [date for date in sample_dates if date in all_dates]
                
                weights_sample = weights_all_dates.loc[sample_dates]
                weights_sample = weights_sample.loc[:, (weights_sample != 0).any()]  # Remove unused assets
                
                weights_sample.plot.area(figsize=(12, 8), stacked=True)
                plt.title('Dynamic Portfolio Allocation Over Time')
                plt.xlabel('Date')
                plt.ylabel('Allocation (%)')
                plt.grid(True)
                plt.savefig(backtest_dir / f"{backtest_id}_allocation.png", dpi=300)
                plt.close()
                
                # 3. Drawdown chart
                plt.figure(figsize=(12, 8))
                drawdowns.plot(figsize=(12, 8), color='red')
                plt.title('Portfolio Drawdown')
                plt.xlabel('Date')
                plt.ylabel('Drawdown (%)')
                plt.grid(True)
                plt.fill_between(drawdowns.index, 0, drawdowns.values, color='red', alpha=0.3)
                plt.savefig(backtest_dir / f"{backtest_id}_drawdown.png", dpi=300)
                plt.close()
            
            logger.info(f"Dynamic backtest completed. Total return: {total_return:.2%}, Sharpe ratio: {sharpe_ratio:.2f}")
            return backtest_results, portfolio_values, portfolio_returns, weights_history
        
        except Exception as e:
            logger.error(f"Dynamic backtest failed: {str(e)}", exc_info=True)
            raise
    
    def backtest_with_macro_factors(self, weights, macro_factors=None, start_date=None, end_date=None, 
                                initial_investment=100000, save_results=True):
        """
        Backtest a portfolio with macroeconomic factor overlays
        
        Parameters:
        - weights: Dictionary of base asset weights {asset: weight}
        - macro_factors: Dictionary of macro factors to monitor and their thresholds
        - start_date: Backtest start date (or None for earliest date)
        - end_date: Backtest end date (or None for latest date)
        - initial_investment: Initial portfolio value
        - save_results: Whether to save backtest results
        """
        try:
            logger.info("Starting backtest with macroeconomic factor overlays...")
            
            # Load macro data
            try:
                macro_path = data_processed_dir / "cleaned_macro.csv"
                if macro_path.exists():
                    macro_data = pd.read_csv(macro_path, index_col=0, parse_dates=True)
                    
                    # Check if macro data contains actual values
                    if macro_data.empty or macro_data.isna().all().all():
                        logger.warning("Macro data file exists but contains no valid data")
                        return self.backtest_static_allocation(weights, start_date, end_date, 
                                                        initial_investment, 'M', save_results)
                else:
                    logger.warning("No macro data found. Running regular backtest.")
                    return self.backtest_static_allocation(weights, start_date, end_date, 
                                                    initial_investment, 'M', save_results)
            except Exception as e:
                logger.warning(f"Error loading macro data: {str(e)}. Running regular backtest.")
                return self.backtest_static_allocation(weights, start_date, end_date, 
                                                initial_investment, 'M', save_results)
            
            # Filter for assets in the portfolio that exist in price data
            valid_assets = [asset for asset in weights.keys() if asset in self.prices.columns]
            if not valid_assets:
                logger.warning("No valid assets found in price data. Running regular backtest.")
                return self.backtest_static_allocation(weights, start_date, end_date, 
                                                initial_investment, 'M', save_results)
            
            # Filter weights and normalize
            filtered_weights = {asset: weights[asset] for asset in valid_assets}
            total_weight = sum(filtered_weights.values())
            if total_weight <= 0:
                logger.warning("Total weight is zero or negative. Running regular backtest.")
                return self.backtest_static_allocation(weights, start_date, end_date, 
                                                initial_investment, 'M', save_results)
                
            normalized_weights = {asset: weight/total_weight for asset, weight in filtered_weights.items()}
            
            # Filter date range
            if start_date:
                prices = self.prices[self.prices.index >= start_date].copy()
                returns = self.returns[self.returns.index >= start_date].copy()
                macro_data = macro_data[macro_data.index >= start_date].copy()
            else:
                prices = self.prices.copy()
                returns = self.returns.copy()
            
            if end_date:
                prices = prices[prices.index <= end_date]
                returns = returns[returns.index <= end_date]
                macro_data = macro_data[macro_data.index <= end_date].copy()
            
            # Fill any NaN values to avoid issues
            macro_data = macro_data.fillna(method='ffill').fillna(method='bfill')
            
            # Align dates (use intersection of prices and macro)
            common_dates = prices.index.intersection(macro_data.index)
            if len(common_dates) == 0:
                logger.warning("No common dates between price and macro data. Running regular backtest.")
                return self.backtest_static_allocation(normalized_weights, start_date, end_date, 
                                                initial_investment, 'M', save_results)
                                                
            prices = prices.loc[common_dates]
            returns = returns.loc[common_dates]
            macro_data = macro_data.loc[common_dates]
            
            # Define default macro factors if none provided
            if macro_factors is None:
                # Default macro overlays - adjust based on available data
                available_columns = set(macro_data.columns)
                
                macro_factors = {}
                
                # Interest rate related adjustments
                if 'FEDFUNDS' in available_columns:
                    macro_factors['FEDFUNDS'] = {
                        'threshold': 3.0,
                        'above_action': {'TLT': 0.8, 'SPY': 0.2},  # Defensive when rates high
                        'below_action': {'SPY': 0.7, 'QQQ': 0.3}   # Aggressive when rates low
                    }
                
                # Inflation related adjustments
                if 'CPIAUCSL' in available_columns:
                    macro_factors['CPIAUCSL_YOY'] = {
                        'derive_from': 'CPIAUCSL',
                        'transformation': 'pct_change_yoy',
                        'threshold': 0.04,  # 4% inflation
                        'above_action': {'GLD': 0.3, 'TIP': 0.3, 'SPY': 0.4},  # Inflation hedge
                        'below_action': {'SPY': 0.6, 'TLT': 0.4}   # Standard allocation
                    }
                
                # Volatility related adjustments
                if 'VIX' in available_columns:
                    macro_factors['VIX'] = {
                        'threshold': 25,
                        'above_action': {'TLT': 0.6, 'SPY': 0.2, 'GLD': 0.2},  # Risk-off
                        'below_action': {'SPY': 0.7, 'QQQ': 0.3}   # Risk-on
                    }
            
            # Prepare derived factors
            for factor_name, factor_config in macro_factors.items():
                if 'derive_from' in factor_config:
                    source = factor_config['derive_from']
                    
                    # Skip if source factor doesn't exist
                    if source not in macro_data.columns:
                        continue
                        
                    transform = factor_config['transformation']
                    
                    try:
                        if transform == 'pct_change_yoy':
                            # Year-over-year percentage change
                            macro_data[factor_name] = macro_data[source].pct_change(periods=252) * 100
                        elif transform == 'diff_yoy':
                            # Year-over-year difference
                            macro_data[factor_name] = macro_data[source].diff(periods=252)
                        elif transform == 'z_score':
                            # Z-score (rolling)
                            rolling_mean = macro_data[source].rolling(window=252).mean()
                            rolling_std = macro_data[source].rolling(window=252).std()
                            macro_data[factor_name] = (macro_data[source] - rolling_mean) / rolling_std
                    except Exception as e:
                        logger.warning(f"Error calculating derived factor {factor_name}: {str(e)}")
            
            # Initialize results
            all_dates = prices.index
            all_assets = prices.columns
            
            # Check if any dates remain
            if len(all_dates) == 0:
                logger.error("No dates available for macro backtesting")
                return self.backtest_static_allocation(normalized_weights, start_date, end_date, 
                                                initial_investment, 'M', save_results)
            
            # Initialize with all dates
            portfolio_values = pd.Series(0.0, index=all_dates, dtype=float)
            portfolio_values.iloc[0] = initial_investment
            
            # Create dictionaries to track units and weights
            asset_units = {asset: 0.0 for asset in all_assets}
            weights_history = pd.DataFrame(index=all_dates, columns=all_assets, data=0.0)
            factor_signals = pd.DataFrame(index=all_dates, columns=list(macro_factors.keys()), data=0.0)
            
            # Initial allocation based on macro factors
            current_date = all_dates[0]
            current_weights = self._adjust_weights_by_macro(normalized_weights, macro_data, macro_factors, current_date)
            
            # Filter for available assets and normalize
            current_weights = {asset: weight for asset, weight in current_weights.items() 
                            if asset in prices.columns}
            weight_sum = sum(current_weights.values())
            if weight_sum > 0:
                current_weights = {asset: weight/weight_sum for asset, weight in current_weights.items()}
            else:
                # Fallback to equal weight if no valid weights
                current_weights = {asset: 1.0/len(prices.columns) for asset in prices.columns}
            
            # Record initial weights and signals
            for asset, weight in current_weights.items():
                weights_history.loc[current_date, asset] = weight
            
            for factor, config in macro_factors.items():
                if factor in macro_data.columns:
                    factor_value = macro_data.loc[current_date, factor]
                    threshold = config['threshold']
                    factor_signals.loc[current_date, factor] = 1 if factor_value > threshold else -1
            
            # Calculate initial units with error checking
            for asset, weight in current_weights.items():
                if asset in prices.columns:
                    price = prices.loc[current_date, asset]
                    if not np.isnan(price) and price > 0:
                        asset_units[asset] = (initial_investment * weight) / price
            
            # Run the backtest
            for i, date in enumerate(all_dates[1:], 1):
                current_date = date
                
                try:
                    # Adjust weights based on current macro factors
                    current_weights = self._adjust_weights_by_macro(normalized_weights, macro_data, macro_factors, current_date)
                    
                    # Filter for available assets and normalize
                    current_weights = {asset: weight for asset, weight in current_weights.items() 
                                    if asset in prices.columns}
                    weight_sum = sum(current_weights.values())
                    if weight_sum > 0:
                        current_weights = {asset: weight/weight_sum for asset, weight in current_weights.items()}
                    
                    # Record weights and signals
                    for asset, weight in current_weights.items():
                        weights_history.loc[current_date, asset] = weight
                    
                    for factor, config in macro_factors.items():
                        if factor in macro_data.columns:
                            factor_value = macro_data.loc[current_date, factor]
                            threshold = config['threshold']
                            factor_signals.loc[current_date, factor] = 1 if factor_value > threshold else -1
                    
                    # Calculate current portfolio value
                    current_portfolio_value = 0
                    for asset, units in asset_units.items():
                        if asset in prices.columns:
                            price = prices.loc[current_date, asset]
                            if not np.isnan(price) and price > 0:
                                current_portfolio_value += units * price
                    
                    # Update portfolio value with error checking
                    if current_portfolio_value > 0:
                        portfolio_values[date] = current_portfolio_value
                    else:
                        # Use the previous value if calculation produced invalid result
                        portfolio_values[date] = portfolio_values.iloc[i-1]
                        logger.warning(f"Invalid portfolio value calculated for {date}. Using previous value.")
                    
                    # Rebalance monthly
                    if i > 0 and (date.month != all_dates[i-1].month):
                        for asset, weight in current_weights.items():
                            if asset in prices.columns:
                                price = prices.loc[current_date, asset]
                                if not np.isnan(price) and price > 0:
                                    asset_units[asset] = (current_portfolio_value * weight) / price
                except Exception as e:
                    logger.warning(f"Error processing date {date}: {str(e)}")
                    # Use previous values if an error occurs
                    if i > 0:
                        portfolio_values[date] = portfolio_values.iloc[i-1]
            
            # Calculate portfolio returns with error handling
            portfolio_returns = portfolio_values.pct_change().fillna(0)
            
            # Remove extreme outliers (likely calculation errors)
            upper_bound = portfolio_returns.mean() + 5 * portfolio_returns.std()
            lower_bound = portfolio_returns.mean() - 5 * portfolio_returns.std()
            portfolio_returns = portfolio_returns.clip(lower=lower_bound, upper=upper_bound)
            
            # Calculate metrics with error handling
            try:
                total_return = (portfolio_values.iloc[-1] / initial_investment) - 1
            except:
                total_return = np.nan
            
            # Calculate drawdowns with error handling
            try:
                portfolio_cum_returns = (1 + portfolio_returns).cumprod()
                portfolio_cum_max = portfolio_cum_returns.cummax()
                
                # Handle zeros in portfolio_cum_max
                with np.errstate(divide='ignore', invalid='ignore'):
                    drawdowns = (portfolio_cum_returns / portfolio_cum_max) - 1
                    drawdowns = drawdowns.replace([np.inf, -np.inf], np.nan)
                
                max_drawdown = drawdowns.min()
            except:
                drawdowns = pd.Series(0, index=all_dates)
                max_drawdown = 0
            
            # Annualized metrics with error handling
            days = (all_dates[-1] - all_dates[0]).days
            years = max(days / 365.25, 0.1)  # Avoid very small year values
            
            try:
                annualized_return = (1 + total_return) ** (1 / years) - 1
            except:
                annualized_return = np.nan
                
            try:
                annualized_vol = portfolio_returns.std() * np.sqrt(252)
            except:
                annualized_vol = np.nan
                
            try:
                sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_vol
            except:
                sharpe_ratio = np.nan
            
            # Prepare results
            backtest_results = {
                'initial_investment': initial_investment,
                'start_date': all_dates[0].strftime('%Y-%m-%d'),
                'end_date': all_dates[-1].strftime('%Y-%m-%d'),
                'macro_factors': list(macro_factors.keys()),
                'total_return': float(total_return if not np.isnan(total_return) else 0),
                'annualized_return': float(annualized_return if not np.isnan(annualized_return) else 0),
                'annualized_volatility': float(annualized_vol if not np.isnan(annualized_vol) else 0),
                'sharpe_ratio': float(sharpe_ratio if not np.isnan(sharpe_ratio) else 0),
                'max_drawdown': float(max_drawdown if not np.isnan(max_drawdown) else 0),
                'final_value': float(portfolio_values.iloc[-1] if not np.isnan(portfolio_values.iloc[-1]) else initial_investment)
            }
            
            # The rest of the function (saving results, plotting, etc.) remains unchanged
            
            # Return results with filled NaN values
            logger.info(f"Macro factor backtest completed. Return: {total_return:.2%}, Sharpe ratio: {sharpe_ratio:.2f}")
            return backtest_results, portfolio_values, portfolio_returns, weights_history
            
        except Exception as e:
            logger.error(f"Macro factor backtest failed: {str(e)}", exc_info=True)
            raise
    
    def _adjust_weights_by_macro(self, base_weights, macro_data, macro_factors, current_date):
        """Helper method to adjust portfolio weights based on macro factors"""
        try:
            # Start with base weights
            adjusted_weights = base_weights.copy()
            
            # Track if any factors were triggered
            factor_triggered = False
            
            # Check each factor
            for factor_name, factor_config in macro_factors.items():
                # Skip factors not in the data
                if factor_name not in macro_data.columns:
                    continue
                    
                # Get current value with error handling
                try:
                    current_value = macro_data.loc[current_date, factor_name]
                    
                    # Skip if value is NaN
                    if pd.isna(current_value):
                        continue
                        
                    threshold = factor_config['threshold']
                    
                    # Check if above threshold
                    if current_value > threshold and 'above_action' in factor_config:
                        factor_weights = factor_config['above_action']
                        # Filter for assets that exist in our price data
                        factor_weights = {asset: weight for asset, weight in factor_weights.items() 
                                        if asset in self.prices.columns}
                        
                        # Only use these weights if we have some valid assets
                        if factor_weights:
                            adjusted_weights = factor_weights
                            factor_triggered = True
                            logger.debug(f"Factor {factor_name} above threshold ({current_value} > {threshold})")
                            break
                    
                    # Check if below threshold
                    elif current_value <= threshold and 'below_action' in factor_config:
                        factor_weights = factor_config['below_action']
                        # Filter for assets that exist in our price data
                        factor_weights = {asset: weight for asset, weight in factor_weights.items() 
                                        if asset in self.prices.columns}
                        
                        # Only use these weights if we have some valid assets
                        if factor_weights:
                            adjusted_weights = factor_weights
                            factor_triggered = True
                            logger.debug(f"Factor {factor_name} below threshold ({current_value} <= {threshold})")
                            break
                except Exception as e:
                    logger.debug(f"Error processing factor {factor_name}: {str(e)}")
                    continue
            
            # Ensure all weights sum to 1
            total = sum(adjusted_weights.values())
            if total > 0:
                adjusted_weights = {asset: weight/total for asset, weight in adjusted_weights.items()}
            else:
                # Fallback to equal weights if something went wrong
                available_assets = [asset for asset in adjusted_weights.keys() if asset in self.prices.columns]
                if available_assets:
                    adjusted_weights = {asset: 1.0/len(available_assets) for asset in available_assets}
                else:
                    # If no assets matched, use all assets in price data
                    adjusted_weights = {asset: 1.0/len(self.prices.columns) for asset in self.prices.columns}
            
            return adjusted_weights
            
        except Exception as e:
            logger.warning(f"Error adjusting weights by macro factors: {str(e)}")
            # Return base weights as fallback
            return base_weights
    
    def run_all_backtests(self, save_results=True):
        """Run all backtesting methods with default parameters"""
        try:
            logger.info("Running all portfolio backtests...")
            
            # Load optimization results for weights
            optimization_dir = data_processed_dir / "optimization"
            
            # Default weights (equal weight if no optimization available)
            equal_weights = {asset: 1.0/len(self.returns.columns) for asset in self.returns.columns}
            
            # Try to load optimized weights
            try:
                max_sharpe_path = optimization_dir / "max_sharpe_optimized_weights.csv"
                if max_sharpe_path.exists():
                    max_sharpe_df = pd.read_csv(max_sharpe_path)
                    max_sharpe_weights = {row['Asset']: row['Weight'] for _, row in max_sharpe_df.iterrows()}
                else:
                    max_sharpe_weights = equal_weights
                    
                min_vol_path = optimization_dir / "min_vol_optimized_weights.csv"
                if min_vol_path.exists():
                    min_vol_df = pd.read_csv(min_vol_path)
                    min_vol_weights = {row['Asset']: row['Weight'] for _, row in min_vol_df.iterrows()}
                else:
                    min_vol_weights = equal_weights
                    
                risk_parity_path = optimization_dir / "risk_parity_data.csv"
                if risk_parity_path.exists():
                    risk_parity_df = pd.read_csv(risk_parity_path)
                    risk_parity_weights = {row['Asset']: row['Weight'] for _, row in risk_parity_df.iterrows()}
                else:
                    risk_parity_weights = equal_weights
            except:
                max_sharpe_weights = equal_weights
                min_vol_weights = equal_weights
                risk_parity_weights = equal_weights
            
            # Define backtest strategies
            strategies = [
                {
                    'name': 'Equal Weight',
                    'weights': equal_weights,
                    'rebalance_frequency': 'M'
                },
                {
                    'name': 'Max Sharpe',
                    'weights': max_sharpe_weights,
                    'rebalance_frequency': 'M'
                },
                {
                    'name': 'Min Volatility',
                    'weights': min_vol_weights,
                    'rebalance_frequency': 'M'
                },
                {
                    'name': 'Risk Parity',
                    'weights': risk_parity_weights,
                    'rebalance_frequency': 'M'
                },
                {
                    'name': 'Buy and Hold',
                    'weights': equal_weights,
                    'rebalance_frequency': None
                }
            ]
            
            # Run strategy comparison
            self.backtest_strategy_comparison(strategies, save_results=save_results)
            
            # Run macro factor backtest with max Sharpe weights
            self.backtest_with_macro_factors(max_sharpe_weights, save_results=save_results)
            
            # Run dynamic allocation example (using lookback momentum strategy)
            def momentum_allocation(historical_returns, current_date):
                # Simple momentum strategy - allocate to assets with highest 6-month return
                if len(historical_returns) < 126:  # Need at least 6 months
                    return equal_weights
                
                # Calculate 6-month returns
                returns_6m = historical_returns.iloc[-126:].mean() * 252
                
                # Rank assets by return
                ranked_assets = returns_6m.sort_values(ascending=False)
                
                # Allocate to top 3 assets
                top_assets = ranked_assets.index[:3]
                momentum_weights = {asset: 1/3 if asset in top_assets else 0 for asset in historical_returns.columns}
                
                return momentum_weights
            
            self.backtest_dynamic_allocation(momentum_allocation, lookback_window=252, 
                                          rebalance_frequency='M', save_results=save_results)
            
            logger.info("All backtests completed successfully.")
            
        except Exception as e:
            logger.error(f"Running all backtests failed: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    backtester = PortfolioBacktester()
    backtester.run_all_backtests()