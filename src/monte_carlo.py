import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from scipy import stats
import time
from datetime import datetime, timedelta
from config.settings import settings
from utils.logger import setup_logger
from functools import partial
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import traceback
import os

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Setup directories
data_processed_dir = project_root / settings.DATA_PROCESSED_DIR
monte_carlo_dir = data_processed_dir / "monte_carlo"
monte_carlo_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)

class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio performance projection"""
    
    def __init__(self, returns_data=None, portfolio_weights=None, cache_dir=None):
        """Initialize with returns data and portfolio weights"""
        self.returns = returns_data
        self.weights = portfolio_weights
        self.risk_free_rate = settings.RISK_FREE_RATE
        
        # Setup caching directory
        self.cache_dir = cache_dir if cache_dir else monte_carlo_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data if not provided
        if self.returns is None:
            returns_path = data_processed_dir / "daily_returns.csv"
            self.returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        # Load default weights if not provided (max Sharpe or equal weight)
        if self.weights is None:
            try:
                optimization_dir = data_processed_dir / "optimization"
                weights_path = optimization_dir / "max_sharpe_optimized_weights.csv"
                
                if weights_path.exists():
                    weights_df = pd.read_csv(weights_path)
                    self.weights = {row['Asset']: row['Weight'] for _, row in weights_df.iterrows()}
                else:
                    # Use equal weights if no optimization results available
                    self.weights = {asset: 1.0/len(self.returns.columns) for asset in self.returns.columns}
            except:
                # Use equal weights as fallback
                self.weights = {asset: 1.0/len(self.returns.columns) for asset in self.returns.columns}
    
    def _get_cache_key(self, initial_investment, num_simulations, time_horizon, return_method):
        """Generate a cache key based on simulation parameters"""
        assets = '_'.join(sorted(self.weights.keys()))
        weights = '_'.join([f"{k}_{v:.4f}" for k, v in sorted(self.weights.items())])
        # Hash the assets and weights to avoid excessively long filenames
        import hashlib
        assets_hash = hashlib.md5(assets.encode()).hexdigest()[:8]
        weights_hash = hashlib.md5(weights.encode()).hexdigest()[:8]
        
        return f"mc_{return_method}_{initial_investment}_{num_simulations}_{time_horizon}_{assets_hash}_{weights_hash}"
    
    def _check_cache(self, cache_key):
        """Check if simulation results are cached"""
        cache_file = self.cache_dir / f"{cache_key}.npz"
        if cache_file.exists():
            try:
                cached_data = np.load(cache_file)
                logger.info(f"Found cached simulation results: {cache_key}")
                return cached_data['sim_data'], json.loads(cached_data['stats'].item())
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        return None, None
    
    def _save_to_cache(self, cache_key, sim_data, stats_dict):
        """Save simulation results to cache"""
        cache_file = self.cache_dir / f"{cache_key}.npz"
        try:
            np.savez(cache_file, 
                     sim_data=sim_data, 
                     stats=json.dumps(stats_dict))
            logger.info(f"Saved simulation results to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def _simulate_path_parametric(self, mean_return, cov_matrix, time_horizon, random_seed=None):
        """Generate a single parametric simulation path"""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Multivariate normal simulation
        asset_returns = np.random.multivariate_normal(mean_return, cov_matrix, size=time_horizon)
        
        # Apply weights to get portfolio returns
        portfolio_returns = np.sum(asset_returns * np.array(list(self.weights.values())), axis=1)
        
        # Convert to cumulative values
        return np.cumprod(1 + portfolio_returns)
    
    def _simulate_path_historical(self, historical_returns, time_horizon, random_seed=None):
        """Generate a single historical simulation path"""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Randomly sample historical portfolio returns
        sampled_returns = np.random.choice(historical_returns, size=time_horizon, replace=True)
        
        # Convert to cumulative values
        return np.cumprod(1 + sampled_returns)
    
    def _simulate_path_bootstrap(self, historical_returns, time_horizon, block_size=20, random_seed=None):
        """Generate a single bootstrap simulation path"""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Block bootstrap to preserve serial correlation
        num_blocks = int(np.ceil(time_horizon / block_size))
        
        # Calculate portfolio returns
        sim_returns = []
        for _ in range(num_blocks):
            # Random starting point for block
            start_idx = np.random.randint(0, len(historical_returns) - block_size)
            block = historical_returns[start_idx:start_idx + block_size]
            sim_returns.extend(block)
        
        # Trim to exact time horizon
        sim_returns = sim_returns[:time_horizon]
        
        # Convert to cumulative values
        return np.cumprod(1 + np.array(sim_returns))
    
    def _run_simulation_with_seed(self, seed, simulate_func):
        """Helper method to run a single simulation with a specific seed"""
        try:
            return simulate_func(random_seed=seed)
        except Exception as e:
            # Log error and return None
            error_msg = f"Error in simulation: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Print to console in multiprocessing context
            return None
    
    def run_simulation(self, initial_investment=100000, num_simulations=1000, time_horizon=252*5, 
                      return_method='parametric', use_cache=True, num_processes=None,
                      save_results=True):
        """
        Run Monte Carlo simulation for portfolio performance
        
        Parameters:
        - initial_investment: Initial portfolio value
        - num_simulations: Number of simulation paths
        - time_horizon: Number of days to simulate
        - return_method: Method to generate returns ('parametric', 'historical', 'bootstrap')
        - use_cache: Whether to use cached results if available
        - num_processes: Number of processes to use (None = auto-detect)
        - save_results: Whether to save simulation results
        """
        try:
            start_time = time.time()
            logger.info(f"Running Monte Carlo simulation with {num_simulations} paths over {time_horizon} days using {return_method} method...")
            
            # Check cache if enabled
            if use_cache:
                cache_key = self._get_cache_key(initial_investment, num_simulations, time_horizon, return_method)
                cached_sim, cached_stats = self._check_cache(cache_key)
                if cached_sim is not None and cached_stats is not None:
                    elapsed_time = time.time() - start_time
                    logger.info(f"Using cached results. Retrieved in {elapsed_time:.2f} seconds")
                    return cached_sim, cached_stats
            
            # Filter returns for assets in the portfolio
            assets = list(self.weights.keys())
            weights_array = np.array([self.weights[asset] for asset in assets])
            
            try:
                portfolio_returns = self.returns[assets]
                # Calculate portfolio historical returns
                historical_portfolio_returns = portfolio_returns.dot(weights_array)
            except Exception as e:
                logger.warning(f"Error calculating portfolio returns: {e}")
                # Create a fallback of all available assets
                available_assets = [a for a in assets if a in self.returns.columns]
                if not available_assets:
                    raise ValueError("No portfolio assets found in returns data")
                
                # Recalculate weights using only available assets
                available_weights = {a: self.weights[a] for a in available_assets}
                total_weight = sum(available_weights.values())
                available_weights = {a: w/total_weight for a, w in available_weights.items()}
                
                portfolio_returns = self.returns[available_assets]
                weights_array = np.array([available_weights[asset] for asset in available_assets])
                historical_portfolio_returns = portfolio_returns.dot(weights_array)
            
            # Prepare simulation parameters based on method
            if return_method == 'parametric':
                # Calculate asset return statistics
                mean_return = portfolio_returns.mean().values
                cov_matrix = portfolio_returns.cov().values
                
                # Define simulation function for this method
                simulate_func = partial(self._simulate_path_parametric, 
                                       mean_return, 
                                       cov_matrix, 
                                       time_horizon)
            
            elif return_method == 'historical':
                # Prepare historical returns
                historical_returns = historical_portfolio_returns.values
                
                # Define simulation function for this method
                simulate_func = partial(self._simulate_path_historical,
                                      historical_returns,
                                      time_horizon)
            
            elif return_method == 'bootstrap':
                # Prepare historical returns
                historical_returns = historical_portfolio_returns.values
                
                # Define simulation function for this method
                simulate_func = partial(self._simulate_path_bootstrap,
                                      historical_returns,
                                      time_horizon)
            
            else:
                raise ValueError(f"Invalid return method: {return_method}")
            
            # Determine number of processes to use
            if num_processes is None:
                num_processes = max(1, multiprocessing.cpu_count() - 1)
            
            # Generate simulation seeds for reproducibility
            seeds = np.random.randint(0, 2**32, size=num_simulations)
            
            # Run simulations in parallel
            logger.info(f"Running simulations using {num_processes} processes")
            
            # Run simulations in parallel
            sim_results = []
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Use functools.partial to pass simulate_func to _run_simulation_with_seed
                import functools
                worker_func = functools.partial(self._run_simulation_with_seed, simulate_func=simulate_func)
                sim_results = list(executor.map(worker_func, seeds))
            
            # Filter out failed simulations
            sim_results = [r for r in sim_results if r is not None]
            
            if len(sim_results) < num_simulations * 0.9:  # Less than 90% success
                logger.warning(f"Only {len(sim_results)}/{num_simulations} simulations completed successfully")
            
            # Stack results into a 2D array
            sim_cumulative = np.column_stack(sim_results) * initial_investment
            
            # Calculate statistics for final values
            final_values = sim_cumulative[-1, :]
            
            stats_dict = {
                'mean': float(np.mean(final_values)),
                'median': float(np.median(final_values)),
                'min': float(np.min(final_values)),
                'max': float(np.max(final_values)),
                'std': float(np.std(final_values)),
                'initial_investment': initial_investment,
                'time_horizon_days': time_horizon,
                'annualized_return': float(((np.mean(final_values) / initial_investment) ** (252 / time_horizon)) - 1),
                'simulation_method': return_method
            }
            
            # Calculate percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                stats_dict[f'percentile_{p}'] = float(np.percentile(final_values, p))
            
            # Calculate probability of meeting various return targets
            return_targets = [-0.2, 0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # -20% to +100%
            for target in return_targets:
                target_value = initial_investment * (1 + target)
                prob = np.mean(final_values >= target_value)
                stats_dict[f'prob_return_{target:.1f}'] = float(prob)
            
            # Cache results if caching is enabled
            if use_cache:
                cache_key = self._get_cache_key(initial_investment, num_simulations, time_horizon, return_method)
                self._save_to_cache(cache_key, sim_cumulative, stats_dict)
            
            # Prepare results for saving
            if save_results:
                # Save simulation paths (sample to reduce file size)
                sample_paths = np.random.choice(range(sim_cumulative.shape[1]), min(100, sim_cumulative.shape[1]), replace=False)
                paths_df = pd.DataFrame(sim_cumulative[:, sample_paths])
                paths_df.index = [f'day_{i+1}' for i in range(time_horizon)]
                paths_df.columns = [f'sim_{i+1}' for i in range(len(sample_paths))]
                paths_df.to_csv(monte_carlo_dir / f"mc_paths_{return_method}.csv")
                
                # Save statistics
                with open(monte_carlo_dir / f"mc_stats_{return_method}.json", 'w') as f:
                    json.dump(stats_dict, f, indent=4)
                
                # Save all final values for histogram
                pd.DataFrame({'final_value': final_values}).to_csv(
                    monte_carlo_dir / f"mc_final_values_{return_method}.csv", index=False)
                
                # Create visualization
                plt.figure(figsize=(12, 8))
                
                # Plot sample paths
                for i in sample_paths[:20]:  # Limit to 20 paths for clarity
                    plt.plot(sim_cumulative[:, i], linewidth=0.5, alpha=0.6)
                
                # Plot percentiles
                percentiles = [10, 50, 90]
                percentile_values = np.percentile(sim_cumulative, percentiles, axis=1)
                for i, p in enumerate(percentiles):
                    plt.plot(percentile_values[i], linewidth=2, 
                             label=f'{p}th Percentile', linestyle=['--', '-', '--'][i])
                
                plt.title(f'Monte Carlo Simulation - {return_method.capitalize()} Method')
                plt.xlabel('Days')
                plt.ylabel('Portfolio Value ($)')
                plt.legend()
                plt.savefig(monte_carlo_dir / f"mc_simulation_{return_method}.png", dpi=300)
                
                # Create histogram of final values
                plt.figure(figsize=(12, 8))
                plt.hist(final_values, bins=50, alpha=0.7)
                plt.axvline(initial_investment, color='r', linestyle='--', 
                           label=f'Initial Investment (${initial_investment:,.0f})')
                plt.axvline(np.mean(final_values), color='g', linestyle='-', 
                           label=f'Mean Final Value (${np.mean(final_values):,.0f})')
                plt.title(f'Distribution of Final Portfolio Values - {return_method.capitalize()} Method')
                plt.xlabel('Portfolio Value ($)')
                plt.ylabel('Frequency')
                plt.legend()
                plt.savefig(monte_carlo_dir / f"mc_histogram_{return_method}.png", dpi=300)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Monte Carlo simulation completed in {elapsed_time:.2f} seconds. Mean final value: ${stats_dict['mean']:,.2f}")
            return sim_cumulative, stats_dict
        
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {str(e)}", exc_info=True)
            raise
    
    def run_stress_test(self, initial_investment=100000, stress_scenarios=None, save_results=True):
        """Run stress tests for the portfolio based on historical market crashes or custom scenarios"""
        try:
            logger.info("Running portfolio stress tests...")
            
            # Filter returns for assets in the portfolio
            assets = list(self.weights.keys())
            weights_array = np.array([self.weights[asset] for asset in assets])
            
            try:
                portfolio_returns = self.returns[assets]
                # Calculate portfolio historical returns
                historical_portfolio_returns = portfolio_returns.dot(weights_array)
            except Exception as e:
                logger.warning(f"Error calculating portfolio returns for stress test: {e}")
                # Create a fallback of all available assets
                available_assets = [a for a in assets if a in self.returns.columns]
                if not available_assets:
                    raise ValueError("No portfolio assets found in returns data")
                
                # Recalculate weights using only available assets
                available_weights = {a: self.weights[a] for a in available_assets}
                total_weight = sum(available_weights.values())
                available_weights = {a: w/total_weight for a, w in available_weights.items()}
                
                portfolio_returns = self.returns[available_assets]
                weights_array = np.array([available_weights[asset] for asset in available_assets])
                historical_portfolio_returns = portfolio_returns.dot(weights_array)
            
            # Define stress scenarios if not provided
            if stress_scenarios is None:
                stress_scenarios = {
                    'covid_crash_2020': {
                        'start': '2020-02-19',
                        'end': '2020-03-23',
                        'description': 'COVID-19 Market Crash (2020)'
                    },
                    'financial_crisis_2008': {
                        'start': '2008-09-01',
                        'end': '2009-03-01',
                        'description': '2008 Financial Crisis'
                    },
                    'dot_com_crash': {
                        'start': '2000-03-01',
                        'end': '2002-10-01',
                        'description': 'Dot-com Bubble Burst (2000-2002)'
                    },
                    'custom_scenario_1': {
                        'returns': [-0.02] * 10 + [-0.05] * 5 + [-0.03] * 10,
                        'description': 'Custom Bearish Scenario'
                    },
                    'custom_scenario_2': {
                        'returns': [0.015] * 20 + [-0.03] * 10 + [0.01] * 15,
                        'description': 'Custom Volatile Scenario'
                    }
                }
            
            stress_results = {}
            
            # Run each stress scenario
            for scenario_name, scenario in stress_scenarios.items():
                logger.info(f"Running stress test: {scenario_name}")
                
                # Get scenario returns
                if 'returns' in scenario:
                    # Custom returns scenario
                    scenario_returns = pd.Series(scenario['returns'])
                else:
                    # Historical scenario
                    try:
                        start_date = scenario['start']
                        end_date = scenario['end']
                        
                        # Check if dates are within our data range
                        if start_date not in portfolio_returns.index or end_date not in portfolio_returns.index:
                            logger.warning(f"Dates for scenario {scenario_name} not in data range. Skipping.")
                            continue
                        
                        # Get returns for period
                        period_returns = portfolio_returns.loc[start_date:end_date]
                        
                        # Calculate portfolio returns
                        scenario_returns = period_returns.dot(weights_array)
                    except Exception as e:
                        logger.warning(f"Error retrieving historical data for {scenario_name}: {str(e)}")
                        continue
                
                # If we have an empty series (could happen with date filtering)
                if len(scenario_returns) == 0:
                    logger.warning(f"No valid returns for scenario {scenario_name}. Using default stress scenario.")
                    # Create a small default scenario
                    scenario_returns = pd.Series([-0.01, -0.02, -0.03, -0.02, -0.01])

                try:
                    # Calculate cumulative portfolio value
                    cumulative_value = initial_investment * np.cumprod(1 + scenario_returns)
                    
                    # Calculate drawdown
                    if isinstance(cumulative_value, pd.Series):
                        max_value = cumulative_value.cummax()
                        drawdown = (cumulative_value - max_value) / max_value
                        max_drawdown = drawdown.min()
                    else:
                        max_value = np.maximum.accumulate(cumulative_value)
                        drawdown = (cumulative_value - max_value) / max_value
                        max_drawdown = np.min(drawdown)
                    
                    # Calculate metrics
                    if isinstance(cumulative_value, pd.Series):
                        total_return = (cumulative_value.iloc[-1] / initial_investment) - 1
                    else:
                        total_return = (cumulative_value[-1] / initial_investment) - 1
                    
                    # Store results
                    stress_results[scenario_name] = {
                        'description': scenario.get('description', ''),
                        'initial_value': float(initial_investment),
                        'final_value': float(cumulative_value.iloc[-1] if isinstance(cumulative_value, pd.Series) else cumulative_value[-1]),
                        'total_return': float(total_return),
                        'max_drawdown': float(max_drawdown),
                        'duration_days': len(scenario_returns),
                        'daily_returns': scenario_returns.tolist() if isinstance(scenario_returns, pd.Series) else list(scenario_returns),
                        'cumulative_values': cumulative_value.tolist() if isinstance(cumulative_value, pd.Series) else list(cumulative_value)
                    }
                except Exception as e:
                    logger.warning(f"Error processing stress scenario {scenario_name}: {str(e)}")
                    # Add a minimal result so we can continue
                    stress_results[scenario_name] = {
                        'description': scenario.get('description', ''),
                        'error': str(e),
                        'status': 'failed'
                    }

                # Create visualization for this scenario
                if save_results:
                    try:
                        # Only create visualization if we have valid data
                        if 'cumulative_values' in stress_results[scenario_name]:
                            plt.figure(figsize=(12, 8))
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                            
                            # Get the values from the results dictionary
                            cumulative_value = stress_results[scenario_name]['cumulative_values']
                            
                            # For drawdown visualization
                            if isinstance(cumulative_value, list):
                                cumulative_value = np.array(cumulative_value)
                            
                            running_max = np.maximum.accumulate(cumulative_value)
                            drawdown = (cumulative_value - running_max) / running_max
                            
                            # Plot portfolio value
                            ax1.plot(cumulative_value)
                            ax1.set_title(f'Stress Test: {stress_results[scenario_name]["description"]}')
                            ax1.set_ylabel('Portfolio Value ($)')
                            ax1.axhline(initial_investment, color='r', linestyle='--')
                            
                            # Plot drawdown
                            ax2.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
                            ax2.plot(drawdown, color='red', label=f'Max Drawdown: {stress_results[scenario_name]["max_drawdown"]:.2%}')
                            ax2.set_ylabel('Drawdown')
                            ax2.set_xlabel('Trading Days')
                            ax2.legend()
                            
                            plt.tight_layout()
                            plt.savefig(monte_carlo_dir / f"stress_test_{scenario_name}.png", dpi=300)
                            plt.close()
                    except Exception as e:
                        logger.warning(f"Could not create visualization for {scenario_name}: {str(e)}")
            
            # Save all stress test results
            if save_results and stress_results:
                with open(monte_carlo_dir / "stress_test_results.json", 'w') as f:
                    json.dump(stress_results, f, indent=4)
                
                # Create a summary table
                summary_data = []
                for scenario_name, result in stress_results.items():
                    if 'status' not in result or result['status'] != 'failed':
                        try:
                            summary_data.append({
                                'Scenario': result.get('description', scenario_name),
                                'Initial Value': result.get('initial_value', 0),
                                'Final Value': result.get('final_value', 0),
                                'Total Return': result.get('total_return', 0),
                                'Max Drawdown': result.get('max_drawdown', 0),
                                'Duration (Days)': result.get('duration_days', 0)
                            })
                        except Exception:
                            pass
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_csv(monte_carlo_dir / "stress_test_summary.csv", index=False)
            
            logger.info(f"Portfolio stress tests completed for {len(stress_results)} scenarios.")
            return stress_results
        
        except Exception as e:
            logger.error(f"Portfolio stress test failed: {str(e)}", exc_info=True)
            raise
            
    def run_all_simulations(self, num_processes=None):
        """Run all Monte Carlo simulations and stress tests"""
        try:
            logger.info("Running all Monte Carlo simulations and stress tests...")
            
            # Run simulations with different methods
            for method in ['parametric', 'historical', 'bootstrap']:
                self.run_simulation(return_method=method, num_processes=num_processes)
            
            # Run stress tests
            self.run_stress_test()
            
            logger.info("All simulations completed successfully.")
            
        except Exception as e:
            logger.error(f"Simulations failed: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    simulator = MonteCarloSimulator()
    simulator.run_all_simulations()