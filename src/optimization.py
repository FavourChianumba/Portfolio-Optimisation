import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from config.settings import settings
from utils.logger import setup_logger
from src.risk_metrics import RiskMetrics

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Setup directories
data_processed_dir = project_root / settings.DATA_PROCESSED_DIR
optimization_dir = data_processed_dir / "optimization"
optimization_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)

class PortfolioOptimizer:
    """Portfolio optimization using various methods including Mean-Variance, Risk Parity, etc."""
    
    def __init__(self, returns_data=None, risk_metrics=None):
        """Initialize with returns data and risk metrics"""
        self.returns = returns_data
        self.risk_free_rate = settings.RISK_FREE_RATE
        
        # Load data if not provided
        if self.returns is None:
            returns_path = data_processed_dir / "daily_returns.csv"
            self.returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        # Initialize risk metrics
        if risk_metrics is None:
            self.risk_metrics = RiskMetrics(returns_data=self.returns)
        else:
            self.risk_metrics = risk_metrics
        
        # Calculate mean returns and covariance matrix
        self.mean_returns = self.returns.mean() * 252  # Annualized
        self.cov_matrix = self.returns.cov() * 252  # Annualized
        
        # Save basic optimization inputs
        self.mean_returns.to_csv(optimization_dir / "mean_returns.csv")
        self.cov_matrix.to_csv(optimization_dir / "covariance_matrix.csv")
    
    def generate_efficient_frontier(self, num_portfolios=100, save_results=True):
        """Generate the efficient frontier with random portfolios"""
        try:
            logger.info("Generating efficient frontier...")
            
            assets = self.returns.columns
            num_assets = len(assets)
            
            # Arrays to store results
            all_weights = np.zeros((num_portfolios, num_assets))
            ret_arr = np.zeros(num_portfolios)
            vol_arr = np.zeros(num_portfolios)
            sharpe_arr = np.zeros(num_portfolios)
            
            # Generate random portfolios
            for i in range(num_portfolios):
                # Random weights
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                all_weights[i, :] = weights
                
                # Portfolio return
                ret_arr[i] = np.sum(weights * self.mean_returns)
                
                # Portfolio volatility
                vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                
                # Sharpe ratio
                sharpe_arr[i] = (ret_arr[i] - self.risk_free_rate) / vol_arr[i]
            
            # Create results DataFrame
            results = pd.DataFrame({
                'Return': ret_arr,
                'Volatility': vol_arr,
                'Sharpe': sharpe_arr
            })
            
            # Add portfolio weights to results
            for i, asset in enumerate(assets):
                results[asset] = all_weights[:, i]
            
            # Find optimal portfolios
            min_vol_idx = np.argmin(vol_arr)
            max_sharpe_idx = np.argmax(sharpe_arr)
            
            min_vol_portfolio = {
                'Return': ret_arr[min_vol_idx],
                'Volatility': vol_arr[min_vol_idx],
                'Sharpe': sharpe_arr[min_vol_idx],
                'Weights': {asset: all_weights[min_vol_idx, i] for i, asset in enumerate(assets)}
            }
            
            max_sharpe_portfolio = {
                'Return': ret_arr[max_sharpe_idx],
                'Volatility': vol_arr[max_sharpe_idx],
                'Sharpe': sharpe_arr[max_sharpe_idx],
                'Weights': {asset: all_weights[max_sharpe_idx, i] for i, asset in enumerate(assets)}
            }
            
            if save_results:
                # Save results
                results.to_csv(optimization_dir / "efficient_frontier.csv")
                
                # Save optimal portfolios
                with open(optimization_dir / "min_vol_portfolio.json", 'w') as f:
                    json.dump(min_vol_portfolio, f, indent=4)
                
                with open(optimization_dir / "max_sharpe_portfolio.json", 'w') as f:
                    json.dump(max_sharpe_portfolio, f, indent=4)
                
                # Save weights as CSV for easier use in Tableau
                min_vol_weights = pd.DataFrame(list(min_vol_portfolio['Weights'].items()), 
                                              columns=['Asset', 'Weight'])
                min_vol_weights.to_csv(optimization_dir / "min_vol_weights.csv", index=False)
                
                max_sharpe_weights = pd.DataFrame(list(max_sharpe_portfolio['Weights'].items()), 
                                                columns=['Asset', 'Weight'])
                max_sharpe_weights.to_csv(optimization_dir / "max_sharpe_weights.csv", index=False)
                
                # Create visualization
                plt.figure(figsize=(12, 8))
                plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.8)
                plt.colorbar(label='Sharpe Ratio')
                plt.scatter(vol_arr[min_vol_idx], ret_arr[min_vol_idx], c='r', marker='*', s=300, label='Min Volatility')
                plt.scatter(vol_arr[max_sharpe_idx], ret_arr[max_sharpe_idx], c='g', marker='*', s=300, label='Max Sharpe')
                plt.title('Efficient Frontier')
                plt.xlabel('Volatility')
                plt.ylabel('Expected Return')
                plt.legend()
                plt.savefig(optimization_dir / "efficient_frontier.png", dpi=300)
            
            logger.info("Efficient frontier generated successfully.")
            return results, min_vol_portfolio, max_sharpe_portfolio
        
        except Exception as e:
            logger.error(f"Efficient frontier generation failed: {str(e)}", exc_info=True)
            raise
    
    def optimize_sharpe_ratio(self, save_results=True):
        """Optimize portfolio for maximum Sharpe ratio"""
        try:
            logger.info("Optimizing portfolio for maximum Sharpe ratio...")
            
            assets = self.returns.columns
            num_assets = len(assets)
            
            # Initial weights
            init_weights = np.array([1.0 / num_assets] * num_assets)
            
            # Constraints - weights sum to 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Bounds - between 0 and 1
            bounds = tuple((0, 1) for asset in range(num_assets))
            
            # Objective function - negative Sharpe ratio (to maximize)
            def objective(weights):
                portfolio_return = np.sum(weights * self.mean_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
                return -sharpe
            
            # Optimize
            result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            
            # Check if optimization was successful
            if not result['success']:
                logger.warning(f"Optimization failed: {result['message']}")
            
            # Process results
            optimal_weights = result['x']
            optimal_portfolio_return = np.sum(optimal_weights * self.mean_returns)
            optimal_portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
            optimal_sharpe = (optimal_portfolio_return - self.risk_free_rate) / optimal_portfolio_vol
            
            # Create portfolio object
            max_sharpe_portfolio = {
                'Return': float(optimal_portfolio_return),
                'Volatility': float(optimal_portfolio_vol),
                'Sharpe': float(optimal_sharpe),
                'Weights': {asset: float(weight) for asset, weight in zip(assets, optimal_weights)}
            }
            
            if save_results:
                # Save optimal portfolio
                with open(optimization_dir / "max_sharpe_optimized.json", 'w') as f:
                    json.dump(max_sharpe_portfolio, f, indent=4)
                
                # Save weights as CSV for easier use in Tableau
                weights_df = pd.DataFrame(list(max_sharpe_portfolio['Weights'].items()), 
                                        columns=['Asset', 'Weight'])
                weights_df.to_csv(optimization_dir / "max_sharpe_optimized_weights.csv", index=False)
                
                # Create pie chart of weights
                plt.figure(figsize=(12, 8))
                non_zero_weights = {k: v for k, v in max_sharpe_portfolio['Weights'].items() if v > 0.001}
                plt.pie(non_zero_weights.values(), labels=non_zero_weights.keys(), autopct='%1.1f%%')
                plt.title('Max Sharpe Portfolio Allocation')
                plt.savefig(optimization_dir / "max_sharpe_allocation.png", dpi=300)
            
            logger.info(f"Maximum Sharpe ratio optimization completed. Sharpe: {optimal_sharpe:.4f}")
            return max_sharpe_portfolio
        
        except Exception as e:
            logger.error(f"Sharpe ratio optimization failed: {str(e)}", exc_info=True)
            raise
    
    def optimize_minimum_volatility(self, save_results=True):
        """Optimize portfolio for minimum volatility"""
        try:
            logger.info("Optimizing portfolio for minimum volatility...")
            
            assets = self.returns.columns
            num_assets = len(assets)
            
            # Initial weights
            init_weights = np.array([1.0 / num_assets] * num_assets)
            
            # Constraints - weights sum to 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Bounds - between 0 and 1
            bounds = tuple((0, 1) for asset in range(num_assets))
            
            # Objective function - portfolio volatility
            def objective(weights):
                return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Optimize
            result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            
            # Check if optimization was successful
            if not result['success']:
                logger.warning(f"Optimization failed: {result['message']}")
            
            # Process results
            optimal_weights = result['x']
            optimal_portfolio_return = np.sum(optimal_weights * self.mean_returns)
            optimal_portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
            optimal_sharpe = (optimal_portfolio_return - self.risk_free_rate) / optimal_portfolio_vol
            
            # Create portfolio object
            min_vol_portfolio = {
                'Return': float(optimal_portfolio_return),
                'Volatility': float(optimal_portfolio_vol),
                'Sharpe': float(optimal_sharpe),
                'Weights': {asset: float(weight) for asset, weight in zip(assets, optimal_weights)}
            }
            
            if save_results:
                # Save optimal portfolio
                with open(optimization_dir / "min_vol_optimized.json", 'w') as f:
                    json.dump(min_vol_portfolio, f, indent=4)
                
                # Save weights as CSV for easier use in Tableau
                weights_df = pd.DataFrame(list(min_vol_portfolio['Weights'].items()), 
                                        columns=['Asset', 'Weight'])
                weights_df.to_csv(optimization_dir / "min_vol_optimized_weights.csv", index=False)
                
                # Create pie chart of weights
                plt.figure(figsize=(12, 8))
                non_zero_weights = {k: v for k, v in min_vol_portfolio['Weights'].items() if v > 0.001}
                plt.pie(non_zero_weights.values(), labels=non_zero_weights.keys(), autopct='%1.1f%%')
                plt.title('Minimum Volatility Portfolio Allocation')
                plt.savefig(optimization_dir / "min_vol_allocation.png", dpi=300)
            
            logger.info(f"Minimum volatility optimization completed. Volatility: {optimal_portfolio_vol:.4f}")
            return min_vol_portfolio
        
        except Exception as e:
            logger.error(f"Minimum volatility optimization failed: {str(e)}", exc_info=True)
            raise
    
    def optimize_efficient_return(self, target_return, save_results=True):
        """Optimize portfolio for minimum volatility at a target return"""
        try:
            logger.info(f"Optimizing portfolio for target return of {target_return:.4f}...")
            
            assets = self.returns.columns
            num_assets = len(assets)
            
            # Initial weights
            init_weights = np.array([1.0 / num_assets] * num_assets)
            
            # Constraints - weights sum to 1 and portfolio return equals target
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x * self.mean_returns) - target_return}
            )
            
            # Bounds - between 0 and 1
            bounds = tuple((0, 1) for asset in range(num_assets))
            
            # Objective function - portfolio volatility
            def objective(weights):
                return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Optimize
            result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            
            # Check if optimization was successful
            if not result['success']:
                logger.warning(f"Optimization failed: {result['message']}")
                return None
            
            # Process results
            optimal_weights = result['x']
            optimal_portfolio_return = np.sum(optimal_weights * self.mean_returns)
            optimal_portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
            optimal_sharpe = (optimal_portfolio_return - self.risk_free_rate) / optimal_portfolio_vol
            
            # Create portfolio object
            efficient_portfolio = {
                'Target_Return': float(target_return),
                'Return': float(optimal_portfolio_return),
                'Volatility': float(optimal_portfolio_vol),
                'Sharpe': float(optimal_sharpe),
                'Weights': {asset: float(weight) for asset, weight in zip(assets, optimal_weights)}
            }
            
            if save_results:
                # Create directory for efficient portfolios if it doesn't exist
                efficient_dir = optimization_dir / "efficient_portfolios"
                efficient_dir.mkdir(parents=True, exist_ok=True)
                
                # Save optimal portfolio
                with open(efficient_dir / f"efficient_return_{target_return:.4f}.json", 'w') as f:
                    json.dump(efficient_portfolio, f, indent=4)
                
                # Save weights as CSV for easier use in Tableau
                weights_df = pd.DataFrame(list(efficient_portfolio['Weights'].items()), 
                                        columns=['Asset', 'Weight'])
                weights_df['Target_Return'] = target_return
                weights_df.to_csv(efficient_dir / f"efficient_weights_{target_return:.4f}.csv", index=False)
            
            logger.info(f"Efficient return optimization completed. Return: {optimal_portfolio_return:.4f}, Volatility: {optimal_portfolio_vol:.4f}")
            return efficient_portfolio
        
        except Exception as e:
            logger.error(f"Efficient return optimization failed: {str(e)}", exc_info=True)
            raise
    
    def generate_efficient_frontier_curve(self, points=20, save_results=True):
        """Generate the efficient frontier curve with optimized portfolios"""
        try:
            logger.info("Generating efficient frontier curve with optimized portfolios...")
            
            # Get the min and max returns
            min_vol_portfolio = self.optimize_minimum_volatility(save_results=False)
            min_return = min_vol_portfolio['Return']
            
            # Find a reasonable maximum return (slightly below the max asset return to ensure feasibility)
            max_return = self.mean_returns.max() * 0.9
            
            # Generate target returns
            target_returns = np.linspace(min_return, max_return, points)
            
            # List to store efficient portfolios
            efficient_portfolios = []
            
            # Calculate efficient portfolios for each target return
            for target_return in target_returns:
                portfolio = self.optimize_efficient_return(target_return, save_results=False)
                if portfolio:
                    efficient_portfolios.append(portfolio)
            
            # Create DataFrame with results
            efficient_frontier = pd.DataFrame({
                'Return': [p['Return'] for p in efficient_portfolios],
                'Volatility': [p['Volatility'] for p in efficient_portfolios],
                'Sharpe': [p['Sharpe'] for p in efficient_portfolios]
            })
            
            if save_results:
                # Save efficient frontier
                efficient_frontier.to_csv(optimization_dir / "efficient_frontier_curve.csv")
                
                # Save all weights in one file for Tableau
                all_weights = []
                for portfolio in efficient_portfolios:
                    for asset, weight in portfolio['Weights'].items():
                        all_weights.append({
                            'Return': portfolio['Return'],
                            'Asset': asset,
                            'Weight': weight
                        })
                
                weights_df = pd.DataFrame(all_weights)
                weights_df.to_csv(optimization_dir / "all_efficient_weights.csv", index=False)
                
                # Create visualization
                plt.figure(figsize=(12, 8))
                plt.plot(efficient_frontier['Volatility'], efficient_frontier['Return'], 'b-', linewidth=3)
                plt.scatter(min_vol_portfolio['Volatility'], min_vol_portfolio['Return'], 
                          c='r', marker='*', s=300, label='Min Volatility')
                
                # Add max Sharpe portfolio if calculated
                try:
                    max_sharpe_portfolio = self.optimize_sharpe_ratio(save_results=False)
                    plt.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'], 
                               c='g', marker='*', s=300, label='Max Sharpe')
                except:
                    pass
                
                plt.title('Efficient Frontier Curve')
                plt.xlabel('Volatility')
                plt.ylabel('Expected Return')
                plt.legend()
                plt.savefig(optimization_dir / "efficient_frontier_curve.png", dpi=300)
            
            logger.info(f"Efficient frontier curve generated with {len(efficient_portfolios)} portfolios.")
            return efficient_frontier, efficient_portfolios
        
        except Exception as e:
            logger.error(f"Efficient frontier curve generation failed: {str(e)}", exc_info=True)
            raise
    
    def optimize_risk_parity(self, risk_budget=None, save_results=True):
        """Optimize portfolio using risk parity approach"""
        try:
            logger.info("Optimizing portfolio using risk parity approach...")
            
            assets = self.returns.columns
            num_assets = len(assets)
            
            # Equal risk contribution if no budget provided
            if risk_budget is None:
                risk_budget = np.array([1.0 / num_assets] * num_assets)
            else:
                # Normalize risk budget to sum to 1
                risk_budget = np.array(risk_budget) / np.sum(risk_budget)
            
            # Initial weights
            init_weights = np.array([1.0 / num_assets] * num_assets)
            
            # Constraints - weights sum to 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Bounds - between 0 and 1
            bounds = tuple((0.001, 1) for asset in range(num_assets))
            
            # Risk parity objective function
            def risk_contribution(weights):
                # Convert weights to numpy array if not already
                weights_array = np.array(weights)
                
                # Convert covariance matrix to numpy array
                cov_array = np.array(self.cov_matrix)
                
                # Calculate portfolio volatility
                portfolio_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_array, weights_array)))
                
                # Calculate risk contribution for each asset
                asset_rc = []
                for i in range(len(weights_array)):
                    # Use numpy arrays for calculations
                    asset_vol = weights_array[i] * np.sum(cov_array[i] * weights_array) / portfolio_vol
                    asset_rc.append(asset_vol)
                
                return np.array(asset_rc)
            
            # Objective: minimize the sum of squared differences between target and actual risk contributions
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                asset_rc = risk_contribution(weights)
                target_rc = risk_budget * portfolio_vol
                return np.sum(np.square(asset_rc - target_rc))
            
            # Optimize
            result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            
            # Check if optimization was successful
            if not result['success']:
                logger.warning(f"Risk parity optimization failed: {result['message']}")
            
            # Process results
            optimal_weights = result['x']
            optimal_portfolio_return = np.sum(optimal_weights * self.mean_returns)
            optimal_portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
            optimal_sharpe = (optimal_portfolio_return - self.risk_free_rate) / optimal_portfolio_vol
            
            # Calculate actual risk contributions
            actual_rc = risk_contribution(optimal_weights)
            rc_percentage = actual_rc / np.sum(actual_rc)
            
            # Create portfolio object
            risk_parity_portfolio = {
                'Return': float(optimal_portfolio_return),
                'Volatility': float(optimal_portfolio_vol),
                'Sharpe': float(optimal_sharpe),
                'Weights': {asset: float(weight) for asset, weight in zip(assets, optimal_weights)},
                'Risk_Contributions': {asset: float(rc) for asset, rc in zip(assets, rc_percentage)}
            }
            
            if save_results:
                # Save optimal portfolio
                with open(optimization_dir / "risk_parity_portfolio.json", 'w') as f:
                    json.dump(risk_parity_portfolio, f, indent=4)
                
                # Save weights and risk contributions as CSV for easier use in Tableau
                rp_data = []
                for i, asset in enumerate(assets):
                    rp_data.append({
                        'Asset': asset,
                        'Weight': optimal_weights[i],
                        'Risk_Contribution': rc_percentage[i]
                    })
                
                rp_df = pd.DataFrame(rp_data)
                rp_df.to_csv(optimization_dir / "risk_parity_data.csv", index=False)
                
                # Create visualization - compare weights vs risk contributions
                plt.figure(figsize=(12, 8))
                x = np.arange(len(assets))
                width = 0.35
                
                plt.bar(x - width/2, optimal_weights, width, label='Weights')
                plt.bar(x + width/2, rc_percentage, width, label='Risk Contributions')
                plt.xlabel('Assets')
                plt.ylabel('Percentage')
                plt.title('Risk Parity: Weights vs Risk Contributions')
                plt.xticks(x, assets, rotation=90)
                plt.legend()
                plt.tight_layout()
                plt.savefig(optimization_dir / "risk_parity_comparison.png", dpi=300)
            
            logger.info(f"Risk parity optimization completed. Return: {optimal_portfolio_return:.4f}, Volatility: {optimal_portfolio_vol:.4f}")
            return risk_parity_portfolio
        
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {str(e)}", exc_info=True)
            raise
    
    def run_all_optimizations(self):
        """Run all optimization methods and save results"""
        try:
            logger.info("Running all portfolio optimizations...")
            
            # Generate efficient frontier with random portfolios
            self.generate_efficient_frontier(num_portfolios=5000)
            
            # Optimize for maximum Sharpe ratio
            self.optimize_sharpe_ratio()
            
            # Optimize for minimum volatility
            self.optimize_minimum_volatility()
            
            # Generate efficient frontier curve
            self.generate_efficient_frontier_curve(points=30)
            
            # Optimize using risk parity
            self.optimize_risk_parity()
            
            # Generate target return optimizations (from low to high)
            min_vol_portfolio = self.optimize_minimum_volatility(save_results=False)
            min_return = min_vol_portfolio['Return']
            max_return = self.mean_returns.max() * 0.9
            
            target_returns = np.linspace(min_return, max_return, 10)
            for target_return in target_returns:
                self.optimize_efficient_return(target_return)
            
            logger.info("All portfolio optimizations completed successfully.")
            
        except Exception as e:
            logger.error(f"Portfolio optimizations failed: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    optimizer = PortfolioOptimizer()
    optimizer.run_all_optimizations()