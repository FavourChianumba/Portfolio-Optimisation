import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from config.settings import settings
from utils.logger import setup_logger

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Setup directories
data_processed_dir = project_root / settings.DATA_PROCESSED_DIR
factor_dir = data_processed_dir / "factor_model"
factor_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)

class FactorOptimizer:
    """Portfolio optimization using factor models"""
    
    def __init__(self, returns_data=None, factor_data=None, risk_free_rate=None):
        """Initialize with returns data and factor data"""
        self.returns = returns_data
        self.factors = factor_data
        self.risk_free_rate = risk_free_rate if risk_free_rate is not None else settings.RISK_FREE_RATE
        
        # Load data if not provided
        if self.returns is None:
            returns_path = data_processed_dir / "daily_returns.csv"
            self.returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        # Load or generate factors
        if self.factors is None:
            self.factors = self._load_or_generate_factors()
        
        # Store factor exposures (betas)
        self.factor_exposures = None
        self.idiosyncratic_vars = None
        self.factor_returns_mean = None
        self.factor_cov = None
        
        # Calculate factor model parameters
        self._estimate_factor_model()
    
    def _load_or_generate_factors(self):
        """Load existing factors or generate basic ones"""
        # Try to load pre-computed factors
        try:
            factors_path = factor_dir / "factor_returns.csv"
            if factors_path.exists():
                logger.info("Loading pre-computed factor returns")
                factor_returns = pd.read_csv(factors_path, index_col=0, parse_dates=True)
                
                # Check for date overlap
                common_dates = self.returns.index.intersection(factor_returns.index)
                if len(common_dates) > 10:  # At least 10 common dates
                    return factor_returns
                else:
                    logger.warning("Loaded factor returns have insufficient date overlap with asset returns")
                    # Proceed to generate factors
            else:
                logger.info("No pre-computed factor returns found")
        except Exception as e:
            logger.warning(f"Error loading factor returns: {str(e)}")
        
        # Generate basic factors if not available or insufficient overlap
        logger.info("Generating basic factor returns")
        return self._generate_basic_factors()
    
    def _generate_basic_factors(self):
        """Generate basic factors including market, size, value, momentum"""
        # Ensure returns data is available
        if self.returns is None or self.returns.empty:
            raise ValueError("Returns data is required to generate factors")
        
        # Create empty DataFrame for factors
        factor_returns = pd.DataFrame(index=self.returns.index)
        
        # 1. Market factor (cap-weighted market return)
        # Use average of all returns as proxy for market
        factor_returns['MKT'] = self.returns.mean(axis=1)
        
        # 2. Size factor (SMB - Small Minus Big)
        # Approximation: Use smallest 30% vs largest 30% of assets by average price
        if hasattr(self, 'prices') and self.prices is not None:
            avg_prices = self.prices.mean()
            small_assets = avg_prices.nsmallest(int(len(avg_prices) * 0.3)).index
            big_assets = avg_prices.nlargest(int(len(avg_prices) * 0.3)).index
            
            small_returns = self.returns[small_assets].mean(axis=1)
            big_returns = self.returns[big_assets].mean(axis=1)
            factor_returns['SMB'] = small_returns - big_returns
        else:
            # Without price data, use volatility as proxy for size
            vols = self.returns.std()
            small_assets = vols.nlargest(int(len(vols) * 0.3)).index  # Higher vol for smaller companies
            big_assets = vols.nsmallest(int(len(vols) * 0.3)).index
            
            small_returns = self.returns[small_assets].mean(axis=1)
            big_returns = self.returns[big_assets].mean(axis=1)
            factor_returns['SMB'] = small_returns - big_returns
        
        # 3. Momentum factor (MOM)
        # 12-month rolling return, skipping most recent month
        lookback = min(252, int(len(self.returns)/2))  # Use 252 days or half the data
        momentum_scores = self.returns.rolling(window=lookback).mean()
        
        # Identify winners and losers based on momentum scores
        winners = momentum_scores.iloc[-1].nlargest(int(len(momentum_scores.columns) * 0.3)).index
        losers = momentum_scores.iloc[-1].nsmallest(int(len(momentum_scores.columns) * 0.3)).index
        
        # Momentum factor = Winners minus Losers
        factor_returns['MOM'] = self.returns[winners].mean(axis=1) - self.returns[losers].mean(axis=1)
        
        # 4. Volatility factor (VOL)
        # Low volatility minus high volatility
        vols = self.returns.std()
        low_vol_assets = vols.nsmallest(int(len(vols) * 0.3)).index
        high_vol_assets = vols.nlargest(int(len(vols) * 0.3)).index
        
        factor_returns['VOL'] = self.returns[low_vol_assets].mean(axis=1) - self.returns[high_vol_assets].mean(axis=1)
        
        # Save generated factors
        factor_returns.to_csv(factor_dir / "factor_returns.csv")
        return factor_returns
    
    def _estimate_factor_model(self):
        """Estimate factor exposures (betas) and idiosyncratic variance for each asset"""
        try:
            logger.info("Estimating factor model parameters")
            
            # Align factor and returns data
            aligned_data = pd.concat([self.returns, self.factors], axis=1).dropna()
            
            if len(aligned_data) == 0:
                raise ValueError("No overlapping data between returns and factors")
            
            # Split aligned data
            Y = aligned_data[self.returns.columns]
            X = aligned_data[self.factors.columns]
            
            # Add constant for regression
            X = sm.add_constant(X)
            
            # Store factor exposures and idiosyncratic variances
            self.factor_exposures = pd.DataFrame(index=self.returns.columns, columns=['const'] + list(self.factors.columns))
            self.idiosyncratic_vars = pd.Series(index=self.returns.columns)
            
            # Run regression for each asset
            for asset in self.returns.columns:
                try:
                    model = OLS(Y[asset], X).fit()
                    
                    # Store factor exposures (betas)
                    self.factor_exposures.loc[asset] = model.params
                    
                    # Store idiosyncratic variance
                    self.idiosyncratic_vars[asset] = model.mse_resid
                    
                except Exception as e:
                    logger.warning(f"Error estimating factor model for {asset}: {str(e)}")
                    # Set NaN for assets with estimation errors
                    self.factor_exposures.loc[asset] = np.nan
                    self.idiosyncratic_vars[asset] = np.nan
            
            # Remove assets with failed estimations
            valid_assets = ~self.factor_exposures.isna().any(axis=1)
            self.factor_exposures = self.factor_exposures[valid_assets]
            self.idiosyncratic_vars = self.idiosyncratic_vars[valid_assets]
            
            # Calculate factor means and covariance
            self.factor_returns_mean = X.mean()
            self.factor_cov = X.cov()
            
            # Save factor model parameters
            self.factor_exposures.to_csv(factor_dir / "factor_exposures.csv")
            self.idiosyncratic_vars.to_frame('idiosyncratic_variance').to_csv(factor_dir / "idiosyncratic_variances.csv")
            
            logger.info(f"Factor model estimated for {len(self.factor_exposures)} assets")
            
            # Return assets with valid factor exposures
            return list(self.factor_exposures.index)
        
        except Exception as e:
            logger.error(f"Error estimating factor model: {str(e)}", exc_info=True)
            raise
    
    def get_expected_returns(self):
        """Calculate expected returns using the factor model"""
        if self.factor_exposures is None:
            raise ValueError("Factor model not estimated. Run _estimate_factor_model first.")
        
        # Calculate expected returns using factor model
        expected_returns = pd.Series(index=self.factor_exposures.index)
        
        for asset in self.factor_exposures.index:
            # E[r] = alpha + sum(beta_i * E[f_i])
            expected_returns[asset] = np.dot(self.factor_exposures.loc[asset], self.factor_returns_mean)
        
        return expected_returns * 252  # Annualize
    
    def get_factor_covariance_matrix(self):
        """Calculate the covariance matrix using the factor model"""
        if self.factor_exposures is None or self.idiosyncratic_vars is None:
            raise ValueError("Factor model not estimated. Run _estimate_factor_model first.")
        
        assets = self.factor_exposures.index
        n_assets = len(assets)
        
        # Initialize covariance matrix
        cov_matrix = np.zeros((n_assets, n_assets))
        
        # Calculate systematic risk component (factor contribution)
        for i, asset_i in enumerate(assets):
            betas_i = self.factor_exposures.loc[asset_i].values
            
            for j, asset_j in enumerate(assets):
                betas_j = self.factor_exposures.loc[asset_j].values
                
                # Systematic risk: beta_i * factor_cov * beta_j'
                cov_matrix[i, j] = np.dot(np.dot(betas_i, self.factor_cov), betas_j)
                
                # Add idiosyncratic risk only on the diagonal
                if i == j:
                    cov_matrix[i, i] += self.idiosyncratic_vars[asset_j]
        
        # Convert to DataFrame
        cov_df = pd.DataFrame(cov_matrix, index=assets, columns=assets)
        
        return cov_df * 252  # Annualize
    
    def optimize_factor_portfolio(self, objective='sharpe', risk_aversion=1.0, target_return=None, 
                                constraints=None, bounds=None, save_results=True):
        """
        Optimize portfolio based on factor model
        
        Parameters:
        - objective: 'sharpe', 'min_variance', or 'utility'
        - risk_aversion: risk aversion parameter for utility optimization
        - target_return: target portfolio return (for efficient return optimization)
        - constraints: additional constraints as a list of dictionaries
        - bounds: dictionary of bounds for each asset {asset: (lower, upper)}
        - save_results: whether to save results
        """
        try:
            logger.info(f"Optimizing factor-based portfolio with objective: {objective}")
            
            # Get expected returns and covariance matrix from factor model
            expected_returns = self.get_expected_returns()
            cov_matrix = self.get_factor_covariance_matrix()
            
            # Ensure matrix is positive definite (important for optimization stability)
            min_eig = np.min(np.linalg.eigvals(cov_matrix.values))
            if min_eig < 0:
                logger.warning(f"Covariance matrix not positive definite. Min eigenvalue: {min_eig}")
                # Add small diagonal adjustment
                cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * abs(min_eig) * 1.1
            
            # Get assets from valid factor model estimates
            assets = cov_matrix.index.tolist()
            n_assets = len(assets)
            
            # Setup default bounds
            if bounds is None:
                bounds = {}
                min_weight = settings.MIN_WEIGHT
                max_weight = settings.MAX_WEIGHT
                
                # Default bounds for all assets
                for asset in assets:
                    bounds[asset] = (min_weight, max_weight)
            
            # Convert bounds to list for optimization
            bounds_list = [bounds.get(asset, (0, 1)) for asset in assets]
            
            # Initial weights (equal weight)
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Setup constraints
            if constraints is None:
                constraints = []
            
            # Add weights sum to 1 constraint
            constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Add target return constraint if specified
            if objective == 'efficient_return' and target_return is not None:
                target_return_constraint = {
                    'type': 'eq',
                    'fun': lambda x: np.sum(x * expected_returns) - target_return
                }
                constraints.append(target_return_constraint)
            
            # Define objective function based on selected objective
            if objective == 'min_variance' or (objective == 'efficient_return' and target_return is not None):
                # Minimize variance
                def objective_function(weights):
                    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                    return portfolio_variance
            
            elif objective == 'sharpe':
                # Maximize Sharpe ratio (minimize negative Sharpe)
                def objective_function(weights):
                    portfolio_return = np.sum(weights * expected_returns)
                    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                    return -sharpe_ratio  # Minimize negative Sharpe
            
            elif objective == 'utility':
                # Maximize mean-variance utility
                def objective_function(weights):
                    portfolio_return = np.sum(weights * expected_returns)
                    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                    utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
                    return -utility  # Minimize negative utility
            
            else:
                raise ValueError(f"Invalid objective: {objective}")
            
            # Run optimization
            optimization_result = minimize(
                fun=objective_function,
                x0=initial_weights,
                method='SLSQP',
                bounds=bounds_list,
                constraints=constraints
            )
            
            # Check if optimization succeeded
            if not optimization_result['success']:
                logger.warning(f"Optimization failed: {optimization_result['message']}")
            
            # Extract optimal weights
            optimal_weights = optimization_result['x']
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(optimal_weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Calculate factor exposures of the portfolio
            portfolio_factor_exposures = pd.Series(
                np.dot(optimal_weights, self.factor_exposures.values),
                index=self.factor_exposures.columns
            )
            
            # Create result object
            result = {
                'objective': objective,
                'return': float(portfolio_return),
                'volatility': float(portfolio_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'weights': {asset: float(weight) for asset, weight in zip(assets, optimal_weights)},
                'factor_exposures': portfolio_factor_exposures.to_dict()
            }
            
            # Save results
            if save_results:
                # Save optimal weights
                weights_df = pd.DataFrame({
                    'Asset': assets,
                    'Weight': optimal_weights
                })
                weights_df.to_csv(factor_dir / f"factor_portfolio_{objective}_weights.csv", index=False)
                
                # Save portfolio metrics
                with open(factor_dir / f"factor_portfolio_{objective}_metrics.json", 'w') as f:
                    json.dump(result, f, indent=4)
                
                # Create visualization of weights
                plt.figure(figsize=(12, 8))
                # Only plot weights above threshold
                threshold = 0.01  # 1%
                plot_weights = {k: v for k, v in zip(assets, optimal_weights) if v > threshold}
                plt.pie(plot_weights.values(), labels=plot_weights.keys(), autopct='%1.1f%%')
                plt.title(f'Factor Model Portfolio Allocation ({objective.capitalize()})')
                plt.savefig(factor_dir / f"factor_portfolio_{objective}_allocation.png", dpi=300)
                plt.close()
                
                # Create visualization of factor exposures
                plt.figure(figsize=(12, 8))
                factor_names = [col for col in portfolio_factor_exposures.index if col != 'const']
                factor_values = [portfolio_factor_exposures[factor] for factor in factor_names]
                plt.bar(factor_names, factor_values)
                plt.title('Portfolio Factor Exposures')
                plt.ylabel('Exposure (Beta)')
                plt.axhline(y=0, color='r', linestyle='-')
                plt.tight_layout()
                plt.savefig(factor_dir / f"factor_portfolio_{objective}_exposures.png", dpi=300)
                plt.close()
            
            logger.info(f"Factor portfolio optimization completed. Return: {portfolio_return:.4f}, Volatility: {portfolio_volatility:.4f}, Sharpe: {sharpe_ratio:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Factor portfolio optimization failed: {str(e)}", exc_info=True)
            raise
    
    def optimize_factor_tilted_portfolio(self, target_factor_exposures, objective='min_tracking_error', 
                                        benchmark_weights=None, save_results=True):
        """
        Optimize a portfolio with specific factor tilts
        
        Parameters:
        - target_factor_exposures: dictionary of {factor: target_exposure}
        - objective: 'min_tracking_error' or 'max_return'
        - benchmark_weights: dictionary of benchmark weights for tracking error
        - save_results: whether to save results
        """
        try:
            logger.info(f"Optimizing factor-tilted portfolio with {objective} objective")
            
            # Get expected returns and covariance matrix from factor model
            expected_returns = self.get_expected_returns()
            cov_matrix = self.get_factor_covariance_matrix()
            
            # Get assets from valid factor model estimates
            assets = cov_matrix.index.tolist()
            n_assets = len(assets)
            
            # Create benchmark weights if not provided (equal weight)
            if benchmark_weights is None:
                benchmark_weights = {asset: 1.0/n_assets for asset in assets}
            
            # Convert benchmark weights to array
            benchmark_array = np.array([benchmark_weights.get(asset, 0) for asset in assets])
            
            # Setup bounds
            min_weight = settings.MIN_WEIGHT
            max_weight = settings.MAX_WEIGHT
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
            
            # Initial weights (equal weight)
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Setup constraints
            constraints = []
            
            # Add weights sum to 1 constraint
            constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Add factor exposure constraints
            for factor, target in target_factor_exposures.items():
                if factor in self.factor_exposures.columns:
                    # Get factor exposures for each asset
                    factor_betas = self.factor_exposures[factor].values
                    
                    # Create constraint: weighted sum of betas = target
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda x, betas=factor_betas, t=target: np.dot(x, betas) - t
                    })
                else:
                    logger.warning(f"Factor {factor} not found in factor model. Ignoring constraint.")
            
            # Define objective function based on selected objective
            if objective == 'min_tracking_error':
                # Minimize tracking error to benchmark
                def objective_function(weights):
                    # Tracking error = sqrt[(w-b)'Σ(w-b)]
                    tracking_error = np.sqrt(np.dot((weights - benchmark_array).T, 
                                                   np.dot(cov_matrix, (weights - benchmark_array))))
                    return tracking_error
            
            elif objective == 'max_return':
                # Maximize expected return
                def objective_function(weights):
                    portfolio_return = np.sum(weights * expected_returns)
                    return -portfolio_return  # Minimize negative return
            
            elif objective == 'min_variance':
                # Minimize portfolio variance
                def objective_function(weights):
                    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                    return portfolio_variance
            
            else:
                raise ValueError(f"Invalid objective: {objective}")
            
            # Run optimization
            optimization_result = minimize(
                fun=objective_function,
                x0=initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Check if optimization succeeded
            if not optimization_result['success']:
                logger.warning(f"Optimization failed: {optimization_result['message']}")
            
            # Extract optimal weights
            optimal_weights = optimization_result['x']
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(optimal_weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Calculate tracking error if relevant
            if objective == 'min_tracking_error':
                tracking_error = np.sqrt(np.dot((optimal_weights - benchmark_array).T, 
                                             np.dot(cov_matrix, (optimal_weights - benchmark_array))))
            else:
                tracking_error = None
            
            # Calculate actual factor exposures of the portfolio
            portfolio_factor_exposures = pd.Series(
                np.dot(optimal_weights, self.factor_exposures.values),
                index=self.factor_exposures.columns
            )
            
            # Create result object
            result = {
                'objective': objective,
                'return': float(portfolio_return),
                'volatility': float(portfolio_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'tracking_error': float(tracking_error) if tracking_error is not None else None,
                'weights': {asset: float(weight) for asset, weight in zip(assets, optimal_weights)},
                'factor_exposures': portfolio_factor_exposures.to_dict(),
                'target_factor_exposures': target_factor_exposures
            }
            
            # Save results
            if save_results:
                # Create directory name based on factor tilts
                tilt_name = "_".join([f"{factor}_{value:.2f}" for factor, value in target_factor_exposures.items()])
                
                # Save optimal weights
                weights_df = pd.DataFrame({
                    'Asset': assets,
                    'Weight': optimal_weights
                })
                weights_df.to_csv(factor_dir / f"factor_tilt_{tilt_name}_weights.csv", index=False)
                
                # Save portfolio metrics
                with open(factor_dir / f"factor_tilt_{tilt_name}_metrics.json", 'w') as f:
                    json.dump(result, f, indent=4)
                
                # Create visualization of weights
                plt.figure(figsize=(12, 8))
                # Only plot weights above threshold
                threshold = 0.01  # 1%
                plot_weights = {k: v for k, v in zip(assets, optimal_weights) if v > threshold}
                plt.pie(plot_weights.values(), labels=plot_weights.keys(), autopct='%1.1f%%')
                plt.title(f'Factor-Tilted Portfolio Allocation ({objective.capitalize()})')
                plt.savefig(factor_dir / f"factor_tilt_{tilt_name}_allocation.png", dpi=300)
                plt.close()
                
                # Create visualization of factor exposures vs targets
                plt.figure(figsize=(12, 8))
                
                # Only include non-constant factors
                factor_names = [col for col in portfolio_factor_exposures.index if col != 'const']
                factor_values = [portfolio_factor_exposures[factor] for factor in factor_names]
                
                # Plot actual exposures
                plt.bar(factor_names, factor_values, alpha=0.7, label='Actual')
                
                # Plot target exposures for specified factors
                target_factors = []
                target_values = []
                for factor in factor_names:
                    if factor in target_factor_exposures:
                        target_factors.append(factor)
                        target_values.append(target_factor_exposures[factor])
                
                if target_factors:
                    # Create a second bar chart with targets
                    bar_positions = [factor_names.index(factor) for factor in target_factors]
                    plt.bar([factor_names[pos] for pos in bar_positions], target_values, 
                           alpha=0.5, color='red', label='Target')
                
                plt.title('Portfolio Factor Exposures vs Targets')
                plt.ylabel('Exposure (Beta)')
                plt.axhline(y=0, color='gray', linestyle='--')
                plt.legend()
                plt.tight_layout()
                plt.savefig(factor_dir / f"factor_tilt_{tilt_name}_exposures.png", dpi=300)
                plt.close()
                
                # Create visualization comparing to benchmark
                if benchmark_weights is not None:
                    plt.figure(figsize=(14, 8))
                    
                    # Create a DataFrame for comparison
                    comparison = pd.DataFrame({
                        'Optimized': pd.Series({asset: weight for asset, weight in zip(assets, optimal_weights)}),
                        'Benchmark': pd.Series(benchmark_weights)
                    })
                    
                    # Sort by optimized weights
                    comparison = comparison.sort_values('Optimized', ascending=False)
                    
                    # Only show top assets
                    top_n = min(15, len(comparison))
                    comparison.iloc[:top_n].plot(kind='bar')
                    
                    plt.title(f'Factor-Tilted Portfolio vs Benchmark Weights (Top {top_n} Assets)')
                    plt.ylabel('Weight')
                    plt.tight_layout()
                    plt.savefig(factor_dir / f"factor_tilt_{tilt_name}_vs_benchmark.png", dpi=300)
                    plt.close()
            
            logger.info(f"Factor-tilted portfolio optimization completed.")
            return result
            
        except Exception as e:
            logger.error(f"Factor-tilted portfolio optimization failed: {str(e)}", exc_info=True)
            raise
    
    def analyze_portfolio_factor_exposures(self, weights, save_results=True):
        """
        Analyze the factor exposures of an existing portfolio
        
        Parameters:
        - weights: dictionary of {asset: weight}
        - save_results: whether to save results
        """
        try:
            logger.info("Analyzing portfolio factor exposures")
            
            # Convert weights to series
            weights_series = pd.Series(weights)
            
            # Only use assets that have factor exposures
            common_assets = set(weights_series.index).intersection(set(self.factor_exposures.index))
            
            if len(common_assets) < len(weights_series) * 0.5:
                logger.warning(f"Only {len(common_assets)}/{len(weights_series)} assets have factor exposures")
            
            # Filter weights and normalize
            filtered_weights = weights_series[list(common_assets)]
            normalized_weights = filtered_weights / filtered_weights.sum()
            
            # Calculate portfolio factor exposures
            portfolio_factor_exposures = pd.Series(
                np.dot(normalized_weights, self.factor_exposures.loc[common_assets].values),
                index=self.factor_exposures.columns
            )
            
            # Calculate expected return and risk based on factor model
            expected_returns = self.get_expected_returns()
            cov_matrix = self.get_factor_covariance_matrix()
            
            # Filter for common assets
            filtered_returns = expected_returns.loc[common_assets]
            filtered_cov = cov_matrix.loc[common_assets, common_assets]
            
            portfolio_return = np.sum(normalized_weights * filtered_returns)
            portfolio_volatility = np.sqrt(np.dot(normalized_weights.T, np.dot(filtered_cov, normalized_weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Prepare results
            result = {
                'return': float(portfolio_return),
                'volatility': float(portfolio_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'weights': {asset: float(weight) for asset, weight in normalized_weights.items()},
                'factor_exposures': portfolio_factor_exposures.to_dict()
            }
            
            # Save results
            if save_results:
                # Generate a unique name based on timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save analysis
                with open(factor_dir / f"portfolio_factor_analysis_{timestamp}.json", 'w') as f:
                    json.dump(result, f, indent=4)
                
                # Create visualization of factor exposures
                plt.figure(figsize=(12, 8))
                
                # Only include non-constant factors
                factor_names = [col for col in portfolio_factor_exposures.index if col != 'const']
                factor_values = [portfolio_factor_exposures[factor] for factor in factor_names]
                
                plt.bar(factor_names, factor_values)
                plt.title('Portfolio Factor Exposures')
                plt.ylabel('Exposure (Beta)')
                plt.axhline(y=0, color='r', linestyle='-')
                plt.tight_layout()
                plt.savefig(factor_dir / f"portfolio_factor_analysis_{timestamp}.png", dpi=300)
                plt.close()
            
            logger.info(f"Portfolio factor analysis completed. Return: {portfolio_return:.4f}, Volatility: {portfolio_volatility:.4f}, Sharpe: {sharpe_ratio:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Portfolio factor analysis failed: {str(e)}", exc_info=True)
            raise
    
    def run_all_optimizations(self):
        """Run all factor-based optimizations"""
        try:
            logger.info("Running all factor-based optimizations")
            
            # 1. Standard optimizations
            self.optimize_factor_portfolio(objective='sharpe')
            self.optimize_factor_portfolio(objective='min_variance')
            
            # 2. Factor-tilted portfolios for different investment styles
            
            # Value tilt (negative exposure to Momentum, positive to Value)
            self.optimize_factor_tilted_portfolio(
                target_factor_exposures={'MOM': -0.2},
                objective='max_return'
            )
            
            # Momentum tilt
            self.optimize_factor_tilted_portfolio(
                target_factor_exposures={'MOM': 0.2},
                objective='max_return'
            )
            
            # Low volatility tilt
            self.optimize_factor_tilted_portfolio(
                target_factor_exposures={'VOL': 0.2},
                objective='min_variance'
            )
            
            # Multi-factor tilt
            self.optimize_factor_tilted_portfolio(
                target_factor_exposures={'MKT': 1.0, 'SMB': 0.2, 'MOM': 0.2, 'VOL': 0.2},
                objective='min_tracking_error'
            )
            
            logger.info("All factor-based optimizations completed successfully.")
            
            return True
            
        except Exception as e:
            logger.error(f"Factor optimizations failed: {str(e)}", exc_info=True)
            return False

if __name__ == "__main__":
    optimizer = FactorOptimizer()
    optimizer.run_all_optimizations()