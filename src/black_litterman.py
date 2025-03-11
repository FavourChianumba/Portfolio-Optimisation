import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from scipy.optimize import minimize
from config.settings import settings
from utils.logger import setup_logger

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Setup directories
data_processed_dir = project_root / settings.DATA_PROCESSED_DIR
bl_dir = data_processed_dir / "black_litterman"
bl_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)

class BlackLittermanOptimizer:
    """Portfolio optimization using the Black-Litterman model"""
    
    def __init__(self, returns_data=None, market_weights=None, risk_free_rate=None, risk_aversion=2.5, tau=0.025):
        """
        Initialize the Black-Litterman model
        
        Parameters:
        - returns_data: DataFrame of asset returns
        - market_weights: Dictionary of market capitalization weights
        - risk_free_rate: Risk-free rate (annual)
        - risk_aversion: Risk aversion parameter (lambda)
        - tau: Uncertainty parameter for prior estimates
        """
        self.returns = returns_data
        self.market_weights = market_weights
        self.risk_free_rate = risk_free_rate if risk_free_rate is not None else settings.RISK_FREE_RATE
        self.risk_aversion = risk_aversion
        self.tau = tau
        
        # Load data if not provided
        if self.returns is None:
            returns_path = data_processed_dir / "daily_returns.csv"
            self.returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        # Generate market weights if not provided (equal weight as fallback)
        if self.market_weights is None:
            self.market_weights = self._estimate_market_weights()
        
        # Calculate prior parameters
        self.cov_matrix = self._calculate_covariance_matrix()
        self.equilibrium_returns = self._calculate_equilibrium_returns()
        
        # Initialize views
        self.views = None
        self.pick_matrix = None
        self.view_confidences = None
        
        # Final estimates
        self.posterior_returns = None
        self.posterior_cov = None
    
    def _estimate_market_weights(self):
        """Estimate market weights based on equal weighting"""
        logger.info("Estimating market weights (equal weight)")
        assets = self.returns.columns
        weights = {asset: 1.0 / len(assets) for asset in assets}
        return weights
    
    def _calculate_covariance_matrix(self):
        """Calculate the covariance matrix from historical returns"""
        logger.info("Calculating covariance matrix")
        
        # Calculate sample covariance matrix (annualized)
        cov_matrix = self.returns.cov() * 252
        
        # Apply shrinkage or other adjustments if needed
        # For simplicity, we're using the sample covariance matrix directly
        
        return cov_matrix
    
    def _calculate_equilibrium_returns(self):
        """
        Calculate implied equilibrium excess returns using reverse optimization
        
        Formula: π = λΣw_mkt
        where:
        - π is the vector of implied excess returns
        - λ is the risk aversion coefficient
        - Σ is the covariance matrix
        - w_mkt is the vector of market weights
        """
        logger.info("Calculating equilibrium returns")
        
        # Get assets from covariance matrix
        assets = self.cov_matrix.index
        
        # Convert market weights to array in the same order as assets
        mkt_weights = np.array([self.market_weights.get(asset, 0) for asset in assets])
        
        # Normalize weights to sum to 1
        if np.sum(mkt_weights) == 0:
            logger.warning("Market weights sum to zero. Using equal weights.")
            mkt_weights = np.ones(len(assets)) / len(assets)
        else:
            mkt_weights = mkt_weights / np.sum(mkt_weights)
        
        # Calculate implied returns: π = λΣw
        implied_returns = self.risk_aversion * self.cov_matrix.values @ mkt_weights
        
        # Convert to DataFrame
        return pd.Series(implied_returns, index=assets)
    
    def add_absolute_view(self, asset, return_view, confidence):
        """
        Add an absolute view on an asset's expected return
        
        Parameters:
        - asset: Asset name
        - return_view: Expected return (annualized)
        - confidence: Confidence level in the view (0-1)
        """
        if not hasattr(self, 'absolute_views'):
            self.absolute_views = []
        
        self.absolute_views.append({
            'type': 'absolute',
            'asset': asset,
            'return': return_view,
            'confidence': confidence
        })
        
        logger.info(f"Added absolute view: {asset} return = {return_view:.2%} (confidence: {confidence:.2f})")
        return self
    
    def add_relative_view(self, asset1, asset2, return_diff, confidence):
        """
        Add a relative view between two assets
        
        Parameters:
        - asset1: First asset name
        - asset2: Second asset name
        - return_diff: Expected return difference (asset1 - asset2, annualized)
        - confidence: Confidence level in the view (0-1)
        """
        if not hasattr(self, 'relative_views'):
            self.relative_views = []
        
        self.relative_views.append({
            'type': 'relative',
            'asset1': asset1,
            'asset2': asset2,
            'return_diff': return_diff,
            'confidence': confidence
        })
        
        logger.info(f"Added relative view: {asset1} - {asset2} = {return_diff:.2%} (confidence: {confidence:.2f})")
        return self
    
    def _prepare_views(self):
        """Prepare views in matrix form for Black-Litterman model"""
        # Get list of assets
        assets = self.cov_matrix.index
        n_assets = len(assets)
        
        # Initialize views arrays
        absolute_views = getattr(self, 'absolute_views', [])
        relative_views = getattr(self, 'relative_views', [])
        
        # Total number of views
        k = len(absolute_views) + len(relative_views)
        
        if k == 0:
            logger.warning("No views provided. Using prior (equilibrium) returns.")
            self.views = None
            self.pick_matrix = None
            self.view_confidences = None
            return
        
        # Initialize view matrices
        self.pick_matrix = np.zeros((k, n_assets))
        self.views = np.zeros(k)
        self.view_confidences = np.zeros(k)
        
        # Process views
        view_index = 0
        
        # Process absolute views
        for view in absolute_views:
            asset = view['asset']
            if asset in assets:
                asset_idx = assets.get_loc(asset)
                self.pick_matrix[view_index, asset_idx] = 1
                self.views[view_index] = view['return']
                self.view_confidences[view_index] = view['confidence']
                view_index += 1
            else:
                logger.warning(f"Asset {asset} not found in returns data. Ignoring view.")
        
        # Process relative views
        for view in relative_views:
            asset1, asset2 = view['asset1'], view['asset2']
            if asset1 in assets and asset2 in assets:
                asset1_idx = assets.get_loc(asset1)
                asset2_idx = assets.get_loc(asset2)
                self.pick_matrix[view_index, asset1_idx] = 1
                self.pick_matrix[view_index, asset2_idx] = -1
                self.views[view_index] = view['return_diff']
                self.view_confidences[view_index] = view['confidence']
                view_index += 1
            else:
                missing = []
                if asset1 not in assets:
                    missing.append(asset1)
                if asset2 not in assets:
                    missing.append(asset2)
                logger.warning(f"Assets {', '.join(missing)} not found in returns data. Ignoring view.")
        
        # If we have fewer valid views than expected, trim matrices
        if view_index < k:
            self.pick_matrix = self.pick_matrix[:view_index]
            self.views = self.views[:view_index]
            self.view_confidences = self.view_confidences[:view_index]
            
            # Update k
            k = view_index
        
        # Convert confidence to uncertainty (omega) matrix
        # Higher confidence = lower uncertainty
        self.omega = np.zeros((k, k))
        
        for i in range(k):
            if self.view_confidences[i] > 0:
                # Scale the uncertainty inversely with confidence
                # 1.0 confidence = very low uncertainty, 0.0 confidence = high uncertainty
                uncertainty = (1.0 / self.view_confidences[i] - 1.0) * 0.1
                
                # Uncertainty is also proportional to the variance of the portfolio in the view
                view_portfolio_var = self.pick_matrix[i] @ self.cov_matrix.values @ self.pick_matrix[i].T
                
                # Set the diagonal element
                self.omega[i, i] = max(uncertainty * view_portfolio_var, 1e-8)
            else:
                # Extremely high uncertainty for zero confidence
                self.omega[i, i] = 1e3
    
    def compute_posterior(self):
        """
        Compute posterior expected returns and covariance using Black-Litterman model
        
        Black-Litterman formula:
        E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)π + P'Ω^(-1)Q]
        
        where:
        - E[R] is the posterior expected returns
        - τ is a scalar representing the uncertainty of the prior
        - Σ is the covariance matrix
        - P is the pick matrix defining the views
        - Ω is the uncertainty of the views
        - π is the equilibrium returns
        - Q is the view returns
        """
        logger.info("Computing Black-Litterman posterior estimates")
        
        # Prepare views if needed
        if self.views is None:
            self._prepare_views()
        
        # If still no views, use prior
        if self.views is None or len(self.views) == 0:
            logger.info("No valid views. Using prior returns.")
            self.posterior_returns = self.equilibrium_returns
            self.posterior_cov = self.cov_matrix
            return self
        
        # Convert Series to numpy array
        prior_returns = self.equilibrium_returns.values
        cov_matrix = self.cov_matrix.values
        
        # Calculate posterior expected returns
        # Step 1: Calculate precision of prior and views
        prior_precision = np.linalg.inv(self.tau * cov_matrix)
        view_precision = np.linalg.inv(self.omega)
        
        # Step 2: Calculate posterior precision
        posterior_precision = prior_precision + self.pick_matrix.T @ view_precision @ self.pick_matrix
        
        # Step 3: Calculate posterior covariance
        posterior_cov_matrix = np.linalg.inv(posterior_precision)
        
        # Step 4: Calculate posterior expected returns
        posterior_returns = posterior_cov_matrix @ (
            prior_precision @ prior_returns + 
            self.pick_matrix.T @ view_precision @ self.views
        )
        
        # Convert to Series/DataFrame
        self.posterior_returns = pd.Series(posterior_returns, index=self.cov_matrix.index)
        self.posterior_cov = pd.DataFrame(posterior_cov_matrix, 
                                         index=self.cov_matrix.index, 
                                         columns=self.cov_matrix.columns)
        
        logger.info("Black-Litterman posterior estimates computed successfully")
        return self
    
    def optimize_portfolio(self, objective='sharpe', risk_aversion=None, save_results=True):
        """
        Optimize portfolio using Black-Litterman posterior estimates
        
        Parameters:
        - objective: 'sharpe', 'min_variance', or 'utility'
        - risk_aversion: Risk aversion parameter for utility (defaults to self.risk_aversion)
        - save_results: Whether to save results
        
        Returns:
        - Dictionary with optimization results
        """
        try:
            logger.info(f"Optimizing Black-Litterman portfolio with {objective} objective")
            
            # Ensure posterior estimates are computed
            if self.posterior_returns is None:
                self.compute_posterior()
            
            # Set risk aversion
            if risk_aversion is None:
                risk_aversion = self.risk_aversion
            
            # Get assets from returns
            assets = self.posterior_returns.index
            n_assets = len(assets)
            
            # Setup bounds
            min_weight = settings.MIN_WEIGHT
            max_weight = settings.MAX_WEIGHT
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
            
            # Initial weights (equal weight)
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Constraints - weights sum to 1
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Define objective function based on selected objective
            if objective == 'sharpe':
                # Maximize Sharpe ratio (minimize negative Sharpe)
                def objective_function(weights):
                    portfolio_return = np.sum(weights * self.posterior_returns)
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.posterior_cov, weights)))
                    return -(portfolio_return - self.risk_free_rate) / portfolio_vol
            
            elif objective == 'min_variance':
                # Minimize portfolio variance
                def objective_function(weights):
                    return np.dot(weights.T, np.dot(self.posterior_cov, weights))
            
            elif objective == 'utility':
                # Maximize mean-variance utility
                def objective_function(weights):
                    portfolio_return = np.sum(weights * self.posterior_returns)
                    portfolio_variance = np.dot(weights.T, np.dot(self.posterior_cov, weights))
                    utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
                    return -utility  # Minimize negative utility
            
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
            portfolio_return = np.sum(optimal_weights * self.posterior_returns)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(self.posterior_cov, optimal_weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Create result object
            result = {
                'objective': objective,
                'return': float(portfolio_return),
                'volatility': float(portfolio_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'weights': {asset: float(weight) for asset, weight in zip(assets, optimal_weights)}
            }
            
            # Save results
            if save_results:
                # Create optimization description
                if hasattr(self, 'absolute_views') or hasattr(self, 'relative_views'):
                    absolute_count = len(getattr(self, 'absolute_views', []))
                    relative_count = len(getattr(self, 'relative_views', []))
                    desc = f"{absolute_count}abs_{relative_count}rel"
                else:
                    desc = "no_views"
                
                # Save optimal weights
                weights_df = pd.DataFrame({
                    'Asset': assets,
                    'Weight': optimal_weights,
                    'Expected Return': self.posterior_returns.values
                })
                weights_df.to_csv(bl_dir / f"bl_{objective}_{desc}_weights.csv", index=False)
                
                # Save portfolio metrics
                with open(bl_dir / f"bl_{objective}_{desc}_metrics.json", 'w') as f:
                    json.dump(result, f, indent=4)
                
                # Create visualization of weights
                plt.figure(figsize=(12, 8))
                # Only plot weights above threshold
                threshold = 0.01  # 1%
                plot_weights = {k: v for k, v in zip(assets, optimal_weights) if v > threshold}
                plt.pie(plot_weights.values(), labels=plot_weights.keys(), autopct='%1.1f%%')
                plt.title(f'Black-Litterman Portfolio Allocation ({objective.capitalize()})')
                plt.savefig(bl_dir / f"bl_{objective}_{desc}_allocation.png", dpi=300)
                plt.close()
                
                # Compare prior and posterior expected returns
                plt.figure(figsize=(14, 8))
                compare_df = pd.DataFrame({
                    'Prior': self.equilibrium_returns,
                    'Posterior': self.posterior_returns
                })
                # Sort by posterior expected return
                compare_df = compare_df.sort_values('Posterior', ascending=False)
                # Only show top assets
                top_n = min(15, len(compare_df))
                compare_df.head(top_n).plot(kind='bar')
                plt.title('Prior vs Posterior Expected Returns (Top Assets)')
                plt.ylabel('Expected Return')
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.savefig(bl_dir / f"bl_{objective}_{desc}_returns_comparison.png", dpi=300)
                plt.close()
                
                # Compare with market weights
                plt.figure(figsize=(14, 8))
                market_weights_series = pd.Series({asset: self.market_weights.get(asset, 0) for asset in assets})
                optimal_weights_series = pd.Series({asset: weight for asset, weight in zip(assets, optimal_weights)})
                
                # Combine and sort by optimal weights
                weights_compare = pd.DataFrame({
                    'Black-Litterman': optimal_weights_series,
                    'Market': market_weights_series
                })
                weights_compare = weights_compare.sort_values('Black-Litterman', ascending=False)
                # Only show top assets
                weights_compare.head(top_n).plot(kind='bar')
                plt.title('Black-Litterman vs Market Weights (Top Assets)')
                plt.ylabel('Weight')
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.savefig(bl_dir / f"bl_{objective}_{desc}_market_comparison.png", dpi=300)
                plt.close()
            
            logger.info(f"Black-Litterman optimization completed. Return: {portfolio_return:.4f}, Volatility: {portfolio_volatility:.4f}, Sharpe: {sharpe_ratio:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {str(e)}", exc_info=True)
            raise
    
    def compare_prior_posterior(self, save_results=True):
        """Compare prior and posterior estimates and analyze the impact of views"""
        try:
            logger.info("Comparing prior and posterior estimates")
            
            # Ensure posterior estimates are computed
            if self.posterior_returns is None:
                self.compute_posterior()
            
            # Compare expected returns
            return_diff = self.posterior_returns - self.equilibrium_returns
            
            # Sort by the magnitude of change
            return_diff = return_diff.sort_values(ascending=False)
            
            # Calculate percentage change
            pct_change = (self.posterior_returns / self.equilibrium_returns - 1) * 100
            pct_change = pct_change.sort_values(ascending=False)
            
            # Prepare comparison DataFrame
            comparison = pd.DataFrame({
                'Prior': self.equilibrium_returns,
                'Posterior': self.posterior_returns,
                'Difference': return_diff,
                'Pct_Change': pct_change
            })
            
            # Identify assets with largest changes
            largest_inc = comparison.nlargest(5, 'Difference')
            largest_dec = comparison.nsmallest(5, 'Difference')
            
            logger.info("Assets with largest increases in expected returns:")
            for asset, row in largest_inc.iterrows():
                logger.info(f"  {asset}: {row['Prior']:.4f} -> {row['Posterior']:.4f} ({row['Pct_Change']:.2f}%)")
            
            logger.info("Assets with largest decreases in expected returns:")
            for asset, row in largest_dec.iterrows():
                logger.info(f"  {asset}: {row['Prior']:.4f} -> {row['Posterior']:.4f} ({row['Pct_Change']:.2f}%)")
            
            # Save results
            if save_results:
                # Create description
                if hasattr(self, 'absolute_views') or hasattr(self, 'relative_views'):
                    absolute_count = len(getattr(self, 'absolute_views', []))
                    relative_count = len(getattr(self, 'relative_views', []))
                    desc = f"{absolute_count}abs_{relative_count}rel"
                else:
                    desc = "no_views"
                
                # Save comparison
                comparison.to_csv(bl_dir / f"bl_comparison_{desc}.csv")
                
                # Visualize changes in expected returns
                plt.figure(figsize=(14, 10))
                
                # Sort by difference and plot top and bottom 10
                top_bottom = pd.concat([return_diff.head(10), return_diff.tail(10)])
                top_bottom = top_bottom.sort_values(ascending=False)
                
                plt.barh(top_bottom.index, top_bottom.values, color=['g' if x > 0 else 'r' for x in top_bottom.values])
                plt.title('Change in Expected Returns (Posterior - Prior)')
                plt.xlabel('Difference in Expected Return')
                plt.grid(True, axis='x')
                plt.tight_layout()
                plt.savefig(bl_dir / f"bl_return_changes_{desc}.png", dpi=300)
                plt.close()
                
                # Visualize correlation changes
                plt.figure(figsize=(14, 12))
                
                # Convert correlation matrices to make them comparable
                prior_corr = self.cov_matrix.corr()
                posterior_corr = self.posterior_cov.corr()
                
                # Calculate correlation differences
                corr_diff = posterior_corr - prior_corr
                
                # Create heatmap of correlation differences
                plt.imshow(corr_diff.values, cmap='coolwarm', vmin=-0.2, vmax=0.2)
                plt.colorbar(label='Correlation Difference')
                plt.title('Change in Correlations (Posterior - Prior)')
                
                # Add asset labels
                plt.xticks(range(len(corr_diff.columns)), corr_diff.columns, rotation=90)
                plt.yticks(range(len(corr_diff.index)), corr_diff.index)
                
                plt.tight_layout()
                plt.savefig(bl_dir / f"bl_correlation_changes_{desc}.png", dpi=300)
                plt.close()
            
            logger.info(f"Comparison of prior and posterior estimates completed")
            return comparison
            
        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}", exc_info=True)
            raise
    
    def run_example_optimizations(self):
        """Run example optimizations with different views"""
        try:
            logger.info("Running example Black-Litterman optimizations")
            
            # Example 1: No views (baseline)
            self.absolute_views = []
            self.relative_views = []
            self.compute_posterior()
            self.optimize_portfolio(objective='sharpe')
            
            # Get assets for views
            assets = list(self.cov_matrix.index)
            
            # Example 2: Single absolute view
            self.absolute_views = []
            self.relative_views = []
            
            # Select a tech stock if available
            tech_stocks = [a for a in assets if a in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']]
            if tech_stocks:
                tech_stock = tech_stocks[0]
                self.add_absolute_view(tech_stock, 0.15, 0.8)  # 15% return with 80% confidence
            else:
                # Use first asset as fallback
                self.add_absolute_view(assets[0], 0.15, 0.8)
            
            self.compute_posterior()
            self.optimize_portfolio(objective='sharpe')
            
            # Example 3: Multiple views
            self.absolute_views = []
            self.relative_views = []
            
            # Add absolute views
            for i, asset in enumerate(assets[:3]):
                # Add different views for first 3 assets
                expected_return = 0.12 + i * 0.02  # 12%, 14%, 16%
                confidence = 0.7 - i * 0.1  # 70%, 60%, 50%
                self.add_absolute_view(asset, expected_return, confidence)
            
            # Add relative views if we have enough assets
            if len(assets) >= 5:
                self.add_relative_view(assets[0], assets[3], 0.05, 0.6)  # Asset0 outperforms Asset3 by 5%
                
                if len(assets) >= 6:
                    self.add_relative_view(assets[1], assets[4], 0.03, 0.7)  # Asset1 outperforms Asset4 by 3%
            
            self.compute_posterior()
            self.optimize_portfolio(objective='sharpe')
            
            # Example 4: Contrarian views
            self.absolute_views = []
            self.relative_views = []
            
            # Identify assets with highest and lowest prior returns
            highest_return_assets = self.equilibrium_returns.nlargest(3).index
            lowest_return_assets = self.equilibrium_returns.nsmallest(3).index
            
            # Add contrarian views: higher returns for lowest return assets
            for i, asset in enumerate(lowest_return_assets):
                self.add_absolute_view(asset, 0.10 + i * 0.01, 0.6)  # 10-12% returns
            
            # Add contrarian views: lower returns for highest return assets
            for i, asset in enumerate(highest_return_assets):
                self.add_absolute_view(asset, 0.06 - i * 0.01, 0.6)  # 5-6% returns
            
            self.compute_posterior()
            self.optimize_portfolio(objective='sharpe')
            
            # Example 5: Different objectives
            self.absolute_views = []
            self.relative_views = []
            
            # Add a mix of views
            if len(assets) >= 3:
                self.add_absolute_view(assets[0], 0.14, 0.7)
                self.add_relative_view(assets[1], assets[2], 0.04, 0.6)
            
            self.compute_posterior()
            
            # Optimize with different objectives
            self.optimize_portfolio(objective='sharpe')
            self.optimize_portfolio(objective='min_variance')
            self.optimize_portfolio(objective='utility', risk_aversion=3.0)
            
            # Compare prior and posterior
            self.compare_prior_posterior()
            
            logger.info("Example Black-Litterman optimizations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Example optimizations failed: {str(e)}", exc_info=True)
            return False

if __name__ == "__main__":
    optimizer = BlackLittermanOptimizer()
    optimizer.run_example_optimizations()