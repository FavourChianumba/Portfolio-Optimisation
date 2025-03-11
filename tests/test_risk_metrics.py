import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.risk_metrics import RiskMetrics
from config.settings import settings

class TestRiskMetrics(unittest.TestCase):
    """Test cases for risk metrics module"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample data for testing
        np.random.seed(42)  # For reproducibility
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        
        # Create sample price data
        self.test_assets = ['AAPL', 'MSFT', 'BONDS', 'GOLD']
        
        # Initialize with different starting prices
        start_prices = {
            'AAPL': 100,
            'MSFT': 200,
            'BONDS': 1000,
            'GOLD': 1500
        }
        
        # Create price data with some trends and volatility
        price_data = {}
        for asset in self.test_assets:
            # Create a trend component
            trend = np.linspace(0, 0.5, 1000) if asset in ['AAPL', 'MSFT'] else np.linspace(0, 0.3, 1000)
            
            # Create a random component (more volatile for stocks, less for bonds)
            vol = 0.015 if asset in ['AAPL', 'MSFT'] else 0.005
            noise = np.random.normal(0, vol, 1000)
            
            # Combine trend and noise
            returns = trend + noise
            
            # Convert to price series
            prices = start_prices[asset] * np.cumprod(1 + returns)
            price_data[asset] = prices
        
        self.price_data = pd.DataFrame(price_data, index=dates)
        
        # Calculate returns
        self.returns_data = self.price_data.pct_change().dropna()
        
        # Add a market downturn in the middle for drawdown testing
        downturn_start = 500
        downturn_length = 100
        downturn_factor = np.linspace(1, 0.7, downturn_length)  # 30% drawdown
        
        for i in range(downturn_length):
            idx = downturn_start + i
            for asset in self.test_assets:
                self.price_data.iloc[idx, self.price_data.columns.get_loc(asset)] *= downturn_factor[i]
        
        # Recalculate returns after downturn
        self.returns_data = self.price_data.pct_change().dropna()
        
        # Save test data
        test_data_dir = project_root / 'data' / 'processed'
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.price_data.to_csv(test_data_dir / 'cleaned_asset_prices.csv')
        self.returns_data.to_csv(test_data_dir / 'daily_returns.csv')
        
        # Create risk metrics object
        self.risk_metrics = RiskMetrics(returns_data=self.returns_data, prices_data=self.price_data)
    
    def test_calculate_volatility_metrics(self):
        """Test calculation of volatility metrics"""
        # Calculate volatility metrics
        vol_metrics = self.risk_metrics.calculate_volatility_metrics()
        
        # Check that metrics were generated for each asset
        for asset in self.test_assets:
            self.assertIn(asset, vol_metrics)
            
            # Check that key metrics were calculated
            metrics = vol_metrics[asset]
            self.assertIn('daily_volatility', metrics)
            self.assertIn('annual_volatility', metrics)
            self.assertIn('upside_volatility', metrics)
            self.assertIn('downside_volatility', metrics)
            
            # Verify annual volatility is approximately daily * sqrt(252)
            self.assertAlmostEqual(
                metrics['annual_volatility'], 
                metrics['daily_volatility'] * np.sqrt(252), 
                delta=0.01
            )
            
            # Verify downside volatility is greater for more volatile assets
            if asset in ['AAPL', 'MSFT']:
                self.assertGreater(metrics['downside_volatility'], 0.1)  # Higher for stocks
            else:
                self.assertLess(metrics['downside_volatility'], 0.1)  # Lower for bonds
    
    def test_calculate_tail_risk(self):
        """Test calculation of tail risk metrics"""
        # Calculate tail risk metrics
        tail_metrics = self.risk_metrics.calculate_tail_risk()
        
        # Check that metrics were generated for each asset
        for asset in self.test_assets:
            self.assertIn(asset, tail_metrics)
            
            # Check that key metrics were calculated
            metrics = tail_metrics[asset]
            self.assertIn('var_95_daily', metrics)
            self.assertIn('cvar_95_daily', metrics)
            self.assertIn('skewness', metrics)
            self.assertIn('excess_kurtosis', metrics)
            
            # VaR should be positive (as a loss measure)
            self.assertGreater(metrics['var_95_daily'], 0)
            
            # CVaR should be greater than VaR
            self.assertGreater(metrics['cvar_95_daily'], metrics['var_95_daily'])
    
    def test_calculate_drawdowns(self):
        """Test calculation of drawdown metrics"""
        # Calculate drawdown metrics
        drawdown_metrics = self.risk_metrics.calculate_drawdowns()
        
        # Check that metrics were generated for each asset
        for asset in self.test_assets:
            self.assertIn(asset, drawdown_metrics)
            
            # Check that key metrics were calculated
            metrics = drawdown_metrics[asset]
            self.assertIn('maximum_drawdown', metrics)
            self.assertIn('average_drawdown', metrics)
            self.assertIn('underwater_days', metrics)
            
            # Maximum drawdown should be significant due to our simulated crash
            self.assertLess(metrics['maximum_drawdown'], -0.2)  # At least 20% drawdown
            
            # Average drawdown should be less severe than maximum
            self.assertGreater(metrics['average_drawdown'], metrics['maximum_drawdown'])
    
    def test_calculate_performance_metrics(self):
        """Test calculation of performance metrics"""
        # Calculate performance metrics
        perf_metrics = self.risk_metrics.calculate_performance_metrics()
        
        # Check that metrics were generated for each asset
        for asset in self.test_assets:
            self.assertIn(asset, perf_metrics)
            
            # Check that key metrics were calculated
            metrics = perf_metrics[asset]
            self.assertIn('total_return', metrics)
            self.assertIn('annualized_return', metrics)
            self.assertIn('sharpe_ratio', metrics)
            self.assertIn('sortino_ratio', metrics)
            
            # Verify calculations
            # Total return should match price data
            expected_total_return = (self.price_data[asset].iloc[-1] / self.price_data[asset].iloc[0]) - 1
            self.assertAlmostEqual(metrics['total_return'], expected_total_return, places=4)
    
    def test_calculate_portfolio_metrics(self):
        """Test calculation of portfolio metrics"""
        # Define test weights
        weights = {asset: 1.0/len(self.test_assets) for asset in self.test_assets}
        
        # Calculate portfolio metrics
        portfolio_metrics = self.risk_metrics.calculate_portfolio_metrics(weights)
        
        # Check that key metrics were calculated
        self.assertIn('mean_return', portfolio_metrics)
        self.assertIn('volatility', portfolio_metrics)
        self.assertIn('sharpe_ratio', portfolio_metrics)
        self.assertIn('sortino_ratio', portfolio_metrics)
        self.assertIn('max_drawdown', portfolio_metrics)
        self.assertIn('var_95_daily', portfolio_metrics)
        self.assertIn('return_series', portfolio_metrics)
        
        # Verify portfolio return series has correct length
        self.assertEqual(len(portfolio_metrics['return_series']), len(self.returns_data))
        
        # Verify diversification effect - portfolio volatility should be less than average asset volatility
        avg_asset_vol = np.mean([self.returns_data[asset].std() * np.sqrt(252) for asset in self.test_assets])
        self.assertLess(portfolio_metrics['volatility'], avg_asset_vol)
    
    def tearDown(self):
        """Clean up test data"""
        test_data_dir = project_root / 'data' / 'processed'
        risk_dir = test_data_dir / 'risk_metrics'
        
        # Remove test files
        for file in ['cleaned_asset_prices.csv', 'daily_returns.csv']:
            file_path = test_data_dir / file
            if file_path.exists():
                os.remove(file_path)
        
        # Remove risk metrics directory if it exists
        if risk_dir.exists():
            for file in risk_dir.glob("*"):
                os.remove(file)
            if risk_dir.exists():
                os.rmdir(risk_dir)

if __name__ == '__main__':
    unittest.main()