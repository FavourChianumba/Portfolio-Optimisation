import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src import optimization
from config.settings import settings

class TestOptimization(unittest.TestCase):
    """Test cases for portfolio optimization module"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample data for testing
        np.random.seed(42)  # For reproducibility
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        
        # Create correlated returns for realistic testing
        cov_matrix = np.array([
            [0.0004, 0.0002, 0.0001, 0.0000],
            [0.0002, 0.0005, 0.0000, 0.0001],
            [0.0001, 0.0000, 0.0006, -0.0001],
            [0.0000, 0.0001, -0.0001, 0.0008]
        ])
        
        means = np.array([0.0005, 0.0004, 0.0006, 0.0008])  # Different expected returns
        assets = ['AAPL', 'MSFT', 'BONDS', 'GOLD']
        
        # Generate multivariate normal returns
        returns_data = np.random.multivariate_normal(means, cov_matrix, size=1000)
        self.returns = pd.DataFrame(returns_data, index=dates, columns=assets)
        
        # Save test data
        test_data_dir = project_root / 'data' / 'processed'
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.returns.to_csv(test_data_dir / 'daily_returns.csv')
        
        # Create optimizer
        self.optimizer = optimization.PortfolioOptimizer(returns_data=self.returns)
    
    def test_generate_efficient_frontier(self):
        """Test generation of efficient frontier with random portfolios"""
        # Run efficient frontier with a small number of portfolios for testing
        results, min_vol, max_sharpe = self.optimizer.generate_efficient_frontier(
            num_portfolios=100, save_results=False)
        
        # Check results
        self.assertEqual(len(results), 100)  # Check number of portfolios
        self.assertIn('Return', results.columns)
        self.assertIn('Volatility', results.columns)
        self.assertIn('Sharpe', results.columns)
        
        # Check min vol portfolio
        self.assertIn('Return', min_vol)
        self.assertIn('Volatility', min_vol)
        self.assertIn('Sharpe', min_vol)
        self.assertIn('Weights', min_vol)
        
        # Check max Sharpe portfolio
        self.assertIn('Return', max_sharpe)
        self.assertIn('Volatility', max_sharpe)
        self.assertIn('Sharpe', max_sharpe)
        self.assertIn('Weights', max_sharpe)
        
        # Verify min vol has lowest volatility
        self.assertEqual(min_vol['Volatility'], results['Volatility'].min())
        
        # Verify max Sharpe has highest Sharpe ratio
        self.assertEqual(max_sharpe['Sharpe'], results['Sharpe'].max())
    
    def test_optimize_sharpe_ratio(self):
        """Test optimization for maximum Sharpe ratio"""
        # Run optimization
        max_sharpe = self.optimizer.optimize_sharpe_ratio(save_results=False)
        
        # Check results
        self.assertIn('Return', max_sharpe)
        self.assertIn('Volatility', max_sharpe)
        self.assertIn('Sharpe', max_sharpe)
        self.assertIn('Weights', max_sharpe)
        
        # Verify weights sum to 1
        self.assertAlmostEqual(sum(max_sharpe['Weights'].values()), 1.0, places=6)
        
        # Verify all weights are non-negative (long-only constraint)
        for weight in max_sharpe['Weights'].values():
            self.assertGreaterEqual(weight, 0)
    
    def test_optimize_minimum_volatility(self):
        """Test optimization for minimum volatility"""
        # Run optimization
        min_vol = self.optimizer.optimize_minimum_volatility(save_results=False)
        
        # Check results
        self.assertIn('Return', min_vol)
        self.assertIn('Volatility', min_vol)
        self.assertIn('Sharpe', min_vol)
        self.assertIn('Weights', min_vol)
        
        # Verify weights sum to 1
        self.assertAlmostEqual(sum(min_vol['Weights'].values()), 1.0, places=6)
        
        # Verify all weights are non-negative (long-only constraint)
        for weight in min_vol['Weights'].values():
            self.assertGreaterEqual(weight, 0)
    
    def test_optimize_efficient_return(self):
        """Test optimization for efficient return"""
        # Pick a target return between min and max asset returns
        target_return = self.returns.mean().mean() * 252 * 1.5
        
        # Run optimization
        efficient_portfolio = self.optimizer.optimize_efficient_return(
            target_return=target_return, save_results=False)
        
        # Check results
        self.assertIn('Return', efficient_portfolio)
        self.assertIn('Volatility', efficient_portfolio)
        self.assertIn('Sharpe', efficient_portfolio)
        self.assertIn('Weights', efficient_portfolio)
        
        # Verify weights sum to 1
        self.assertAlmostEqual(sum(efficient_portfolio['Weights'].values()), 1.0, places=6)
        
        # Verify target return is achieved (within tolerance)
        self.assertAlmostEqual(efficient_portfolio['Return'], target_return, places=4)
    
    def test_risk_parity(self):
        """Test risk parity optimization"""
        # Run optimization
        risk_parity = self.optimizer.optimize_risk_parity(save_results=False)
        
        # Check results
        self.assertIn('Return', risk_parity)
        self.assertIn('Volatility', risk_parity)
        self.assertIn('Sharpe', risk_parity)
        self.assertIn('Weights', risk_parity)
        self.assertIn('Risk_Contributions', risk_parity)
        
        # Verify weights sum to 1
        self.assertAlmostEqual(sum(risk_parity['Weights'].values()), 1.0, places=6)
        
        # Verify risk contributions sum to 1
        self.assertAlmostEqual(sum(risk_parity['Risk_Contributions'].values()), 1.0, places=6)
        
        # Verify risk contributions are roughly equal
        risk_values = list(risk_parity['Risk_Contributions'].values())
        self.assertLess(max(risk_values) - min(risk_values), 0.1)  # Within 10% tolerance
    
    def tearDown(self):
        """Clean up test data"""
        test_data_dir = project_root / 'data' / 'processed'
        optimization_dir = test_data_dir / 'optimization'
        
        # Remove test files
        returns_path = test_data_dir / 'daily_returns.csv'
        if returns_path.exists():
            os.remove(returns_path)
        
        # Remove optimization directory if it exists
        if optimization_dir.exists():
            for file in optimization_dir.glob("*"):
                os.remove(file)
            os.rmdir(optimization_dir)

if __name__ == '__main__':
    unittest.main()