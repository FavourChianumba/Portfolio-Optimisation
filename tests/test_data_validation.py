import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src import data_validation
from config.settings import settings

class TestDataValidation(unittest.TestCase):
    """Test cases for data validation module"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample data for testing
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Create sample asset price data
        self.test_assets = ['AAPL', 'MSFT', 'GOOG']
        self.asset_data = pd.DataFrame(
            np.random.randn(100, 3) * 0.01 + 1.0,  # Daily returns around 1.0 (0%)
            index=dates,
            columns=self.test_assets
        ).cumprod() * 100  # Starting at price 100
        
        # Add missing values to test handling
        self.asset_data.iloc[10:15, 0] = np.nan
        
        # Create sample returns data
        self.returns_data = self.asset_data.pct_change().dropna()
        
        # Create sample macro data
        self.test_macro = ['CPIAUCSL', 'UNRATE', 'FEDFUNDS']
        self.macro_data = pd.DataFrame(
            np.random.randn(100, 3) * 0.1,  # Random macro data
            index=dates,
            columns=self.test_macro
        )
        
        # Create sample directories and save test data
        test_data_dir = project_root / 'data' / 'processed'
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.asset_data.to_csv(test_data_dir / 'cleaned_asset_prices.csv')
        self.returns_data.to_csv(test_data_dir / 'daily_returns.csv')
        self.macro_data.to_csv(test_data_dir / 'cleaned_macro.csv')

    def test_validate_asset_data(self):
        """Test asset data validation function"""
        # Run the validation
        validation_results = data_validation.validate_asset_data()
        
        # Check that validation ran successfully
        self.assertIsNotNone(validation_results)
        
        # Check that validation picked up the correct number of assets
        self.assertEqual(validation_results['asset_count'], len(self.test_assets))
        
        # Check that validation identified missing values
        self.assertGreater(validation_results['missing_values']['total'], 0)
        
        # Check that return statistics were generated for each asset
        for asset in self.test_assets:
            self.assertIn(asset, validation_results['return_statistics'])
            
        # Check that stationarity tests were performed
        for asset in self.test_assets:
            self.assertIn(asset, validation_results['stationarity_tests'])
    
    def test_validate_macro_data(self):
        """Test macro data validation function"""
        # Run the validation
        validation_results = data_validation.validate_macro_data()
        
        # Check that validation ran successfully
        self.assertIsNotNone(validation_results)
        
        # Check that validation picked up the correct number of factors
        self.assertEqual(validation_results['macro_factor_count'], len(self.test_macro))
        
        # Check that factor statistics were generated for each asset
        for factor in self.test_macro:
            self.assertIn(factor, validation_results['factor_statistics'])
            
        # Check that autocorrelation tests were performed
        for factor in self.test_macro:
            self.assertIn(factor, validation_results['autocorrelation'])
    
    def tearDown(self):
        """Clean up test data"""
        test_data_dir = project_root / 'data' / 'processed'
        
        # Remove test files if they exist
        for file in ['cleaned_asset_prices.csv', 'daily_returns.csv', 'cleaned_macro.csv']:
            file_path = test_data_dir / file
            if file_path.exists():
                os.remove(file_path)

if __name__ == '__main__':
    unittest.main()