import os
from pathlib import Path
from dotenv import load_dotenv

# Try to load .env file if it exists
env_path = Path('.env')
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

class Settings:
    """Global settings for the portfolio optimization project"""
    
    def __init__(self):
        # Project directories
        self.DATA_RAW_DIR = "data/raw"
        self.DATA_PROCESSED_DIR = "data/processed"
        
        # API configurations
        self.YFINANCE_TIMEOUT = 30  # Seconds
        self.FRED_API_KEY = ""
        self.ALPHA_VANTAGE_API_KEY = ""
        self.POLYGON_API_KEY = ""
        
        # Data source priorities
        self.DATA_SOURCE_PRIORITY = ["yahoo", "alphavantage", "polygon", "sample"]
        
        # Data cleaning parameters
        self.MAX_MISSING_PCT = 15  # Maximum percentage of missing values allowed
        self.OUTLIER_THRESHOLD = 4.0  # Z-score threshold for outliers
        
        # Risk parameters
        self.RISK_FREE_RATE = 0.035  # Annual risk-free rate (3.5%)
        
        # Optimization constraints
        self.MIN_WEIGHT = 0.01  # Minimum asset weight (1%)
        self.MAX_WEIGHT = 0.40  # Maximum asset weight (40%)
        
        # Load environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load settings from environment variables"""
        # API keys and credentials
        self.FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
        self.ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
        self.POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
        
        # Override data source priority if set
        if os.environ.get('DATA_SOURCE_PRIORITY'):
            self.DATA_SOURCE_PRIORITY = os.environ.get('DATA_SOURCE_PRIORITY').lower().split(',')
        
        # Override default settings if environment variables exist
        if os.environ.get('RISK_FREE_RATE'):
            self.RISK_FREE_RATE = float(os.environ.get('RISK_FREE_RATE'))
        
        if os.environ.get('MAX_WEIGHT'):
            self.MAX_WEIGHT = float(os.environ.get('MAX_WEIGHT'))
        
        if os.environ.get('MIN_WEIGHT'):
            self.MIN_WEIGHT = float(os.environ.get('MIN_WEIGHT'))
        
        if os.environ.get('YFINANCE_TIMEOUT'):
            self.YFINANCE_TIMEOUT = int(os.environ.get('YFINANCE_TIMEOUT'))
            
        # Add support for data cleaning parameters from environment
        if os.environ.get('MAX_MISSING_PCT'):
            self.MAX_MISSING_PCT = float(os.environ.get('MAX_MISSING_PCT'))
            
        if os.environ.get('OUTLIER_THRESHOLD'):
            self.OUTLIER_THRESHOLD = float(os.environ.get('OUTLIER_THRESHOLD'))

# Create a singleton instance
settings = Settings()