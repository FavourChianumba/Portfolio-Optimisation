import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import yahooquery as yq
import datetime
import time
import random
import certifi
import requests
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from config.settings import settings
from utils.logger import setup_logger

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Ensure data directory exists
data_dir = project_root / settings.DATA_RAW_DIR
data_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)

# Rate limiting decorator for API calls
def rate_limit(calls_limit=5, period=60):
    """Decorator to rate limit API calls"""
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if we need to wait
            now = time.time()
            # Remove old calls
            while calls and calls[0] < now - period:
                calls.pop(0)
            # If we've made too many calls, wait
            if len(calls) >= calls_limit:
                sleep_time = period - (now - calls[0])
                if sleep_time > 0:
                    logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
            # Add current call timestamp
            calls.append(time.time())
            # Make the API call
            return func(*args, **kwargs)
        return wrapper
    return decorator

class DataSource:
    """Base class for data sources"""
    
    def __init__(self, name):
        self.name = name
    
    def get_asset_data(self, assets, start_date, end_date):
        """Fetch asset price data"""
        raise NotImplementedError("Subclasses must implement get_asset_data")
    
    def get_macro_data(self, indicators, start_date, end_date):
        """Fetch macroeconomic data"""
        raise NotImplementedError("Subclasses must implement get_macro_data")

class YahooFinanceSource(DataSource):
    """Yahoo Finance data source"""
    
    def __init__(self):
        super().__init__("Yahoo Finance")
    
    def get_asset_data(self, assets, start_date, end_date):
        """Fetch asset price data from Yahoo Finance using yahooquery"""
        try:
            logger.info(f"Attempting to download asset data from {self.name} (yahooquery)...")
            
            # Initialize yahooquery Ticker object
            tickers = yq.Ticker(assets)
            
            # Download historical data
            downloaded_data = tickers.history(start=start_date, end=end_date)
            
            # Process the data
            if isinstance(downloaded_data.index, pd.MultiIndex):
                # Create an empty DataFrame for results
                asset_data = pd.DataFrame()
                
                # Process each asset
                for ticker in assets:
                    # Filter data for this ticker
                    ticker_data = downloaded_data.xs(ticker, level='symbol')
                    
                    if not ticker_data.empty:
                        # Add to our result DataFrame
                        if asset_data.empty:
                            asset_data = pd.DataFrame(index=ticker_data.index)
                        
                        # Use close price
                        asset_data[ticker] = ticker_data['close']
                        logger.info(f"Successfully downloaded data for {ticker}")
                    else:
                        logger.warning(f"No data available for {ticker}")
                
                # Make sure the index is sorted
                asset_data = asset_data.sort_index()
            else:
                logger.warning("Unexpected data format from yahooquery")
                return None
            
            # Check if we have enough assets
            if len(asset_data.columns) < len(assets) * 0.5:  # At least 50% success
                logger.warning(f"Only downloaded {len(asset_data.columns)}/{len(assets)} assets from {self.name}.")
                return None
            
            # Add debug info about the data
            logger.debug(f"Asset data shape: {asset_data.shape}")
            logger.debug(f"Date range: {asset_data.index[0]} to {asset_data.index[-1]}")
            logger.debug(f"Sample data:\n{asset_data.head().to_string()}")
            
            return asset_data
            
        except Exception as e:
            logger.error(f"{self.name} data download failed: {str(e)}")
            return None
        
    def get_macro_data(self, indicators, start_date, end_date):
        """Fetch macro indicators from Yahoo Finance (e.g., ^VIX) using yahooquery"""
        try:
            logger.info(f"Attempting to download macro indicators from {self.name} (yahooquery)...")
            
            # Filter for Yahoo indicators (those starting with ^)
            yahoo_indicators = [ind for ind in indicators if ind.startswith('^')]
            
            if not yahoo_indicators:
                logger.info(f"No Yahoo macro indicators to download")
                return None
            
            # Initialize yahooquery Ticker object
            tickers = yq.Ticker(yahoo_indicators)
            
            # Download historical data
            downloaded_data = tickers.history(start=start_date, end=end_date)
            
            # Process the data
            if isinstance(downloaded_data.index, pd.MultiIndex):
                # Create an empty DataFrame for results
                yahoo_macro = pd.DataFrame()
                
                # Process each indicator
                for indicator in yahoo_indicators:
                    # Filter data for this indicator
                    try:
                        indicator_data = downloaded_data.xs(indicator, level='symbol')
                        
                        if not indicator_data.empty:
                            # Add to our result DataFrame
                            if yahoo_macro.empty:
                                yahoo_macro = pd.DataFrame(index=indicator_data.index)
                            
                            # Use close price and rename without the ^ prefix
                            clean_name = indicator.replace('^', '')
                            yahoo_macro[clean_name] = indicator_data['close']
                            logger.info(f"Successfully downloaded {indicator}")
                        else:
                            logger.warning(f"No data available for {indicator}")
                    except KeyError:
                        logger.warning(f"Indicator {indicator} not found in downloaded data")
                
                # Make sure the index is sorted
                yahoo_macro = yahoo_macro.sort_index()
            else:
                logger.warning("Unexpected data format from yahooquery")
                return None
            
            # Check if we have enough indicators
            if len(yahoo_macro.columns) < len(yahoo_indicators) * 0.5:  # At least 50% success
                logger.warning(f"Only downloaded {len(yahoo_macro.columns)}/{len(yahoo_indicators)} indicators from {self.name}.")
                return None
            
            # Add debug info about the data
            logger.debug(f"Macro data shape: {yahoo_macro.shape}")
            if not yahoo_macro.empty:
                logger.debug(f"Date range: {yahoo_macro.index[0]} to {yahoo_macro.index[-1]}")
                logger.debug(f"Sample data:\n{yahoo_macro.head().to_string()}")
            else:
                logger.debug("Macro data is empty")
            
            logger.info(f"Successfully fetched macro indicators from {self.name}")
            return yahoo_macro
            
        except Exception as e:
            logger.warning(f"{self.name} macro indicators download failed: {str(e)}")
            return None

class AlphaVantageSource(DataSource):
    """Alpha Vantage data source"""
    
    def __init__(self, api_key):
        super().__init__("Alpha Vantage")
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    @rate_limit(calls_limit=5, period=60)  # Free tier: 5 API calls per minute
    def _fetch_daily_data(self, symbol):
        """Fetch daily time series data for a single symbol"""
        # Check if API key is available
        if not self.api_key:
            logger.warning(f"Alpha Vantage API key not provided")
            return None
            
        params = {
            "function": "TIME_SERIES_DAILY",  # Changed from TIME_SERIES_DAILY_ADJUSTED to use free endpoint
            "symbol": symbol,
            "outputsize": "full",
            "datatype": "json",
            "apikey": self.api_key
        }
        
        try:
            # Add retry mechanism with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(self.base_url, params=params, timeout=30)
                    response.raise_for_status()  # Raise exception for 4XX/5XX responses
                    data = response.json()
                    break
                except (requests.exceptions.RequestException, ValueError) as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {symbol}: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Check for error messages
            if "Error Message" in data:
                logger.warning(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                return None
                
            # Check for information messages (including premium endpoint messages)
            if "Information" in data:
                logger.warning(f"Alpha Vantage information for {symbol}: {data['Information']}")
                return None
            
            # Parse time series data
            if "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                df = pd.DataFrame.from_dict(time_series, orient="index")
                df = df.astype(float)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df["4. close"]  # Use regular close price instead of adjusted close
            else:
                logger.warning(f"No time series data returned for {symbol}")
                return None
        
        except Exception as e:
            logger.warning(f"Error fetching {symbol} from Alpha Vantage: {str(e)}")
            return None

    def get_asset_data(self, assets, start_date, end_date):
        """Fetch asset price data from Alpha Vantage"""
        try:
            logger.info(f"Attempting to download asset data from {self.name}...")
            
            if not self.api_key:
                logger.warning(f"{self.name} API key not provided")
                return None
            
            # Initialize empty DataFrame for results
            all_data = pd.DataFrame()
            
            # Fetch data for each asset
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Use ThreadPoolExecutor but still respect rate limits through the decorator
                results = list(executor.map(self._fetch_daily_data, assets))
            
            # Combine results
            for i, asset in enumerate(assets):
                if results[i] is not None:
                    all_data[asset] = results[i]
                    logger.info(f"Successfully downloaded {asset} from {self.name}")
            
            # Filter by date range
            if not all_data.empty:
                all_data = all_data[(all_data.index >= pd.Timestamp(start_date)) & 
                                    (all_data.index <= pd.Timestamp(end_date))]
            
            # Check if we have enough assets
            if len(all_data.columns) < len(assets) * 0.5:  # At least 50% success
                logger.warning(f"Only downloaded {len(all_data.columns)}/{len(assets)} assets from {self.name}.")
                return None
            
            return all_data
        
        except Exception as e:
            logger.error(f"{self.name} data download failed: {str(e)}")
            return None
    
    def get_macro_data(self, indicators, start_date, end_date):
        """Alpha Vantage doesn't have a direct equivalent for all macro indicators"""
        logger.info(f"{self.name} doesn't support macroeconomic indicators through this implementation")
        return None

class PolygonSource(DataSource):
    """Polygon.io data source"""
        
    def __init__(self, api_key):
        super().__init__("Polygon")
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
        # Check API key
        if not self.api_key:
            logger.warning("Polygon API key not provided. API calls will likely fail.")
    
    @rate_limit(calls_limit=5, period=60)  # Adjust based on your API tier
    def _fetch_daily_data(self, symbol, start_date, end_date):
        """Fetch daily time series data for a single symbol"""
        # Check if API key is provided
        if not self.api_key:
            logger.warning(f"Polygon API key not provided")
            return None
            
        # Format dates for Polygon API (YYYY-MM-DD)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # API endpoint for daily aggregates
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_str}/{end_str}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # Add retry mechanism with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()  # Raise exception for 4XX/5XX responses
                    data = response.json()
                    break
                except (requests.exceptions.RequestException, ValueError) as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {symbol}: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Check for errors
            if data.get("status") == "ERROR":
                logger.warning(f"Polygon error for {symbol}: {data.get('error')}")
                return None
            
            # Parse results
            if "results" in data and data["results"]:
                # Create DataFrame from results
                df = pd.DataFrame(data["results"])
                # Convert timestamp to datetime
                df["t"] = pd.to_datetime(df["t"], unit="ms")
                # Set timestamp as index
                df = df.set_index("t")
                # Use close price
                df = df[["c"]].rename(columns={"c": symbol})
                return df
            else:
                logger.warning(f"No results returned for {symbol}")
                return None
        
        except Exception as e:
            logger.warning(f"Error fetching {symbol} from Polygon: {str(e)}")
            return None
    
    def get_asset_data(self, assets, start_date, end_date):
        """Fetch asset price data from Polygon"""
        try:
            logger.info(f"Attempting to download asset data from {self.name}...")
            
            if not self.api_key:
                logger.warning(f"{self.name} API key not provided")
                return None
            
            # Initialize empty DataFrame for results
            all_data = pd.DataFrame()
            
            # Fetch data for each asset
            for asset in assets:
                # Polygon API rate limits are stricter, so we don't use ThreadPoolExecutor here
                asset_data = self._fetch_daily_data(asset, start_date, end_date)
                if asset_data is not None and not asset_data.empty:
                    # Merge with existing data
                    if all_data.empty:
                        all_data = asset_data
                    else:
                        all_data = all_data.join(asset_data, how="outer")
                    logger.info(f"Successfully downloaded {asset} from {self.name}")
            
            # Check if we have enough assets
            if len(all_data.columns) < len(assets) * 0.5:  # At least 50% success
                logger.warning(f"Only downloaded {len(all_data.columns)}/{len(assets)} assets from {self.name}.")
                return None
            
            return all_data
        
        except Exception as e:
            logger.error(f"{self.name} data download failed: {str(e)}")
            return None
    
    def get_macro_data(self, indicators, start_date, end_date):
        """Polygon doesn't have macroeconomic indicators in the basic API"""
        logger.info(f"{self.name} doesn't support macroeconomic indicators through this implementation")
        return None

def collect_data():
    """Fetch raw asset prices and macro data with fallback to sample data"""
    try:
        logger.info("Starting data collection...")
        
        # Check API keys and warn about missing ones
        if not settings.FRED_API_KEY:
            logger.warning("FRED API key not found. FRED data collection may fail.")
        if not settings.ALPHA_VANTAGE_API_KEY:
            logger.warning("Alpha Vantage API key not found. Alpha Vantage data collection may fail.")
        if not settings.POLYGON_API_KEY:
            logger.warning("Polygon API key not found. Polygon data collection may fail.")
        
        # SSL configuration for FRED API
        try:
            import ssl
            import certifi
            # Use certifi certificates for SSL
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            logger.warning("certifi package not found. SSL certificate verification might fail.")
        
        # Configuration - core assets list
        assets = [
            "SPY",   # S&P 500
            "QQQ",   # Nasdaq 100
            "TLT",   # Long-Term Treasury
            "GLD",   # Gold
            "AAPL",  # Apple
            "MSFT",  # Microsoft
            "AMZN",  # Amazon
            "JPM",   # JPMorgan Chase
            "XOM"    # Exxon Mobil
        ]
        
        # Rest of your existing code...
        
        # Macro factors list
        macro_factors = [
            "CPIAUCSL",  # Consumer Price Index
            "UNRATE",    # Unemployment Rate
            "FEDFUNDS",  # Federal Funds Rate
            "T10Y2Y",    # 10-Year - 2-Year Treasury Spread
            "BAMLH0A0HYM2",  # ICE BofA US High Yield Index Option-Adjusted Spread
            "^VIX",      # Volatility Index
            "^TNX",      # 10-Year Treasury Yield
            "^TYX",      # 30-Year Treasury Yield
            "^DJI"       # Dow Jones Industrial Average
        ]
        
        # Define date range
        end_date = datetime.datetime.now() - datetime.timedelta(days=1)
        start_date = end_date - datetime.timedelta(days=365*5)  # 5 years
        
        # Initialize data sources
        data_sources = [
            YahooFinanceSource(),
            AlphaVantageSource(settings.ALPHA_VANTAGE_API_KEY),
            PolygonSource(settings.POLYGON_API_KEY)
        ]
        
        # Try each data source for asset data
        asset_data = None
        for source in data_sources:
            asset_data = source.get_asset_data(assets, start_date, end_date)
            if asset_data is not None and not asset_data.empty:
                logger.info(f"Successfully fetched asset data from {source.name}")
                break
        
        # If all data sources fail, generate sample data
        if asset_data is None or asset_data.empty:
            logger.warning("All data sources failed. Using sample asset data.")
            asset_data = generate_sample_data(assets, start_date, end_date)
        
        # 2. Fetch FRED macro data
        fred_data = pd.DataFrame()
        try:
            logger.info("Attempting to download macro data from FRED...")
            
            # Use fredapi if available
            try:
                from fredapi import Fred
                
                # Initialize Fred with API key
                fred_api_key = settings.FRED_API_KEY
                if not fred_api_key:
                    logger.warning("FRED API key not found. Check your .env file.")
                    raise ValueError("FRED API key not found")
                
                fred = Fred(api_key=fred_api_key)
                # Try setting SSL context if attribute exists
                if hasattr(fred, 'session') and hasattr(fred.session, 'verify'):
                    fred.session.verify = certifi.where()
                
                # Collect each macro series
                for factor in [f for f in macro_factors if not f.startswith('^')]:
                    try:
                        series = fred.get_series(factor, start_date, end_date)
                        if not series.empty:
                            fred_data[factor] = series
                            logger.info(f"Successfully downloaded {factor} from FRED")
                        else:
                            logger.warning(f"No data returned for {factor}")
                    except Exception as e:
                        logger.warning(f"Error downloading {factor}: {str(e)}")
                
            except (ImportError, ValueError) as e:
                logger.warning(f"FRED API issue: {str(e)}. Using pandas_datareader instead.")
                
                # Fallback to pandas_datareader
                import pandas_datareader as pdr
                for factor in [f for f in macro_factors if not f.startswith('^')]:
                    try:
                        series = pdr.get_data_fred(factor, start_date, end_date)
                        if not series.empty:
                            fred_data[factor] = series
                            logger.info(f"Successfully downloaded {factor} from FRED via pandas_datareader")
                        else:
                            logger.warning(f"No data returned for {factor}")
                    except Exception as e:
                        logger.warning(f"Error downloading {factor}: {str(e)}")
            
            # If we still don't have any data, use sample data
            if fred_data.empty:
                logger.warning("No FRED data downloaded. Using sample macro data instead.")
                fred_data = generate_sample_macro_data([f for f in macro_factors if not f.startswith('^')], 
                                                      start_date, end_date)
                
        except Exception as e:
            logger.error(f"FRED data collection failed: {str(e)}")
            logger.warning("Using sample macro data instead.")
            fred_data = generate_sample_macro_data([f for f in macro_factors if not f.startswith('^')], 
                                                  start_date, end_date)
        
        # 3. Get Yahoo Finance macro indicators (VIX, etc.)
        yahoo_macro = None
        for source in data_sources:
            yahoo_macro = source.get_macro_data([f for f in macro_factors if f.startswith('^')], 
                                              start_date, end_date)
            if yahoo_macro is not None and not yahoo_macro.empty:
                logger.info(f"Successfully fetched macro indicators from {source.name}")
                break
        
        # Save raw data
        asset_path = data_dir / "asset_prices.csv"
        fred_path = data_dir / "fred_macro.csv"
        yahoo_macro_path = data_dir / "yahoo_macro.csv"
        
        asset_data.to_csv(asset_path)
        fred_data.to_csv(fred_path)
        
        if yahoo_macro is not None and not yahoo_macro.empty:
            yahoo_macro.to_csv(yahoo_macro_path)
        
        # Save metadata
        metadata = {
            "collection_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "assets_count": len(asset_data.columns),
            "fred_factors_count": len(fred_data.columns),
            "yahoo_macro_count": len(yahoo_macro.columns) if yahoo_macro is not None else 0,
            "actual_assets": list(asset_data.columns),
            "actual_fred_factors": list(fred_data.columns),
            "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "using_sample_data": len(asset_data.columns) < len(assets)
        }
        
        # Save as JSON
        import json
        with open(data_dir / "collection_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Data collection completed. Files saved to: {data_dir}")
        return asset_data, fred_data, yahoo_macro if yahoo_macro is not None else None

    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}", exc_info=True)
        raise
    
def generate_sample_data(assets, start_date, end_date):
    """Generate realistic sample data for assets"""
    logger.info("Generating sample asset price data...")
    
    # Create date range at business day frequency
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Create empty DataFrame
    sample_data = pd.DataFrame(index=dates)
    
    # Generate sample data for each asset
    for asset in assets:
        # Set starting price based on typical range for the asset
        if asset in ["SPY", "QQQ"]:
            start_price = random.uniform(200, 400)
        elif asset in ["TLT", "GLD"]:
            start_price = random.uniform(100, 200)
        elif asset in ["AAPL", "MSFT", "AMZN"]:
            start_price = random.uniform(100, 300)
        else:
            start_price = random.uniform(50, 150)
        
        # Set daily return parameters
        daily_return_mean = random.uniform(0.0002, 0.0008)  # ~5-20% annual
        daily_return_std = random.uniform(0.008, 0.025)     # ~13-40% annual vol
        
        # Generate returns with momentum (autocorrelation)
        n_days = len(dates)
        random_component = np.random.normal(daily_return_mean, daily_return_std, n_days)
        momentum_component = np.zeros(n_days)
        
        for i in range(1, n_days):
            momentum_component[i] = 0.2 * random_component[i-1]
        
        returns = random_component + momentum_component
        
        # Add some realistic market drops
        # 2020 COVID crash
        covid_start = pd.Timestamp('2020-02-19')
        covid_end = pd.Timestamp('2020-03-23')
        if covid_start >= dates[0] and covid_end <= dates[-1]:
            covid_mask = (dates >= covid_start) & (dates <= covid_end)
            covid_indices = np.where(covid_mask)[0]
            for i in covid_indices:
                returns[i] = returns[i] - 0.03  # Additional 3% daily drop
        
        # Convert returns to prices
        prices = start_price * np.cumprod(1 + returns)
        
        # Add to DataFrame
        sample_data[asset] = prices
    
    return sample_data

def generate_sample_macro_data(factors, start_date, end_date):
    """Generate realistic sample data for macroeconomic factors"""
    logger.info("Generating sample macroeconomic data...")
    
    # Create date range at monthly frequency (macro data typically monthly)
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Create empty DataFrame
    macro_data = pd.DataFrame(index=monthly_dates)
    
    # Generate sample data for each factor
    for factor in factors:
        if factor == "CPIAUCSL":  # CPI
            # Start around 250-260 (typical 2018-2019 values)
            start_value = random.uniform(250, 260)
            # Monthly increase 0.1-0.4% (1.2-4.8% annual inflation)
            monthly_change = np.random.normal(0.002, 0.001, len(monthly_dates))
            # Higher inflation in 2021-2022
            high_inf_start = pd.Timestamp('2021-06-01')
            high_inf_end = pd.Timestamp('2022-06-01')
            if high_inf_start >= monthly_dates[0] and high_inf_end <= monthly_dates[-1]:
                high_inf_mask = (monthly_dates >= high_inf_start) & (monthly_dates <= high_inf_end)
                monthly_change[high_inf_mask] += 0.005  # Additional 0.5% monthly (6% annual)
                
        elif factor == "UNRATE":  # Unemployment
            # Start around 3.5-4% (pre-COVID)
            start_value = random.uniform(3.5, 4.0)
            # Small monthly changes
            monthly_change = np.random.normal(0, 0.1, len(monthly_dates))
            # COVID spike
            covid_start = pd.Timestamp('2020-03-01')
            covid_peak = pd.Timestamp('2020-04-01')
            covid_recovery = pd.Timestamp('2020-12-01')
            if covid_start >= monthly_dates[0] and covid_recovery <= monthly_dates[-1]:
                # Spike unemployment to ~14%
                covid_idx = np.where(monthly_dates == covid_peak)[0]
                if len(covid_idx) > 0:
                    monthly_change[covid_idx[0]] += 10  # Big jump
                # Gradual recovery
                recovery_period = np.where((monthly_dates > covid_peak) & (monthly_dates <= covid_recovery))[0]
                for i, idx in enumerate(recovery_period):
                    monthly_change[idx] -= 0.7  # Monthly recovery
                
        elif factor == "FEDFUNDS":  # Fed Funds Rate
            # Start around 2%
            start_value = random.uniform(1.5, 2.5)
            # Small monthly changes
            monthly_change = np.random.normal(0, 0.05, len(monthly_dates))
            # COVID drop to zero
            covid_drop = pd.Timestamp('2020-03-01')
            rate_hike_start = pd.Timestamp('2022-03-01')
            if covid_drop >= monthly_dates[0] and covid_drop <= monthly_dates[-1]:
                covid_idx = np.where(monthly_dates >= covid_drop)[0]
                if len(covid_idx) > 0:
                    monthly_change[covid_idx[0]] = -start_value  # Drop to zero
                    for i in range(1, min(12, len(covid_idx))):
                        monthly_change[covid_idx[i]] = 0  # Stay at zero
            # Rate hikes in 2022
            if rate_hike_start >= monthly_dates[0] and rate_hike_start <= monthly_dates[-1]:
                hike_idx = np.where(monthly_dates >= rate_hike_start)[0]
                for i in range(min(10, len(hike_idx))):
                    monthly_change[hike_idx[i]] = 0.25  # 25bps hikes
        
        else:
            # Generic factor
            start_value = random.uniform(50, 150)
            monthly_change = np.random.normal(0, 0.02, len(monthly_dates))
        
        # Convert changes to values
        values = np.zeros(len(monthly_dates))
        values[0] = start_value
        for i in range(1, len(monthly_dates)):
            if factor in ["FEDFUNDS", "UNRATE"]:  # These can't go below zero
                values[i] = max(0, values[i-1] + monthly_change[i])
            else:
                values[i] = values[i-1] * (1 + monthly_change[i])
        
        # Add to DataFrame
        macro_data[factor] = values
    
    # Resample to business day frequency with forward fill
    business_day_data = macro_data.resample('B').ffill()
    
    return business_day_data

if __name__ == "__main__":
    collect_data()