import pandas as pd
import numpy as np
from pathlib import Path
import sys
from config.settings import settings
from utils.logger import setup_logger

# Add the project root directory to Python path - KEEP ORIGINAL PATH DETERMINATION
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Setup directories
data_raw_dir = project_root / settings.DATA_RAW_DIR
data_processed_dir = project_root / settings.DATA_PROCESSED_DIR
data_processed_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)

def clean_asset_data(asset_data=None):
    """Clean and process asset price data"""
    try:
        logger.info("Starting asset data cleaning...")
        
        # Load data if not provided
        if asset_data is None:
            asset_path = data_raw_dir / "asset_prices.csv"
            if not asset_path.exists():
                logger.error(f"Asset price data file not found: {asset_path}")
                raise FileNotFoundError(f"Asset price data file not found: {asset_path}")
            
            logger.info(f"Loading asset data from {asset_path}")
            asset_data = pd.read_csv(asset_path, index_col=0, parse_dates=True)
        
        # Check for empty dataframe
        if asset_data.empty:
            logger.error("Asset data is empty")
            raise ValueError("Asset data is empty")
            
        # Check for and handle missing values
        missing_pct = asset_data.isna().mean() * 100
        logger.info(f"Missing values before cleaning: {missing_pct.describe()}")
        
        # Remove assets with too many missing values
        valid_assets = asset_data.columns[missing_pct < settings.MAX_MISSING_PCT]
        invalid_assets = set(asset_data.columns) - set(valid_assets)
        
        if invalid_assets:
            logger.warning(f"Removing assets with too many missing values: {invalid_assets}")
            asset_data = asset_data[valid_assets]
            
        if asset_data.empty or len(asset_data.columns) == 0:
            logger.error("No valid assets remaining after filtering")
            raise ValueError("No valid assets remaining after filtering")
        
        # Forward fill remaining missing values (standard approach for time series)
        asset_data = asset_data.fillna(method='ffill')
        
        # Backfill any remaining NAs at the beginning
        asset_data = asset_data.fillna(method='bfill')
        
        # Check if we still have NAs after filling
        if asset_data.isna().any().any():
            logger.warning("Some missing values could not be filled")
            # Drop rows with any remaining NAs as a last resort
            asset_data = asset_data.dropna()
            
        # Check for outliers using rolling Z-score
        returns = asset_data.pct_change().dropna()
        # Add min_periods to avoid NaN issues with shorter windows
        rolling_mean = returns.rolling(window=20, min_periods=5).mean()
        rolling_std = returns.rolling(window=20, min_periods=5).std()
        # Handle cases where std is 0
        rolling_std = rolling_std.replace(0, np.nan)
        z_scores = (returns - rolling_mean) / rolling_std
        
        # Flag potential outliers
        outliers = (z_scores.abs() > settings.OUTLIER_THRESHOLD).sum()
        if outliers.sum() > 0:
            logger.warning(f"Potential outliers detected: {outliers}")
            
            # Winsorize extreme values rather than removing
            for col in returns.columns:
                upper_bound = returns[col].quantile(0.995)
                lower_bound = returns[col].quantile(0.005)
                returns[col] = returns[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Recalculate prices from winsorized returns
            clean_prices = (1 + returns).cumprod() * asset_data.iloc[0]
            
            # Keep original data for the first day
            clean_prices.iloc[0] = asset_data.iloc[0]
            
            asset_data = clean_prices
        
        # Ensure all tickers align to same trading days (business days only)
        if not isinstance(asset_data.index, pd.DatetimeIndex):
            logger.warning("Index is not a DatetimeIndex, attempting to convert")
            try:
                asset_data.index = pd.to_datetime(asset_data.index)
            except Exception as e:
                logger.warning(f"Could not convert index to DatetimeIndex: {str(e)}")
                logger.warning("Skipping asfreq step")
                
        # Only run asfreq if we have a DatetimeIndex
        if isinstance(asset_data.index, pd.DatetimeIndex):
            # When resampling to business days, fill any new rows
            asset_data = asset_data.asfreq('B').fillna(method='ffill')

        # Save cleaned data
        output_path = data_processed_dir / "cleaned_asset_prices.csv"
        asset_data.to_csv(output_path)
        logger.info(f"Saved cleaned asset prices to {output_path}")
        
        # Calculate and save daily returns
        returns = asset_data.pct_change().dropna()
        returns_path = data_processed_dir / "daily_returns.csv"
        returns.to_csv(returns_path)
        logger.info(f"Saved daily returns to {returns_path}")
        
        # Calculate and save monthly returns for longer-term analysis
        monthly_returns = asset_data.resample('M').last().pct_change().dropna()
        monthly_returns_path = data_processed_dir / "monthly_returns.csv"
        monthly_returns.to_csv(monthly_returns_path)
        logger.info(f"Saved monthly returns to {monthly_returns_path}")
        
        logger.info(f"Asset data cleaning completed. Files saved to: {data_processed_dir}")
        return asset_data, returns, monthly_returns
        
    except Exception as e:
        logger.error(f"Asset data cleaning failed: {str(e)}", exc_info=True)
        raise

def clean_macro_data():
    """Clean and process macroeconomic data"""
    try:
        logger.info("Starting macro data cleaning...")
        
        # Load macro data
        fred_path = data_raw_dir / "fred_macro.csv"
        if not fred_path.exists():
            logger.error(f"FRED macro data file not found: {fred_path}")
            raise FileNotFoundError(f"FRED macro data file not found: {fred_path}")
            
        logger.info(f"Loading FRED macro data from {fred_path}")
        fred_data = pd.read_csv(fred_path, index_col=0, parse_dates=True)
        
        # Check for empty dataframe
        if fred_data.empty:
            logger.error("FRED macro data is empty")
            raise ValueError("FRED macro data is empty")
        
        # Check for yahoo macro data
        yahoo_macro_path = data_raw_dir / "yahoo_macro.csv"
        combined_macro = fred_data
        
        if yahoo_macro_path.exists():
            logger.info(f"Loading Yahoo macro data from {yahoo_macro_path}")
            yahoo_data = pd.read_csv(yahoo_macro_path, index_col=0, parse_dates=True)
            
            if not yahoo_data.empty:
                # Combine data sources if both exist
                logger.info("Combining FRED and Yahoo macro data")
                combined_macro = pd.concat([fred_data, yahoo_data], axis=1)
                # Remove duplicate columns if any
                combined_macro = combined_macro.loc[:, ~combined_macro.columns.duplicated()]
        
        # Ensure index is DateTime
        if not isinstance(combined_macro.index, pd.DatetimeIndex):
            logger.warning("Index is not a DatetimeIndex, attempting to convert")
            try:
                combined_macro.index = pd.to_datetime(combined_macro.index)
            except Exception as e:
                logger.warning(f"Could not convert to DatetimeIndex: {str(e)}")
                logger.warning("Using linear interpolation instead of time interpolation")
                # Fall back to linear interpolation
                combined_macro = combined_macro.interpolate(method='linear')
                combined_macro = combined_macro.fillna(method='ffill')
                combined_macro = combined_macro.fillna(method='bfill')
        else:
            # Handle missing values - for macro data, interpolation is appropriate
            combined_macro = combined_macro.interpolate(method='time')
            combined_macro = combined_macro.fillna(method='ffill')
            combined_macro = combined_macro.fillna(method='bfill')
                
        # Create business day frequency for alignment with asset data
        if isinstance(combined_macro.index, pd.DatetimeIndex):
            try:
                combined_macro = combined_macro.resample('B').interpolate(method='time')
            except Exception as e:
                logger.warning(f"Error during resampling: {str(e)}")
                # Try a simpler approach if resampling fails
                combined_macro = combined_macro.resample('B').ffill()
        
        # Calculate percent changes for easier interpretation
        macro_changes = pd.DataFrame()
        
        for col in combined_macro.columns:
            # Skip columns with too many NaNs
            if combined_macro[col].isna().mean() > 0.3:  # More than 30% missing
                logger.warning(f"Skipping column {col} due to too many missing values")
                continue
                
            try:
                # Different handling based on data type
                if col in ['FEDFUNDS', 'DGS10', 'DGS2', 'T10Y2Y', 'VIX']:
                    # These are already rates or spreads, use level changes
                    macro_changes[f"{col}_change"] = combined_macro[col].diff()
                else:
                    # For others, use percent changes
                    macro_changes[f"{col}_pct_change"] = combined_macro[col].pct_change() * 100
            except Exception as e:
                logger.warning(f"Error calculating changes for {col}: {str(e)}")
        
        # Drop initial NaN values
        macro_changes = macro_changes.dropna()
        
        # Save cleaned data
        output_path = data_processed_dir / "cleaned_macro.csv"
        combined_macro.to_csv(output_path)
        logger.info(f"Saved cleaned macro data to {output_path}")
        
        # Save macro changes
        changes_path = data_processed_dir / "macro_changes.csv"
        macro_changes.to_csv(changes_path)
        logger.info(f"Saved macro changes to {changes_path}")
        
        logger.info(f"Macro data cleaning completed. Files saved to: {data_processed_dir}")
        return combined_macro, macro_changes
        
    except Exception as e:
        logger.error(f"Macro data cleaning failed: {str(e)}", exc_info=True)
        raise

def verify_data_files():
    """Verify that necessary raw data files exist"""
    required_files = [
        "asset_prices.csv",
        "fred_macro.csv"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = data_raw_dir / file
        if not file_path.exists():
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing required data files: {missing_files}")
        return False
    
    return True

if __name__ == "__main__":
    clean_asset_data()
    clean_macro_data()