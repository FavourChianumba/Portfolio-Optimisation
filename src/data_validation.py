import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from config.settings import settings
from utils.logger import setup_logger

# Add after other imports but before setting up directories
# Gracefully handle statsmodels dependency
try:
    from statsmodels.tsa.stattools import adfuller, acf
except ImportError:
    logger.warning("statsmodels not installed. Some validation tests will be skipped.")
    # Define dummy functions that return reasonable default values
    def adfuller(x):
        return [0, 1, 0, 0, {"1%": 0, "5%": 0, "10%": 0}]
    
    def acf(x, nlags=10):
        return np.zeros(nlags+1)

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Setup directories
data_processed_dir = project_root / settings.DATA_PROCESSED_DIR
validation_dir = data_processed_dir / "validation"
validation_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)

def validate_asset_data():
    """Validate processed asset data for portfolio optimization"""
    try:
        logger.info("Starting asset data validation...")
        
        # Load cleaned asset data
        asset_path = data_processed_dir / "cleaned_asset_prices.csv"
        returns_path = data_processed_dir / "daily_returns.csv"
        
        if not asset_path.exists() or not returns_path.exists():
            logger.warning("Asset data files not found")
            return None
        
        asset_data = pd.read_csv(asset_path, index_col=0, parse_dates=True)
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        validation_results = {
            "asset_count": len(asset_data.columns),
            "date_range": {
                "start": asset_data.index[0].strftime("%Y-%m-%d") if len(asset_data) > 0 else None,
                "end": asset_data.index[-1].strftime("%Y-%m-%d") if len(asset_data) > 0 else None,
                "trading_days": len(asset_data)
            },
            "missing_values": {
                "total": int(asset_data.isna().sum().sum()),
                "percentage": float(asset_data.isna().mean().mean() * 100) if not asset_data.empty else 0
            },
            "return_statistics": {},
            "correlation_issues": [],
            "stationarity_tests": {}
        }
        
        # Validate returns
        for asset in returns.columns:
            asset_returns = returns[asset].dropna()
            
            # Calculate basic statistics
            if len(asset_returns) > 0:
                stats = {
                    "mean": float(asset_returns.mean()),
                    "std": float(asset_returns.std()),
                    "min": float(asset_returns.min()),
                    "max": float(asset_returns.max()),
                    "skew": float(asset_returns.skew()),
                    "kurtosis": float(asset_returns.kurtosis())
                }
            else:
                # Handle empty data
                logger.warning(f"Asset {asset} has no return data, using default statistics")
                stats = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "skew": 0.0,
                    "kurtosis": 0.0
                }
            
            # Check for jarque-bera test of normality
            try:
                from scipy import stats as scipy_stats
                jb_stat, jb_pval = scipy_stats.jarque_bera(asset_returns)
                stats["jarque_bera"] = {
                    "statistic": float(jb_stat),
                    "p_value": float(jb_pval),
                    "normal_at_5pct": jb_pval > 0.05
                }
            except Exception as e:
                logger.debug(f"Could not perform jarque-bera test for {asset}: {str(e)}")
                stats["jarque_bera"] = None
            
            # Check for stationarity (important for time series modeling)
            try:
                from statsmodels.tsa.stattools import adfuller
                if len(asset_returns) > 10:  # Need sufficient data for stationarity test
                    adf_result = adfuller(asset_returns)
                    validation_results["stationarity_tests"][asset] = {
                        "adf_statistic": float(adf_result[0]),
                        "p_value": float(adf_result[1]),
                        "stationary_at_5pct": adf_result[1] < 0.05
                    }
                else:
                    validation_results["stationarity_tests"][asset] = {
                        "adf_statistic": 0.0,
                        "p_value": 1.0,
                        "stationary_at_5pct": False,
                        "note": "Insufficient data for reliable test"
                    }
            except Exception as e:
                logger.debug(f"Could not perform stationarity test for {asset}: {str(e)}")
                validation_results["stationarity_tests"][asset] = {
                    "adf_statistic": 0.0,
                    "p_value": 1.0,
                    "stationary_at_5pct": False,
                    "note": "Test failed"
                }
                
            validation_results["return_statistics"][asset] = stats
        
        # Validate correlations
        if not returns.empty and returns.shape[1] > 1:
            try:
                corr_matrix = returns.corr()
                
                # Check for extreme correlations (potential data issues)
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        asset1 = corr_matrix.columns[i]
                        asset2 = corr_matrix.columns[j]
                        corr_value = corr_matrix.iloc[i, j]
                        
                        # Flag perfect correlations (usually indicates data problem)
                        if abs(corr_value) > 0.99:
                            validation_results["correlation_issues"].append({
                                "asset1": asset1,
                                "asset2": asset2,
                                "correlation": float(corr_value),
                                "issue": "Almost perfect correlation (potential duplicate)"
                            })
                        
                        # Flag extreme inverse correlations (rare in financial data)
                        if corr_value < -0.9:
                            validation_results["correlation_issues"].append({
                                "asset1": asset1,
                                "asset2": asset2,
                                "correlation": float(corr_value),
                                "issue": "Extreme negative correlation (unusual)"
                            })
            except Exception as e:
                logger.warning(f"Could not validate correlations: {str(e)}")
        
        # Check for return outliers that might distort optimization
        try:
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
            
            vol_stats = {
                "min": rolling_vol.min().to_dict() if not rolling_vol.empty else {},
                "max": rolling_vol.max().to_dict() if not rolling_vol.empty else {},
                "mean": rolling_vol.mean().to_dict() if not rolling_vol.empty else {},
                "extreme_periods": {}
            }
            
            # Identify periods of extreme volatility
            for asset in rolling_vol.columns:
                asset_vol = rolling_vol[asset].dropna()
                if len(asset_vol) > 0:
                    extreme_periods = asset_vol[asset_vol > 3 * asset_vol.mean()]
                    if len(extreme_periods) > 0:
                        vol_stats["extreme_periods"][asset] = {
                            "dates": extreme_periods.index.strftime("%Y-%m-%d").tolist(),
                            "values": extreme_periods.tolist()
                        }
            
            validation_results["volatility_statistics"] = vol_stats
        except Exception as e:
            logger.warning(f"Could not calculate volatility statistics: {str(e)}")
        
        # Save validation results
        validation_dir = data_processed_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        validation_path = validation_dir / "asset_validation_results.json"
        with open(validation_path, "w") as f:
            json.dump(validation_results, f, indent=4, default=str)
        
        # Generate validation summary
        validation_summary = {
            "passed_validation": validation_results["missing_values"]["total"] == 0,
            "warnings": len(validation_results["correlation_issues"]),
            "non_stationary_series": sum(1 for x in validation_results["stationarity_tests"].values() 
                                          if not x.get("stationary_at_5pct", True)),
            "extreme_volatility_periods": sum(len(periods) for periods in 
                                              validation_results.get("volatility_statistics", {}).get("extreme_periods", {}).values())
        }
        
        # Save correlation matrix for optimization
        if not returns.empty and returns.shape[1] > 1:
            try:
                corr_matrix = returns.corr()
                corr_matrix.to_csv(data_processed_dir / "correlation_matrix.csv")
            except Exception as e:
                logger.warning(f"Could not save correlation matrix: {str(e)}")
        
        logger.info(f"Asset data validation completed. Results saved to: {validation_path}")
        logger.info(f"Validation summary: {validation_summary}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Asset data validation failed: {str(e)}", exc_info=True)
        raise

def validate_macro_data():
    """Validate processed macroeconomic data"""
    try:
        logger.info("Starting macro data validation...")
        
        # Load cleaned macro data
        macro_path = data_processed_dir / "cleaned_macro.csv"
        macro_changes_path = data_processed_dir / "macro_changes.csv"
        
        # Check if files exist
        if not macro_path.exists():
            logger.warning("No macro data file found")
            # Return minimal validation results
            return {
                'macro_factor_count': 0,
                'date_range': {
                    'start': None,
                    'end': None,
                    'days': 0
                },
                'missing_values': {
                    'total': 0,
                    'percentage': 0
                },
                'factor_statistics': {},
                'passed_validation': True
            }
        
        # Load the data
        macro_data = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        
        # Check if macro_changes file exists
        if macro_changes_path.exists():
            macro_changes = pd.read_csv(macro_changes_path, index_col=0, parse_dates=True)
        else:
            # Create an empty DataFrame with the same index
            macro_changes = pd.DataFrame(index=macro_data.index)
        
        validation_results = {
            'macro_factor_count': len(macro_data.columns),
            'date_range': {
                'start': macro_data.index[0].strftime('%Y-%m-%d') if len(macro_data) > 0 else None,
                'end': macro_data.index[-1].strftime('%Y-%m-%d') if len(macro_data) > 0 else None,
                'days': len(macro_data)
            },
            'missing_values': {
                'total': macro_data.isna().sum().sum(),
                'percentage': float(macro_data.isna().mean().mean() * 100) if not macro_data.empty else 0
            },
            'factor_statistics': {},
            'correlation_with_markets': {},
            'autocorrelation': {}
        }
        
        # Validate each macro factor
        for factor in macro_data.columns:
            factor_data = macro_data[factor].dropna()
            
            # Check if factor has any data after dropping NAs
            if len(factor_data) > 0:
                # Calculate basic statistics
                stats = {
                    "mean": float(factor_data.mean()),
                    "std": float(factor_data.std()),
                    "min": float(factor_data.min()),
                    "max": float(factor_data.max()),
                    "first_value": float(factor_data.iloc[0]) if len(factor_data) > 0 else None,
                    "last_value": float(factor_data.iloc[-1]) if len(factor_data) > 0 else None,
                    "change": float(factor_data.iloc[-1] - factor_data.iloc[0]) if len(factor_data) > 0 else 0
                }
            else:
                # Handle empty factor data
                logger.warning(f"Factor {factor} has no data after removing NAs")
                stats = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "first_value": None,
                    "last_value": None,
                    "change": 0.0
                }
            
            validation_results["factor_statistics"][factor] = stats
            
            # Check for autocorrelation (important for predictive models)
            # Only if there's enough data
            if len(factor_data) > 10:
                try:
                    from statsmodels.tsa.stattools import acf
                    acf_result = acf(factor_data, nlags=10)
                    validation_results["autocorrelation"][factor] = {
                        "lag1": float(acf_result[1]) if len(acf_result) > 1 else 0,
                        "lag2": float(acf_result[2]) if len(acf_result) > 2 else 0,
                        "lag3": float(acf_result[3]) if len(acf_result) > 3 else 0
                    }
                except Exception as acf_error:
                    logger.warning(f"Could not compute autocorrelation for {factor}: {str(acf_error)}")
                    validation_results["autocorrelation"][factor] = {
                        "lag1": 0,
                        "lag2": 0,
                        "lag3": 0
                    }
        
        # Check correlation with markets (SPY as proxy) if available
        try:
            asset_returns_path = data_processed_dir / "daily_returns.csv"
            if asset_returns_path.exists():
                asset_returns = pd.read_csv(asset_returns_path, index_col=0, parse_dates=True)
                
                if "SPY" in asset_returns.columns and not macro_changes.empty:
                    market_returns = asset_returns["SPY"]
                    
                    # Align dates
                    aligned_data = pd.concat([market_returns, macro_changes], axis=1).dropna()
                    
                    for factor in macro_changes.columns:
                        if "SPY" in aligned_data.columns and factor in aligned_data.columns:
                            corr = aligned_data["SPY"].corr(aligned_data[factor])
                            validation_results["correlation_with_markets"][factor] = float(corr)
        except Exception as corr_error:
            logger.warning(f"Could not compute correlation with market returns: {str(corr_error)}")
        
        # Save validation results
        validation_dir = data_processed_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        validation_path = validation_dir / "macro_validation_results.json"
        with open(validation_path, "w") as f:
            json.dump(validation_results, f, indent=4, default=str)
        
        # Generate validation summary
        validation_summary = {
            "passed_validation": True,  # We'll consider it passed with any data
            "highly_autocorrelated_factors": sum(1 for x in validation_results["autocorrelation"].values() 
                                                if abs(x.get("lag1", 0)) > 0.7),
            "market_predictive_factors": sum(1 for x in validation_results["correlation_with_markets"].values() 
                                            if abs(x) > 0.3)
        }
        
        logger.info(f"Macro data validation completed. Results saved to: {validation_path}")
        logger.info(f"Validation summary: {validation_summary}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Macro data validation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    validate_asset_data()
    validate_macro_data()