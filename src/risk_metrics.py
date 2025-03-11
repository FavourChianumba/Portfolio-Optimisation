import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from scipy import stats
from config.settings import settings
from utils.logger import setup_logger

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Setup directories
data_processed_dir = project_root / settings.DATA_PROCESSED_DIR
risk_dir = data_processed_dir / "risk_metrics"
risk_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)

class RiskMetrics:
    """Calculate risk and performance metrics for assets and portfolios"""
    
    def __init__(self, returns_data=None, prices_data=None):
        """Initialize with returns and/or price data"""
        self.returns = returns_data
        self.prices = prices_data
        self.risk_free_rate = settings.RISK_FREE_RATE
        
        # Load data if not provided
        if self.returns is None:
            returns_path = data_processed_dir / "daily_returns.csv"
            self.returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        if self.prices is None:
            prices_path = data_processed_dir / "cleaned_asset_prices.csv" 
            self.prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    
    def calculate_all_metrics(self):
        """Calculate and save all risk metrics"""
        try:
            logger.info("Calculating asset risk metrics...")
            
            # Core volatility metrics
            vol_metrics = self.calculate_volatility_metrics()
            
            # Tail risk and drawdown metrics
            tail_metrics = self.calculate_tail_risk()
            drawdown_metrics = self.calculate_drawdowns()
            
            # Performance metrics
            perf_metrics = self.calculate_performance_metrics()
            
            # Combined metrics
            all_metrics = {}
            for asset in self.returns.columns:
                all_metrics[asset] = {
                    **vol_metrics.get(asset, {}),
                    **tail_metrics.get(asset, {}),
                    **drawdown_metrics.get(asset, {}),
                    **perf_metrics.get(asset, {})
                }
            
            # Save metrics
            metrics_path = risk_dir / "asset_risk_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(all_metrics, f, indent=4, default=str)
            
            # Create summary table
            metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
            metrics_df.to_csv(risk_dir / "asset_risk_metrics.csv")
            
            logger.info(f"Risk metrics calculation completed. Results saved to: {metrics_path}")
            return all_metrics
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {str(e)}", exc_info=True)
            raise
    
    def calculate_volatility_metrics(self):
        """Calculate volatility-based risk metrics"""
        metrics = {}
        
        for asset in self.returns.columns:
            asset_returns = self.returns[asset].dropna()
            
            # Basic volatility (annualized)
            daily_vol = asset_returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            
            # Rolling volatility at different windows
            rolling_vol_21 = asset_returns.rolling(window=21).std() * np.sqrt(252)  # ~1 month
            rolling_vol_63 = asset_returns.rolling(window=63).std() * np.sqrt(252)  # ~3 months
            
            # EWMA volatility (more reactive to recent changes)
            ewma_vol = asset_returns.ewm(span=30).std() * np.sqrt(252)
            
            # Upside/downside volatility
            upside_returns = asset_returns[asset_returns > 0]
            downside_returns = asset_returns[asset_returns < 0]
            
            upside_vol = upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else np.nan
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else np.nan
            
            # Volatility of volatility (vol-of-vol)
            vol_of_vol = rolling_vol_21.std() / rolling_vol_21.mean() if len(rolling_vol_21.dropna()) > 0 else np.nan
            
            metrics[asset] = {
                "daily_volatility": daily_vol,
                "annual_volatility": annual_vol,
                "recent_volatility": rolling_vol_21.iloc[-1] if len(rolling_vol_21.dropna()) > 0 else np.nan,
                "quarterly_volatility": rolling_vol_63.iloc[-1] if len(rolling_vol_63.dropna()) > 0 else np.nan,
                "ewma_volatility": ewma_vol.iloc[-1] if len(ewma_vol.dropna()) > 0 else np.nan,
                "upside_volatility": upside_vol,
                "downside_volatility": downside_vol,
                "volatility_ratio": downside_vol / upside_vol if upside_vol > 0 else np.nan,
                "volatility_of_volatility": vol_of_vol
            }
        
        # Save volatility metrics specifically
        vol_metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
        vol_metrics_df.to_csv(risk_dir / "volatility_metrics.csv")
        
        return metrics
    
    def calculate_tail_risk(self):
        """Calculate tail risk metrics"""
        metrics = {}
        
        for asset in self.returns.columns:
            asset_returns = self.returns[asset].dropna()
            
            # Value at Risk (VaR)
            var_95 = np.percentile(asset_returns, 5)
            var_99 = np.percentile(asset_returns, 1)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = asset_returns[asset_returns <= var_95].mean()
            cvar_99 = asset_returns[asset_returns <= var_99].mean()
            
            # Skewness and Kurtosis
            skew = asset_returns.skew()
            kurt = asset_returns.kurtosis()
            
            # Maximum daily loss
            max_loss = asset_returns.min()
            
            # Calculate Modified VaR using Cornish-Fisher expansion
            # Accounts for non-normality in returns
            try:
                z_score = stats.norm.ppf(0.05)
                modified_var_95 = -(asset_returns.mean() + 
                                    z_score * asset_returns.std() * 
                                    (1 + (skew * z_score) / 6 + 
                                     ((kurt - 3) * z_score**2) / 24 - 
                                     ((skew**2) * z_score**3) / 36))
            except:
                modified_var_95 = np.nan
            
            metrics[asset] = {
                "var_95_daily": -var_95,  # Convert to positive for easier interpretation
                "var_99_daily": -var_99,
                "cvar_95_daily": -cvar_95,
                "cvar_99_daily": -cvar_99,
                "modified_var_95_daily": modified_var_95,
                "skewness": skew,
                "excess_kurtosis": kurt,
                "maximum_daily_loss": -max_loss
            }
        
        # Save tail risk metrics specifically
        tail_metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
        tail_metrics_df.to_csv(risk_dir / "tail_risk_metrics.csv")
        
        return metrics
    
    def calculate_drawdowns(self):
        """Calculate drawdown metrics"""
        metrics = {}
        
        for asset in self.prices.columns:
            asset_prices = self.prices[asset].dropna()
            
            # Calculate running maximum
            running_max = asset_prices.cummax()
            
            # Calculate drawdowns
            drawdowns = (asset_prices / running_max - 1)
            
            # Maximum drawdown
            max_drawdown = drawdowns.min()
            
            # Identify worst drawdown period
            worst_dd_end = drawdowns.idxmin()
            
            # Find the start of this drawdown (last time we were at a peak)
            worst_dd_start = asset_prices[:worst_dd_end].idxmax()
            
            # Calculate recovery time (if recovered)
            try:
                recovery_date = asset_prices[worst_dd_end:][asset_prices[worst_dd_end:] >= asset_prices[worst_dd_start]].index[0]
                recovery_time = (recovery_date - worst_dd_end).days
            except:
                recovery_time = np.nan  # Still hasn't recovered
            
            # Calculate time underwater (in days)
            underwater_time = (asset_prices < running_max).sum()
            
            # Average drawdown
            avg_drawdown = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0
            
            # Save drawdown time series for visualization
            drawdowns.to_csv(risk_dir / f"{asset}_drawdowns.csv")
            
            metrics[asset] = {
                "maximum_drawdown": max_drawdown,
                "average_drawdown": avg_drawdown,
                "worst_dd_start": worst_dd_start,
                "worst_dd_end": worst_dd_end,
                "worst_dd_duration_days": (worst_dd_end - worst_dd_start).days,
                "worst_dd_recovery_days": recovery_time,
                "underwater_days": underwater_time,
                "underwater_ratio": underwater_time / len(asset_prices) if len(asset_prices) > 0 else np.nan
            }
        
        # Save drawdown metrics specifically
        dd_metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
        dd_metrics_df.to_csv(risk_dir / "drawdown_metrics.csv")
        
        return metrics
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics"""
        metrics = {}
        
        for asset in self.returns.columns:
            asset_returns = self.returns[asset].dropna()
            
            # Annualized return
            total_return = (1 + asset_returns).prod() - 1
            days = len(asset_returns)
            annualized_return = (1 + total_return) ** (252 / days) - 1
            
            # Sharpe ratio
            excess_return = asset_returns - (self.risk_free_rate / 252)
            sharpe = excess_return.mean() / excess_return.std() * np.sqrt(252)
            
            # Sortino ratio (penalizes only downside volatility)
            downside_returns = asset_returns[asset_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252)
            sortino = (annualized_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else np.nan
            
            # Calmar ratio (return / max drawdown)
            calmar = (annualized_return - self.risk_free_rate) / -self.calculate_drawdowns()[asset]["maximum_drawdown"] \
                        if self.calculate_drawdowns()[asset]["maximum_drawdown"] < 0 else np.nan
            
            # Information ratio (vs. market benchmark - assuming SPY is in the data)
            if "SPY" in self.returns.columns:
                benchmark_returns = self.returns["SPY"].dropna()
                # Align dates
                aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
                tracking_error = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).std() * np.sqrt(252)
                information_ratio = (annualized_return - 
                                    ((1 + benchmark_returns).prod() - 1) ** (252 / len(benchmark_returns)) - 1) / \
                                    tracking_error if tracking_error > 0 else np.nan
            else:
                information_ratio = np.nan
            
            # Positive periods ratio
            positive_periods = (asset_returns > 0).sum() / len(asset_returns)
            
            metrics[asset] = {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "calmar_ratio": calmar,
                "information_ratio": information_ratio,
                "positive_periods_ratio": positive_periods,
                "return_volatility_ratio": annualized_return / (asset_returns.std() * np.sqrt(252))
                                           if asset_returns.std() > 0 else np.nan
            }
        
        # Save performance metrics specifically
        perf_metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
        perf_metrics_df.to_csv(risk_dir / "performance_metrics.csv")
        
        return metrics
    
    def calculate_portfolio_metrics(self, weights):
        """Calculate risk metrics for a portfolio with given weights"""
        # Ensure weights and returns are aligned
        assets = list(weights.keys())
        weights_array = np.array([weights[asset] for asset in assets])
        portfolio_returns = self.returns[assets]
        
        # Calculate portfolio return series
        portfolio_return_series = portfolio_returns.dot(weights_array)
        
        # Calculate mean return and volatility
        mean_return = portfolio_return_series.mean() * 252  # Annualized
        volatility = portfolio_return_series.std() * np.sqrt(252)  # Annualized
        
        # Calculate drawdowns
        portfolio_value = (1 + portfolio_return_series).cumprod()
        running_max = portfolio_value.cummax()
        drawdowns = (portfolio_value / running_max - 1)
        max_drawdown = drawdowns.min()
        
        # Sharpe and Sortino ratios
        sharpe = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else np.nan
        downside_returns = portfolio_return_series[portfolio_return_series < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (mean_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else np.nan
        
        # Calmar ratio
        calmar = (mean_return - self.risk_free_rate) / -max_drawdown if max_drawdown < 0 else np.nan
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_return_series, 5)
        cvar_95 = portfolio_return_series[portfolio_return_series <= var_95].mean()
        
        return {
            "mean_return": mean_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "var_95_daily": -var_95,
            "cvar_95_daily": -cvar_95,
            "return_series": portfolio_return_series
        }