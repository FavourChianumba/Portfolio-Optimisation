# Modern Portfolio Optimization Toolkit

A comprehensive toolkit for portfolio optimization using Modern Portfolio Theory and advanced financial techniques. This system offers end-to-end functionality from data collection to interactive visualization, helping you create optimized investment portfolios with rigorous risk management.

![Portfolio Optimization](https://via.placeholder.com/800x400.png?text=Portfolio+Optimization+Dashboard)

## Table of Contents

- [Features](#features)
- [Quick Start Guide](#quick-start-guide)
- [Detailed Setup Instructions](#detailed-setup-instructions)
- [Running the System](#running-the-system)
- [Project Structure](#project-structure)
- [Working with the Code](#working-with-the-code)
- [Data Sources and Configuration](#data-sources-and-configuration)
- [Visualization and Dashboards](#visualization-and-dashboards)
- [Extending the System](#extending-the-system)
- [Interpreting Results](#interpreting-results)
- [Troubleshooting](#troubleshooting)
- [API Issues](#api-issues)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Data Collection** - Automated gathering of asset prices and macroeconomic indicators
- **Risk Analysis** - Comprehensive metrics including volatility, drawdowns, VaR, and tail risk
- **Portfolio Optimization** - Efficient frontier generation, max Sharpe ratio, min volatility, risk parity
- **Monte Carlo Simulation** - Future performance projections with multiple simulation techniques
- **Backtesting Engine** - Historical performance analysis with various allocation strategies
- **Macro Factor Overlays** - Dynamic allocations based on economic conditions
- **Tableau Integration** - Data preparation for interactive dashboards
- **Modular Architecture** - Easily extensible to add new assets, strategies, or metrics
- **Robust Fallbacks** - Automatic generation of synthetic data when APIs fail

## Quick Start Guide

### Prerequisites

- **Python 3.8+ installed** (Python 2.7 is not supported)
- Git installed
- Required packages: pandas, numpy, matplotlib, scipy, requests, certifi
- [Optional] Docker installed for containerized deployment
- [Optional] Tableau Desktop for visualization (or use the included matplotlib visualizations)

### Installation in 5 Minutes

1. **Check your Python version** first:
   ```bash
   python --version  # Should be 3.8 or higher
   ```
   
   If this returns Python 2.x or a version below 3.8, use a specific Python command:
   ```bash
   python3 --version  # Or try python3.9, python3.11, etc.
   ```
   
   Use the appropriate Python command that returns 3.8+ in the next steps.

2. Clone the repository and navigate to the project:
   ```bash
   git clone https://github.com/favourchianumba/portfolio-optimization.git
   cd portfolio-optimization
   ```

3. Initialize the project (creates directories, virtual environment, and installs dependencies):
   ```bash
   chmod +x init.sh
   
   # Use the correct Python command (replace python3 with specific version if needed)
   PYTHON_CMD=python3 ./init.sh
   ```

4. Create your environment file and add required API keys:
   ```bash
   # Skip if init.sh did this already
   cp .env.template .env
   
   # Edit .env with your API keys and settings
   nano .env
   ```

5. Install SSL certificates (especially important for macOS users):
   ```bash
   # For macOS:
   cd /Applications/Python\ 3.x/  # Replace 3.x with your version
   ./Install\ Certificates.command
   
   # Alternatively, install certifi:
   pip install --upgrade certifi
   ```

6. Run the complete pipeline:
   ```bash
   # Make sure you're in the virtual environment
   source venv/bin/activate
   
   python main.py
   ```

7. Explore the results in the `dashboard/` directory

## Detailed Setup Instructions

### Setting Up Without Docker

1. Create a virtual environment with the correct Python version:
   ```bash
   # Replace python3 with python3.9 or python3.11 if needed
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create environment file:
   ```bash
   cp .env.template .env
   ```

5. Edit the `.env` file with your settings:
   - Add your FRED API key (get one at https://fred.stlouisfed.org/docs/api/api_key.html)
     - This must be exactly 32 characters and lowercase alphanumeric
   - Adjust risk-free rate and other parameters

### Docker Setup

1. Ensure Docker is installed on your machine

2. Create environment file:
   ```bash
   cp .env.template .env
   ```

3. Edit the `.env` file with your settings

4. Build and run the container:
   ```bash
   # With Docker Compose
   docker-compose up --build
   
   # OR with regular Docker
   docker build -t portfolio-optimization .
   docker run --env-file .env -v $(pwd)/data:/app/data -v $(pwd)/dashboard:/app/dashboard portfolio-optimization
   ```

## Running the System

### Running the Complete Pipeline

To run the entire optimization pipeline:

```bash
python main.py
```

This executes all the following steps in sequence:
1. Data collection from Yahoo Finance and FRED (falls back to synthetic data if APIs fail)
2. Data cleaning and preprocessing
3. Data validation and quality checks
4. Risk metrics calculation
5. Portfolio optimization
6. Monte Carlo simulation
7. Backtesting
8. Dashboard data generation

### Running Individual Components

You can run specific components separately:

```bash
# Data Collection only
python -m src.data_collection

# Risk Metrics only
python -m src.risk_metrics

# Portfolio Optimization only
python -m src.optimization

# Backtesting only
python -m src.backtesting

# Dashboard Generation only
python -m src.dashboard
```

### Interactive Exploration with Jupyter Notebook

For interactive analysis and visualization:

```bash
# Start Jupyter Notebook
jupyter notebook notebooks/exploration.ipynb
```

The included exploration notebook walks through each step of the process interactively, allowing you to experiment with different parameters and visualize the results.

### Troubleshooting Python Version Issues

If you encounter "No module named 'pandas'" or similar errors:

1. Check your Python version in the virtual environment:
   ```bash
   python --version
   ```

2. If needed, recreate the virtual environment with the correct Python version:
   ```bash
   # Deactivate current environment
   deactivate
   
   # Remove existing environment
   rm -rf venv
   
   # Create new environment with correct Python (use the version that works for you)
   python3.9 -m venv venv  # or python3.8, python3.11, etc.
   
   # Activate and reinstall
   source venv/bin/activate
   pip install -r requirements.txt
   ```
## Project Structure

The project is organized into modular components:

```
portfolio-optimization/
├── config/                  # Configuration settings
│   ├── __init__.py
│   └── settings.py          # Global settings
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── logger.py            # Logging setup
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_collection.py   # Data collection
│   ├── data_cleaning.py     # Data preprocessing
│   ├── data_validation.py   # Data quality checks
│   ├── risk_metrics.py      # Risk calculations
│   ├── optimization.py           #Portolio optimization
│   ├── factor_optimization.py 
│   ├── streamlit_dashboard.py    
│   ├── black_litterman.py        
│   ├── monte_carlo.py       # Monte Carlo simulations
│   ├── backtesting.py       # Historical backtesting
│   └── dashboard.py         # Dashboard generation
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_data_validation.py
│   ├── test_optimization.py
│   └── test_risk_metrics.py
├── data/                    # Data storage
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
├── dashboard/               # Dashboard outputs
├── logs/                    # Log files
├── notebooks/               # Jupyter notebooks
│   └── exploration.ipynb    # Interactive analysis notebook
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── .env.template            # Template for environment variables
├── requirements.txt         # Python dependencies
├── setup.py                 # Package installation
├── init.sh                  # Project initialization script
├── main.py                  # Main execution script
└── README.md                # Project documentation
```

## Working with the Code

### Data Collection

The data collection module fetches historical price data from multiple sources (Yahoo Finance, Alpha Vantage, Polygon.io) and macroeconomic indicators from FRED. The system automatically attempts each data source in order and falls back to the next one if any source fails.

#### Customizing Financial Data Collection

1. Edit the `assets` list in `src/data_collection.py`:
   ```python
   assets = [
       "SPY",   # S&P 500
       "QQQ",   # Nasdaq 100
       "TLT",   # Long-Term Treasury
       "GLD",   # Gold
       "AAPL",  # Apple
       # Add your assets here
   ]
   ```

2. Modify the date range:
   ```python
   end_date = datetime.datetime.now() - datetime.timedelta(days=1)
   start_date = end_date - datetime.timedelta(days=365*5)  # 5 years
   ```

3. Add or modify macro factors:
   ```python
   macro_factors = [
       "CPIAUCSL",  # Consumer Price Index
       "UNRATE",    # Unemployment Rate
       "FEDFUNDS",  # Federal Funds Rate
       "^VIX",      # Volatility Index (Yahoo Finance)
       # Add your factors here
   ]
   ```

#### Configuring Data Sources

1. Set your API keys in the `.env` file:
   ```
   FRED_API_KEY=your_fred_api_key
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   POLYGON_API_KEY=your_polygon_key
   ```

2. Customize data source priority (optional):
   ```
   # Comma-separated list: yahoo, alphavantage, polygon, sample
   DATA_SOURCE_PRIORITY=yahoo,polygon,alphavantage,sample
   ```

#### Data Source Features

| Source | API Key Required | Free Tier Limits | Data Coverage | Data Quality Considerations |
|--------|------------------|------------------|---------------|----------------------------|
| Yahoo Finance | No | Unlimited (but less reliable) | Stocks, ETFs, indices | Generally good quality with adjusted prices, occasionally unstable API |
| Polygon.io | Yes | 5 calls/minute | Stocks, options, forex | High-quality data with good reliability |
| Alpha Vantage | Yes | 5 calls/minute, 25 calls/day | Stocks, ETFs, forex, crypto | Free tier uses unadjusted prices; adjusted prices require premium subscription |
| FRED | Yes | 120 calls/minute | Macroeconomic indicators | High-quality official economic data |

> **Note on Alpha Vantage**: We've configured Alpha Vantage to use the free `TIME_SERIES_DAILY` endpoint instead of the premium `TIME_SERIES_DAILY_ADJUSTED` endpoint. This means the data doesn't account for stock splits and dividends, which can impact long-term analysis. Due to this limitation and the strict daily call limit (25/day), Alpha Vantage is positioned as a lower-priority data source in our default configuration.

#### How Fallback Works

The system will:
1. Try the first data source in your priority list
2. If it fails or returns insufficient data (<50% of requested assets), try the next source
3. Continue until a source succeeds or all sources are exhausted
4. Generate synthetic sample data as a last resort for testing purposes

#### Manual Data Source Selection

To manually specify which data source to use:

```python
from src.data_collection import YahooFinanceSource, AlphaVantageSource, PolygonSource

# Initialize a specific source
source = AlphaVantageSource(settings.ALPHA_VANTAGE_API_KEY)

# Get data directly from this source
asset_data = source.get_asset_data(assets, start_date, end_date)
```

### Portfolio Optimization

To customize the optimization process:

1. Change optimization constraints in `config/settings.py`:
   ```python
   self.MIN_WEIGHT = 0.01  # Minimum 1% allocation
   self.MAX_WEIGHT = 0.40  # Maximum 40% allocation
   ```

2. Run a specific optimization method:
   ```python
   from src.optimization import PortfolioOptimizer
   
   optimizer = PortfolioOptimizer()
   
   # For maximum Sharpe ratio
   max_sharpe = optimizer.optimize_sharpe_ratio()
   
   # For minimum volatility
   min_vol = optimizer.optimize_minimum_volatility()
   
   # For risk parity
   risk_parity = optimizer.optimize_risk_parity()
   ```

## Advanced Optimization Techniques

The toolkit also provides advanced portfolio optimization methods that extend beyond classic Modern Portfolio Theory. These techniques address key limitations of standard mean-variance optimization and are particularly valuable for institutional investors or complex investment scenarios.

The following advanced methods can be used alongside or as alternatives to the standard optimization approaches:

# Factor-Based Portfolio Optimization

The Modern Portfolio Optimization Toolkit now includes factor-based portfolio optimization, which extends beyond classic Modern Portfolio Theory to incorporate factor models similar to those used by institutional investors.

## Overview

Factor models explain asset returns using common risk factors, providing a more nuanced understanding of portfolio risk and return characteristics. Key benefits include:

- **Better risk decomposition**: Understand the drivers of portfolio risk
- **Style analysis**: Target specific investment styles (value, momentum, etc.)
- **Enhanced diversification**: Diversify across risk factors rather than just assets
- **Improved forecasting**: Use factor structure for more reliable return predictions

## Supported Features

1. **Factor Model Generation**
   - Automatic generation of key factors (Market, Size, Momentum, Volatility)
   - Factor regression analysis to determine asset exposures

2. **Factor-Based Portfolio Optimization**
   - Maximum Sharpe ratio optimization using factor expected returns
   - Minimum variance optimization with factor-based covariance
   - Mean-variance utility optimization with risk aversion parameter

3. **Factor-Tilted Portfolios**
   - Create portfolios with targeted factor exposures
   - Flexible constraint system for customized factor tilts
   - Tracking error minimization relative to benchmark

4. **Portfolio Analysis**
   - Decompose existing portfolios into factor exposures
   - Visualize factor tilts and attributions
   - Compare factor exposures between portfolios

## Usage Examples

### Basic Factor Optimization

```python
from src.factor_optimization import FactorOptimizer

# Initialize with returns data
optimizer = FactorOptimizer(returns_data=returns)

# Maximum Sharpe ratio optimization using factor model
max_sharpe = optimizer.optimize_factor_portfolio(objective='sharpe')

# Minimum variance optimization using factor model
min_var = optimizer.optimize_factor_portfolio(objective='min_variance')
```

### Creating Factor-Tilted Portfolios

```python
# Create a momentum-tilted portfolio
momentum_portfolio = optimizer.optimize_factor_tilted_portfolio(
    target_factor_exposures={'MOM': 0.2},
    objective='max_return'
)

# Create a multi-factor portfolio
multi_factor = optimizer.optimize_factor_tilted_portfolio(
    target_factor_exposures={
        'MKT': 1.0,
        'SMB': 0.2,
        'MOM': 0.2,
        'VOL': 0.2
    },
    objective='min_tracking_error'
)
```

### Analyzing Existing Portfolios

```python
# Analyze factor exposures of an existing portfolio
analysis = optimizer.analyze_portfolio_factor_exposures(
    weights={'AAPL': 0.2, 'MSFT': 0.2, 'SPY': 0.6}
)

# View factor exposures
print("Factor Exposures:")
for factor, exposure in analysis['factor_exposures'].items():
    print(f"  {factor}: {exposure:.2f}")
```

## Factor Descriptions

The system uses the following factors:

- **MKT (Market)**: Overall market risk premium
- **SMB (Size)**: Small minus big companies (size premium)
- **MOM (Momentum)**: Winners minus losers performance
- **VOL (Volatility)**: Low volatility minus high volatility assets

## Implementation Notes

- The factor model uses OLS regression to determine asset exposures to factors
- Covariance estimation uses factor structure to reduce estimation noise
- Optimization constraints ensure portfolio weights sum to 1
- Default weight bounds of 1%-40% apply (can be customized)

Factor-based optimization often results in more intuitive, balanced portfolios than traditional mean-variance optimization, with greater stability in weights over time.

# Black-Litterman Portfolio Optimization

## Overview

The Black-Litterman (BL) model is an advanced portfolio optimization framework that addresses key limitations of traditional Mean-Variance Optimization by incorporating investor views into market equilibrium returns. This implementation enables you to:

- Combine market-implied returns with your own investment views
- Express views with different confidence levels
- Generate more stable, intuitive portfolio allocations
- Reduce the sensitivity to input estimation errors

## Key Features

1. **Market Equilibrium Integration**
   - Uses reverse optimization to extract implied returns from market weights
   - Provides a stable starting point for expected returns

2. **Flexible View Specification**
   - Absolute views: Direct return expectations for specific assets
   - Relative views: Expected outperformance between assets
   - Confidence levels: Specify uncertainty in each view

3. **Posterior Estimation**
   - Blends prior (equilibrium) returns with investor views
   - Accounts for confidence levels and correlation structure
   - Produces updated expected returns and covariance matrix

4. **Portfolio Optimization**
   - Maximum Sharpe ratio optimization
   - Minimum variance optimization
   - Mean-variance utility optimization with risk aversion parameter

## Usage Examples

### Basic Usage

```python
from src.black_litterman import BlackLittermanOptimizer

# Initialize optimizer
bl = BlackLittermanOptimizer(returns_data=returns)

# Add investor views
bl.add_absolute_view(asset="AAPL", return_view=0.15, confidence=0.7)  # 15% return with 70% confidence
bl.add_relative_view(asset1="MSFT", asset2="GOOGL", return_diff=0.03, confidence=0.6)  # MSFT outperforms GOOGL by 3%

# Compute posterior estimates
bl.compute_posterior()

# Optimize portfolio
portfolio = bl.optimize_portfolio(objective='sharpe')
```

### Advanced Usage

```python
# Customize market weights
market_weights = {'SPY': 0.4, 'QQQ': 0.3, 'AAPL': 0.15, 'MSFT': 0.15}

# Customize model parameters
bl = BlackLittermanOptimizer(
    returns_data=returns,
    market_weights=market_weights,
    risk_aversion=2.5,  # Lambda parameter
    tau=0.025  # Confidence in prior
)

# Add multiple views
bl.add_absolute_view("AAPL", 0.20, 0.8)
bl.add_absolute_view("MSFT", 0.15, 0.7)
bl.add_relative_view("AAPL", "GOOGL", 0.05, 0.6)

# Compute and analyze
bl.compute_posterior()
comparison = bl.compare_prior_posterior()

# Optimize with different objectives
sharpe_portfolio = bl.optimize_portfolio(objective='sharpe')
minvar_portfolio = bl.optimize_portfolio(objective='min_variance')
utility_portfolio = bl.optimize_portfolio(objective='utility', risk_aversion=3.0)
```

## Mathematical Background

The Black-Litterman model uses Bayesian inference to combine:

1. **Prior distribution** (market equilibrium returns)
   - π = λΣw_mkt  (reverse optimization)
   - Where λ is risk aversion, Σ is covariance, w_mkt is market weights

2. **Investor views**
   - P × r = q + ε  
   - Where P is the pick matrix, q is view returns, ε is uncertainty

3. **Posterior distribution**
   - E[r] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) × [(τΣ)^(-1)π + P'Ω^(-1)q]
   - Where τ is prior uncertainty, Ω is view uncertainty matrix

## Best Practices

1. **Setting View Confidence**
   - High confidence (0.7-1.0): Strong conviction based on thorough analysis
   - Medium confidence (0.4-0.7): Moderate conviction
   - Low confidence (0.1-0.4): Speculative views

2. **Mixing View Types**
   - Use absolute views for specific return forecasts
   - Use relative views for sector rotations or pairs trading ideas

3. **View Consistency**
   - Avoid contradictory views (the model will blend them based on confidence)
   - Consider correlations between assets when setting views

4. **Interpretation**
   - Compare prior and posterior distributions to understand view impact
   - Examine weight changes relative to market portfolio

5. **Dealing with Uncertainty**
   - Higher view confidence (lower uncertainty) creates larger deviations from market weights
   - The tau parameter controls overall shrinkage toward prior

### Running Tests

To run the test suite:

```bash
# Run all tests
python -m unittest discover tests

# Run a specific test file
python -m unittest tests.test_optimization

# Run a specific test case
python -m unittest tests.test_optimization.TestOptimization.test_optimize_sharpe_ratio
```

## Data Sources and Configuration

### Data Sources

The system uses the following data sources:

1. **Yahoo Finance** - For historical asset prices (stocks, ETFs, etc.)
2. **FRED** - For macroeconomic indicators (inflation, interest rates, etc.)
3. **Synthetic Data Generation** - Automatically generates realistic market data when APIs fail

## Configuration

The main configuration parameters are in:

1. **Environment Variables** (`.env`):
   - `FRED_API_KEY` - Your FRED API key (32-character alphanumeric string)
   - `ALPHA_VANTAGE_API_KEY` - Your Alpha Vantage API key
   - `POLYGON_API_KEY` - Your Polygon.io API key
   - `RISK_FREE_RATE` - Annual risk-free rate (default: 0.035 or 3.5%)
   - Other settings that may change between environments

2. **Settings Class** (`config/settings.py`):
   - `MIN_WEIGHT`/`MAX_WEIGHT` - Portfolio allocation constraints
   - `OUTLIER_THRESHOLD` - Z-score threshold for detecting outliers
   - `MAX_MISSING_PCT` - Maximum percentage of missing values allowed
   - `YFINANCE_TIMEOUT` - Timeout for Yahoo Finance API requests

### Logs

Log files are stored in the `logs/` directory:
- `main.log` - Main pipeline execution log
- `data_collection.log` - Data collection specific logs
- `optimization.log` - Optimization specific logs
- Other component-specific logs

## Visualization and Dashboards

### Built-in Visualizations

Basic visualizations are automatically generated in the `dashboard/` directory:
- Efficient frontier plots
- Portfolio allocation charts
- Backtesting performance comparisons
- Risk metrics heatmaps
- Monte Carlo projection graphs

### Tableau Integration

For more advanced visualizations with Tableau:

1. Connect Tableau to the CSV files in the `dashboard/` directory
2. Follow the guidance in `dashboard/dashboard_setup_guide.md`
3. Key visualizations to create:
   - Efficient Frontier: Risk-Return scatter plot
   - Asset Allocation: Pie or stacked bar chart
   - Strategy Comparison: Line chart of cumulative returns
   - Risk Analysis: Heatmap of risk metrics

### Jupyter Notebook Exploration

The included `notebooks/exploration.ipynb` provides an interactive way to:
- Visualize data and correlations
- Explore risk-return characteristics
- Generate and analyze efficient frontiers
- Implement advanced optimization techniques (Factor-Based and Black-Litterman models)
- Simulate portfolio performance with parallel processing
- Compare backtesting results across multiple strategies
- Generate investment recommendations for different risk profiles

## Extending the System

### Adding New Assets

To add new assets or asset classes:

1. Edit the `assets` list in `src/data_collection.py`
2. Run the data collection process:
   ```bash
   python -m src.data_collection
   ```

### Implementing New Strategies

To add a new optimization strategy:

1. Add a new method to the `PortfolioOptimizer` class in `src/optimization.py`
2. Incorporate the strategy into the backtesting comparison in `src/backtesting.py`

Example of a custom strategy implementation:

```python
def optimize_minimum_tail_risk(self, save_results=True):
    """Optimize portfolio to minimize tail risk (CVaR/Expected Shortfall)"""
    # Implementation details...
    return optimized_portfolio
```

### Creating Custom Risk Metrics

To add new risk metrics:

1. Add a new method to the `RiskMetrics` class in `src/risk_metrics.py`
2. Incorporate the metrics into the dashboard generation in `src/dashboard.py`

## Interpreting Results

The optimization process generates several key metrics to help you understand and evaluate your portfolios:

1. **Sharpe Ratio**: Measures risk-adjusted return. Higher is better, with values above 1.0 generally considered good.
   - Max Sharpe strategy typically produces the highest Sharpe ratio (often 2.0+)
   - This indicates strong returns relative to the risk taken

2. **Portfolio Allocation**: Shows the optimal asset weights for each strategy.
   - Max Sharpe portfolios often concentrate in high-performing assets
   - Min Volatility and Risk Parity typically have more diversified allocations

3. **Monte Carlo Projections**: Provide probabilistic future performance estimates.
   - The median projection is the most likely outcome
   - The 95th percentile shows potential upside in favorable conditions
   - The 5th percentile helps assess downside risk

4. **Backtesting Results**: Compare how strategies would have performed historically.
   - Look at total return as well as risk-adjusted metrics
   - Pay attention to maximum drawdown as a measure of downside risk

When evaluating the results, consider:
- Your risk tolerance (Min Volatility may be better for risk-averse investors)
- Investment horizon (longer horizons can tolerate more volatility)
- Need for diversification (Risk Parity offers better risk distribution)

## Troubleshooting

### Common Issues

1. **Data Collection Failures**
   - *Symptom*: Error in data_collection.py
   - *Possible Causes*: API rate limits, network issues, invalid tickers, missing API keys
   - *Solution*: 
     - Ensure API keys are correctly set in .env file
     - Add delays between requests with rate limiting
     - Check ticker validity
     - Install SSL certificates for FRED API (see installation step 5)

2. **SSL Certificate Verification Issues**
   - *Symptom*: "SSL: CERTIFICATE_VERIFY_FAILED" errors especially with FRED API
   - *Cause*: Missing or outdated SSL certificates
   - *Solution*:
     - For macOS: Run `/Applications/Python\ 3.x/Install\ Certificates.command`
     - For other systems: `pip install --upgrade certifi`
     - Temporary workaround: Add this code at the top of your script:
       ```python
       import ssl
       ssl._create_default_https_context = ssl._create_unverified_context
       ```

3. **Yahoo Finance API Issues**
   - *Symptom*: "Expecting value: line 1 column 1 (char 0)" errors
   - *Cause*: Yahoo Finance API doesn't have a stable public API
   - *Solution*: 
     - Use alternative data sources like Polygon.io (requires API key)
     - Try yahooquery package instead of yfinance
     - Add user-agent headers to avoid rate limiting
     - The system automatically falls back to other data sources or sample data generation

4. **API Authentication Errors**
   - *Symptom*: "Unauthorized" or "Unknown API Key" errors
   - *Cause*: Invalid or missing API keys
   - *Solution*:
     - Ensure API keys are correctly set in .env file
     - Check that keys are valid and not expired
     - Verify free tier usage limits haven't been exceeded

5. **Missing Data**
   - *Symptom*: Warnings about missing data in logs
   - *Possible Causes*: Some assets have different trading calendars
   - *Solution*: Adjust the `MAX_MISSING_PCT` setting or improve data cleaning

### API Issues

1. **Yahoo Finance API Issues**
   - *Symptom*: "Expecting value: line 1 column 1 (char 0)" errors
   - *Cause*: Yahoo Finance API doesn't have a stable public API
   - *Solution*: The system automatically falls back to sample data generation. For production use, consider using paid APIs like Alpha Vantage or IEX Cloud.

2. **FRED API Key Format**
   - *Symptom*: "Bad Request. The value for variable api_key is not a 32 character alpha-numeric lower-case string."
   - *Cause*: Incorrect FRED API key format
   - *Solution*: Ensure your FRED API key is exactly 32 characters and contains only lowercase alphanumeric characters. Register for an API key at https://fred.stlouisfed.org/docs/api/api_key.html

3. **Alpha Vantage Premium Endpoint Issues**
   - *Symptom*: "Thank you for using Alpha Vantage! This is a premium endpoint..."
   - *Cause*: The free tier Alpha Vantage API key doesn't have access to the adjusted daily data endpoint
   - *Solution*: 
     - We've modified the code to use the free `TIME_SERIES_DAILY` endpoint instead of the premium `TIME_SERIES_DAILY_ADJUSTED` endpoint
     - Be aware this returns unadjusted prices that don't account for dividends and stock splits
     - Due to this data quality limitation and the strict API limits (25 calls/day), Alpha Vantage is set as a lower priority source
     - For more reliable data with adjustments for corporate actions, consider upgrading to Alpha Vantage premium or using Yahoo Finance/Polygon.io

### Getting Help

If you encounter issues:

1. Check the detailed logs in the `logs/` directory
2. Review the error messages in the console output
3. Ensure your environment variables are correctly set
4. Try running individual components to isolate the issue

## License

[MIT License](LICENSE)

## Acknowledgments

- Modern Portfolio Theory by Harry Markowitz
- PyPortfolioOpt library by Robert Martin
- Pandas and NumPy communities