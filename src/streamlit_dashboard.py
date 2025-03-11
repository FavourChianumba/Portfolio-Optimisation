import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import json
import sys
from PIL import Image
from datetime import datetime
from scipy import stats

# Set up debugging
DEBUG_MODE = True

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Setup directory paths - with more robust checking
dashboard_dir = project_root / "dashboard"
data_processed_dir = project_root / "data" / "processed"

# Try alternative path if the first one doesn't exist
if not data_processed_dir.exists():
    data_processed_dir = project_root / "data_processed"
    if not data_processed_dir.exists():
        # One more attempt with settings
        try:
            from config.settings import settings
            data_processed_dir = project_root / settings.DATA_PROCESSED_DIR
        except:
            # Will be handled later with debug info
            pass

# Set up additional directories
optimization_dir = data_processed_dir / "optimization"
risk_dir = data_processed_dir / "risk_metrics"
monte_carlo_dir = data_processed_dir / "monte_carlo"
backtest_dir = data_processed_dir / "backtest"

# Set page configuration
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #1E88E5 !important;
        margin-bottom: 20px !important;
    }
    .subtitle {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: #333 !important;
        margin-top: 30px !important;
        margin-bottom: 15px !important;
    }
    .card {
        border-radius: 5px;
        background-color: #f9f9f9;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        color: #666;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 20px;
        border-left: 5px solid #1E88E5;
        margin: 20px 0;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Create header
def create_header():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<p class="main-title">Portfolio Optimization Dashboard</p>', unsafe_allow_html=True)
    with col2:
        current_date = datetime.now().strftime("%b %d, %Y")
        st.markdown(f"<p style='text-align: right; padding-top: 20px;'>{current_date}</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample data for testing if real data is not available"""
    st.warning("Using sample data for demonstration. Run the optimization pipeline for actual results.")
    
    # Create sample dates
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='B')
    
    # Sample assets
    assets = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GLD']
    
    # Create sample prices and returns
    np.random.seed(42)  # For reproducibility
    sample_prices = pd.DataFrame(index=dates)
    
    for asset in assets:
        # Generate random walk for price
        prices = [100]
        for i in range(1, len(dates)):
            # Different characteristics for different assets
            if asset == 'SPY':
                mu, sigma = 0.0006, 0.010  # Lower volatility
            elif asset == 'AAPL' or asset == 'MSFT':
                mu, sigma = 0.0008, 0.015  # Higher return, higher volatility
            elif asset == 'QQQ':
                mu, sigma = 0.0007, 0.012  # Middle ground
            else:  # GLD
                mu, sigma = 0.0004, 0.008  # Low correlation to equities
                
            prices.append(prices[-1] * (1 + np.random.normal(mu, sigma)))
        sample_prices[asset] = prices
    
    # Calculate returns
    sample_returns = sample_prices.pct_change().dropna()
    
    # Create sample data dictionary
    sample_data = {
        'returns': sample_returns,
        'prices': sample_prices,
    }
    
    # Add efficient frontier data
    vol_range = np.linspace(0.1, 0.3, 100)
    ret_range = 0.05 + 0.5 * vol_range + np.random.normal(0, 0.01, 100)
    sharpe = ret_range / vol_range
    
    # Random portfolios
    ef_data = pd.DataFrame({
        'Volatility': vol_range,
        'Return': ret_range,
        'Sharpe': sharpe,
        'Type': 'Random Portfolio'
    })
    
    # Add portfolio weights
    max_sharpe_weights = pd.DataFrame({
        'Asset': assets,
        'Weight': [0.3, 0.2, 0.2, 0.2, 0.1],
        'Portfolio': 'Max Sharpe'
    })
    
    min_vol_weights = pd.DataFrame({
        'Asset': assets,
        'Weight': [0.4, 0.1, 0.1, 0.1, 0.3],
        'Portfolio': 'Min Volatility'
    })
    
    sample_data['efficient_frontier'] = ef_data
    sample_data['portfolio_weights'] = pd.concat([max_sharpe_weights, min_vol_weights])
    
    # Add backtest results
    backtest_values = pd.DataFrame(index=dates[1:])
    strategies = ['Equal Weight', 'Max Sharpe', 'Min Volatility', 'Risk Parity']
    
    for strategy in strategies:
        # Generate slightly different performance for each strategy
        if strategy == 'Max Sharpe':
            multiplier = 1.1  # Best performer
        elif strategy == 'Min Volatility':
            multiplier = 0.95  # Lower return but less volatility
        elif strategy == 'Risk Parity':
            multiplier = 1.05  # Middle ground
        else:
            multiplier = 1.0  # Baseline
            
        values = [100000]
        for i in range(1, len(dates)):
            values.append(values[-1] * (1 + np.random.normal(0.0006 * multiplier, 0.01 / multiplier)))
        backtest_values[strategy] = values[1:]  # Skip first value to align with returns
    
    sample_data['backtest_values'] = backtest_values
    
    # Create backtest results dictionary
    backtest_results = {}
    for strategy in strategies:
        values = backtest_values[strategy]
        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(values)) - 1
        returns = values.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility
        
        backtest_results[strategy] = {
            'total_return': float(total_return),
            'annualized_return': float(annual_return),
            'annualized_volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(-0.15 + np.random.normal(0, 0.05)),  # Random drawdown around -15%
            'initial_investment': 100000,
            'final_value': float(values.iloc[-1]),
            'weights': {asset: 1/len(assets) for asset in assets},
            'rebalance_frequency': 'M'
        }
    
    sample_data['backtest_results'] = backtest_results
    
    # Add Monte Carlo stats
    monte_carlo_stats = {}
    for method in ['parametric', 'historical', 'bootstrap']:
        monte_carlo_stats[method] = {
            'mean': float(100000 * (1 + np.random.uniform(0.3, 0.5))),
            'median': float(100000 * (1 + np.random.uniform(0.25, 0.45))),
            'percentile_5': float(100000 * (1 + np.random.uniform(-0.1, 0.1))),
            'percentile_95': float(100000 * (1 + np.random.uniform(0.6, 0.8))),
            'initial_investment': 100000,
            'time_horizon_days': 1260,
            'prob_return_-0.2': float(np.random.uniform(0.02, 0.05)),
            'prob_return_0.0': float(np.random.uniform(0.1, 0.2)),
            'prob_return_0.2': float(np.random.uniform(0.5, 0.7)),
            'prob_return_0.5': float(np.random.uniform(0.3, 0.5)),
            'prob_return_1.0': float(np.random.uniform(0.1, 0.2))
        }
    
    sample_data['monte_carlo_stats'] = monte_carlo_stats
    
    return sample_data

# Load data
@st.cache_data
def load_data():
    """Load all the data needed for the dashboard"""
    data = {}
    debug_info = {}
    
    if DEBUG_MODE:
        # Display path information for debugging
        debug_info["project_root"] = str(project_root)
        debug_info["dashboard_dir_exists"] = dashboard_dir.exists()
        debug_info["data_processed_dir_exists"] = data_processed_dir.exists()
        debug_info["data_processed_dir"] = str(data_processed_dir)
        
        # List files in the data_processed_dir if it exists
        if data_processed_dir.exists():
            debug_info["data_processed_files"] = [str(f) for f in data_processed_dir.glob("*")]
    
    # Load returns and prices
    try:
        returns_path = data_processed_dir / "daily_returns.csv"
        prices_path = data_processed_dir / "cleaned_asset_prices.csv"
        
        if DEBUG_MODE:
            debug_info["returns_path_exists"] = returns_path.exists()
            debug_info["prices_path_exists"] = prices_path.exists()
            debug_info["returns_path"] = str(returns_path)
            debug_info["prices_path"] = str(prices_path)
        
        if not returns_path.exists():
            # Try alternative locations
            alt_paths = [
                project_root / "data" / "daily_returns.csv",
                project_root / "data_processed" / "daily_returns.csv"
            ]
            
            for path in alt_paths:
                if path.exists():
                    returns_path = path
                    if DEBUG_MODE:
                        debug_info["alt_returns_path_found"] = str(path)
                    break
        
        if not prices_path.exists():
            # Try alternative locations
            alt_paths = [
                project_root / "data" / "cleaned_asset_prices.csv",
                project_root / "data_processed" / "cleaned_asset_prices.csv"
            ]
            
            for path in alt_paths:
                if path.exists():
                    prices_path = path
                    if DEBUG_MODE:
                        debug_info["alt_prices_path_found"] = str(path)
                    break
        
        # Load the data if files exist
        if returns_path.exists() and prices_path.exists():
            data['returns'] = pd.read_csv(returns_path, index_col=0, parse_dates=True)
            data['prices'] = pd.read_csv(prices_path, index_col=0, parse_dates=True)
            
            if DEBUG_MODE:
                debug_info["returns_shape"] = data['returns'].shape
                debug_info["prices_shape"] = data['prices'].shape
        else:
            if DEBUG_MODE:
                debug_info["data_load_error"] = "Returns or prices files not found"
            else:
                st.error("Returns or prices data files not found. Please run the optimization pipeline first.")
    except Exception as e:
        if DEBUG_MODE:
            debug_info["data_load_error"] = str(e)
        else:
            st.error(f"Error loading returns or prices data: {str(e)}")
    
    # Load efficient frontier data
    try:
        ef_viz_path = dashboard_dir / "efficient_frontier_viz.csv"
        ef_path = optimization_dir / "efficient_frontier.csv"
        
        # Try multiple paths
        if ef_viz_path.exists():
            data['efficient_frontier'] = pd.read_csv(ef_viz_path)
            if DEBUG_MODE:
                debug_info["efficient_frontier_source"] = "dashboard dir"
        elif ef_path.exists():
            data['efficient_frontier'] = pd.read_csv(ef_path)
            if DEBUG_MODE:
                debug_info["efficient_frontier_source"] = "optimization dir"
        else:
            # Try additional paths
            alt_paths = [
                project_root / "dashboard" / "efficient_frontier_viz.csv",
                project_root / "data" / "processed" / "optimization" / "efficient_frontier.csv"
            ]
            
            for path in alt_paths:
                if path.exists():
                    data['efficient_frontier'] = pd.read_csv(path)
                    if DEBUG_MODE:
                        debug_info["efficient_frontier_source"] = f"alt path: {path}"
                    break
    except Exception as e:
        if DEBUG_MODE:
            debug_info["efficient_frontier_error"] = str(e)
    
    # Load portfolio weights
    try:
        weights_path = dashboard_dir / "portfolio_weights.csv"
        
        if weights_path.exists():
            data['portfolio_weights'] = pd.read_csv(weights_path)
            if DEBUG_MODE:
                debug_info["portfolio_weights_source"] = "dashboard dir"
        else:
            # Try to create from optimization results
            portfolios = []
            
            # Get paths for weight files
            max_sharpe_path = optimization_dir / "max_sharpe_optimized_weights.csv"
            min_vol_path = optimization_dir / "min_vol_optimized_weights.csv"
            risk_parity_path = optimization_dir / "risk_parity_data.csv"
            
            # Load from each path if available
            if max_sharpe_path.exists():
                max_sharpe_df = pd.read_csv(max_sharpe_path)
                max_sharpe_df['Portfolio'] = 'Max Sharpe'
                portfolios.append(max_sharpe_df)
                if DEBUG_MODE:
                    debug_info["max_sharpe_weights_found"] = True
            
            if min_vol_path.exists():
                min_vol_df = pd.read_csv(min_vol_path)
                min_vol_df['Portfolio'] = 'Min Volatility'
                portfolios.append(min_vol_df)
                if DEBUG_MODE:
                    debug_info["min_vol_weights_found"] = True
            
            if risk_parity_path.exists():
                risk_parity_df = pd.read_csv(risk_parity_path)
                if 'Weight' in risk_parity_df.columns:
                    risk_parity_df = risk_parity_df[['Asset', 'Weight']]
                    risk_parity_df['Portfolio'] = 'Risk Parity'
                    portfolios.append(risk_parity_df)
                    if DEBUG_MODE:
                        debug_info["risk_parity_weights_found"] = True
            
            # Combine if any portfolios found
            if portfolios:
                data['portfolio_weights'] = pd.concat(portfolios)
                if DEBUG_MODE:
                    debug_info["portfolio_weights_source"] = "created from optimization files"
    except Exception as e:
        if DEBUG_MODE:
            debug_info["portfolio_weights_error"] = str(e)
    
    # Load backtest results
    try:
        backtest_files = list(backtest_dir.glob("comparison_*_results.json"))
        
        if backtest_files:
            with open(backtest_files[0], 'r') as f:
                data['backtest_results'] = json.load(f)
                if DEBUG_MODE:
                    debug_info["backtest_results_source"] = str(backtest_files[0])
            
            values_file = str(backtest_files[0]).replace("_results.json", "_portfolio_values.csv")
            if os.path.exists(values_file):
                data['backtest_values'] = pd.read_csv(values_file, index_col=0, parse_dates=True)
                if DEBUG_MODE:
                    debug_info["backtest_values_source"] = values_file
        else:
            # Try alternative paths
            alt_path = project_root / "data" / "processed" / "backtest"
            if alt_path.exists():
                alt_files = list(alt_path.glob("comparison_*_results.json"))
                if alt_files:
                    with open(alt_files[0], 'r') as f:
                        data['backtest_results'] = json.load(f)
                    
                    alt_values_file = str(alt_files[0]).replace("_results.json", "_portfolio_values.csv")
                    if os.path.exists(alt_values_file):
                        data['backtest_values'] = pd.read_csv(alt_values_file, index_col=0, parse_dates=True)
                    
                    if DEBUG_MODE:
                        debug_info["backtest_results_source"] = f"alt path: {alt_files[0]}"
    except Exception as e:
        if DEBUG_MODE:
            debug_info["backtest_results_error"] = str(e)
    
    # Load Monte Carlo results
    try:
        monte_carlo_stats = {}
        found_methods = []
        
        for method in ['parametric', 'historical', 'bootstrap']:
            stats_file = monte_carlo_dir / f"mc_stats_{method}.json"
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    monte_carlo_stats[method] = json.load(f)
                found_methods.append(method)
        
        if found_methods:
            data['monte_carlo_stats'] = monte_carlo_stats
            if DEBUG_MODE:
                debug_info["monte_carlo_methods"] = found_methods
        else:
            # Try alternative paths
            alt_dir = project_root / "data" / "processed" / "monte_carlo"
            if alt_dir.exists():
                for method in ['parametric', 'historical', 'bootstrap']:
                    alt_file = alt_dir / f"mc_stats_{method}.json"
                    if alt_file.exists():
                        with open(alt_file, 'r') as f:
                            if 'monte_carlo_stats' not in data:
                                data['monte_carlo_stats'] = {}
                            data['monte_carlo_stats'][method] = json.load(f)
                        found_methods.append(method)
                
                if found_methods and DEBUG_MODE:
                    debug_info["monte_carlo_methods"] = f"alt path: {found_methods}"
    except Exception as e:
        if DEBUG_MODE:
            debug_info["monte_carlo_error"] = str(e)
    
    # Add debug information to data if in debug mode
    if DEBUG_MODE:
        st.sidebar.markdown("### Debug Information")
        st.sidebar.json(debug_info)
    
    return data

# Create dashboard
create_header()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview", 
    "🎯 Efficient Frontier", 
    "🧰 Portfolio Analysis", 
    "📈 Backtesting", 
    "⚠️ Risk Metrics", 
    "🔮 Monte Carlo",
    "ℹ️ About"
])

# Debug toggle
if st.sidebar.checkbox("Show Debug Info", value=DEBUG_MODE):
    DEBUG_MODE = True
else:
    DEBUG_MODE = False

# Load data
data = load_data()

# Check if we have essential data and offer to use sample data if not
if data is None or 'returns' not in data or 'prices' not in data:
    st.warning("Essential data is missing. The dashboard may not work properly.")
    
    # Offer to use sample data
    if st.button("Use Sample Data for Demonstration"):
        data = generate_sample_data()
        st.success("Using sample data for demonstration. Results are not based on your actual portfolio data.")
    else:
        st.info("Please run the optimization pipeline first or click 'Use Sample Data for Demonstration'.")
        st.stop()

# Calculate some metrics for display if returns data is available
if 'returns' in data:
    annual_returns = data['returns'].mean() * 252
    annual_volatility = data['returns'].std() * np.sqrt(252)
    sharpe_ratio = (annual_returns - 0.035) / annual_volatility  # Assuming 3.5% risk-free rate
    cumulative_returns = (1 + data['returns']).cumprod()
    total_returns = (cumulative_returns.iloc[-1] - 1) * 100  # in percentage

# Overview Page
if page == "📊 Overview":
    st.markdown('<p class="subtitle">Portfolio Optimization Overview</p>', unsafe_allow_html=True)
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">{}</div>
            <div class="metric-label">Assets Analyzed</div>
        </div>
        """.format(len(data['returns'].columns)), unsafe_allow_html=True)
    
    with col2:
        best_asset = total_returns.idxmax()
        best_return = total_returns.max()
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">{:.1f}%</div>
            <div class="metric-label">Best Asset Return ({})</div>
        </div>
        """.format(best_return, best_asset), unsafe_allow_html=True)
    
    with col3:
        best_sharpe_asset = sharpe_ratio.idxmax()
        best_sharpe = sharpe_ratio.max()
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">{:.2f}</div>
            <div class="metric-label">Highest Sharpe Ratio ({})</div>
        </div>
        """.format(best_sharpe, best_sharpe_asset), unsafe_allow_html=True)
    
    with col4:
        if 'backtest_results' in data:
            best_strategy = max(data['backtest_results'].items(), key=lambda x: x[1]['sharpe_ratio'])[0]
            best_strategy_sharpe = max(data['backtest_results'].items(), key=lambda x: x[1]['sharpe_ratio'])[1]['sharpe_ratio']
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{}</div>
                <div class="metric-label">Best Strategy</div>
            </div>
            """.format(best_strategy), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">N/A</div>
                <div class="metric-label">Best Strategy</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Portfolio Returns")
        
        if 'returns' in data:
            # Create interactive plotly chart
            fig = px.line(cumulative_returns, title='Cumulative Returns')
            fig.update_layout(
                height=500,
                xaxis_title="Date",
                yaxis_title="Growth of $1",
                legend_title="Asset",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Asset Correlation Heatmap")
        
        if 'returns' in data:
            # Create correlation matrix
            corr = data['returns'].corr()
            
            # Create interactive heatmap
            mask = np.triu(np.ones_like(corr, dtype=bool))
            df_mask = corr.mask(mask)
            
            fig = px.imshow(
                df_mask,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1
            )
            fig.update_layout(
                height=500,
                title="Asset Correlation Matrix",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Strategy comparison if available
    if 'backtest_results' in data:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Strategy Comparison")
        
        # Create a DataFrame for visualization
        results_df = pd.DataFrame()
        for strategy, metrics in data['backtest_results'].items():
            results_df[strategy] = pd.Series({
                'Annual Return (%)': metrics.get('annualized_return', 0) * 100,
                'Volatility (%)': metrics.get('annualized_volatility', 0) * 100,
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100
            })
        
        # Transpose for better visualization
        results_df = results_df.T
        
        # Create interactive bar chart for each metric
        tabs = st.tabs(["Annual Return", "Volatility", "Sharpe Ratio", "Max Drawdown"])
        
        with tabs[0]:
            fig = px.bar(
                results_df,
                y='Annual Return (%)',
                color='Annual Return (%)',
                color_continuous_scale='Blues',
                text_auto='.2f'
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            fig = px.bar(
                results_df,
                y='Volatility (%)',
                color='Volatility (%)',
                color_continuous_scale='Reds',
                text_auto='.2f'
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            fig = px.bar(
                results_df,
                y='Sharpe Ratio',
                color='Sharpe Ratio',
                color_continuous_scale='Greens',
                text_auto='.2f'
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:
            fig = px.bar(
                results_df,
                y='Max Drawdown (%)',
                color='Max Drawdown (%)',
                color_continuous_scale='Reds_r',
                text_auto='.2f'
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Efficient Frontier Page
elif page == "🎯 Efficient Frontier":
    st.markdown('<p class="subtitle">Efficient Frontier Analysis</p>', unsafe_allow_html=True)
    
    if 'efficient_frontier' in data:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Risk-Return Trade-off")
        
        # Create interactive efficient frontier plot
        ef_data = data['efficient_frontier']
        
        # Create scatter plot with Plotly
        fig = px.scatter(
            ef_data,
            x='Volatility',
            y='Return',
            color='Sharpe',
            color_continuous_scale='viridis',
            hover_data=['Sharpe'],
            title="Efficient Frontier"
        )
        
        # Add individual assets
        if 'returns' in data:
            for i, asset in enumerate(data['returns'].columns):
                asset_vol = annual_volatility[asset]
                asset_ret = annual_returns[asset]
                fig.add_trace(
                    go.Scatter(
                        x=[asset_vol],
                        y=[asset_ret],
                        mode='markers+text',
                        marker=dict(size=12, color='red'),
                        text=[asset],
                        textposition="top center",
                        name=asset
                    )
                )
        

        # Update layout
        fig.update_layout(
            height=600,
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            legend_title="Asset",
            template="plotly_white",
            hovermode="closest"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Minimum Volatility Portfolio")
            
            # Check if min vol portfolio exists in data
            min_vol_found = False
            
            if 'portfolio_weights' in data:
                df = data['portfolio_weights']
                if 'Portfolio' in df.columns and 'Min Volatility' in df['Portfolio'].values:
                    min_vol_weights = df[df['Portfolio'] == 'Min Volatility']
                    
                    # Create pie chart
                    fig = px.pie(
                        min_vol_weights,
                        values='Weight',
                        names='Asset',
                        title="Min Volatility Portfolio Allocation",
                        hole=0.4
                    )
                    fig.update_layout(height=400, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    min_vol_found = True
            
            if not min_vol_found:
                st.info("Minimum volatility portfolio data not available.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Maximum Sharpe Portfolio")
            
            # Check if max sharpe portfolio exists in data
            max_sharpe_found = False
            
            if 'portfolio_weights' in data:
                df = data['portfolio_weights']
                if 'Portfolio' in df.columns and 'Max Sharpe' in df['Portfolio'].values:
                    max_sharpe_weights = df[df['Portfolio'] == 'Max Sharpe']
                    
                    # Create pie chart
                    fig = px.pie(
                        max_sharpe_weights,
                        values='Weight',
                        names='Asset',
                        title="Max Sharpe Portfolio Allocation",
                        hole=0.4
                    )
                    fig.update_layout(height=400, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    max_sharpe_found = True
            
            if not max_sharpe_found:
                st.info("Maximum Sharpe portfolio data not available.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Efficient frontier data not found. Using sample data or run the optimization pipeline.")
        
        # Create a basic efficient frontier visualization
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Sample Efficient Frontier")
        
        # Generate sample efficient frontier data
        vol_range = np.linspace(0.1, 0.3, 100)
        ret_range = 0.05 + 0.5 * vol_range + np.random.normal(0, 0.01, 100)
        sharpe = (ret_range - 0.035) / vol_range
        
        # Create DataFrame
        ef_data = pd.DataFrame({
            'Volatility': vol_range,
            'Return': ret_range,
            'Sharpe': sharpe
        })
        
        # Create scatter plot
        fig = px.scatter(
            ef_data,
            x='Volatility',
            y='Return',
            color='Sharpe',
            color_continuous_scale='viridis',
            title="Sample Efficient Frontier (Demo Data)"
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("This is a demonstration visualization with sample data.")
        st.markdown('</div>', unsafe_allow_html=True)

# Portfolio Analysis Page
elif page == "🧰 Portfolio Analysis":
    st.markdown('<p class="subtitle">Portfolio Performance Analysis</p>', unsafe_allow_html=True)
    
    if 'portfolio_weights' in data and 'returns' in data:
        # Create portfolio selection
        portfolios = data['portfolio_weights']['Portfolio'].unique()
        portfolio = st.selectbox("Select Portfolio", portfolios, index=0)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"{portfolio} Portfolio Composition")
        
        # Filter weights for selected portfolio
        port_weights = data['portfolio_weights'][data['portfolio_weights']['Portfolio'] == portfolio]
        
        # Create pie chart
        fig = px.pie(
            port_weights,
            values='Weight',
            names='Asset',
            title=f"{portfolio} Portfolio Allocation",
            hole=0.4
        )
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create dictionary of weights
        weights_dict = dict(zip(port_weights['Asset'], port_weights['Weight']))
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0, index=data['returns'].index)
        for asset, weight in weights_dict.items():
            if asset in data['returns'].columns:
                portfolio_returns += data['returns'][asset] * weight
        
        # Calculate cumulative returns
        cum_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate drawdowns
        peak = cum_returns.cummax()
        drawdown = (cum_returns / peak) - 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Cumulative Returns")
            
            # Create line chart
            fig = px.line(
                cum_returns,
                y=cum_returns,
                title=f"{portfolio} Portfolio - Cumulative Returns"
            )
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Growth of $1",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Drawdowns")
            
            # Create area chart for drawdowns
            fig = px.area(
                drawdown,
                y=drawdown,
                title=f"{portfolio} Portfolio - Drawdowns",
                color_discrete_sequence=["rgba(255, 0, 0, 0.5)"]
            )
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Drawdown",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate performance metrics
        total_return = cum_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annualized_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.035) / annualized_vol  # Assuming 3.5% risk-free rate
        
        # Calculate Sortino ratio (handle case with no negative returns)
        negative_returns = portfolio_returns[portfolio_returns < 0]
        if len(negative_returns) > 0:
            sortino_ratio = (annualized_return - 0.035) / (negative_returns.std() * np.sqrt(252))
        else:
            sortino_ratio = float('inf')  # No downside deviation
        
        max_dd = drawdown.min()
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Performance Metrics")
        
        # Create metrics in columns
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Total Return", f"{total_return:.2%}")
            st.metric("Annualized Return", f"{annualized_return:.2%}")
        
        with metrics_col2:
            st.metric("Annualized Volatility", f"{annualized_vol:.2%}")
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        with metrics_col3:
            st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
            st.metric("Maximum Drawdown", f"{max_dd:.2%}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Monthly returns heatmap
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Monthly Returns Heatmap")
        
        try:
            # Resample to monthly returns
            monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            # Create DataFrame with year and month
            monthly_df = pd.DataFrame({
                'Return': monthly_returns,
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month
            })
            
            # Pivot table for heatmap
            pivot_df = monthly_df.pivot(index='Year', columns='Month', values='Return')
            
            # Replace month numbers with names
            month_names = {
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            }
            pivot_df.columns = [month_names[m] for m in pivot_df.columns]
            
            # Create heatmap using plotly
            fig = px.imshow(
                pivot_df,
                color_continuous_scale='RdYlGn',
                text_auto='.1%',
                aspect="auto"
            )
            fig.update_layout(
                height=400,
                template="plotly_white",
                title="Monthly Returns (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create monthly returns heatmap: {str(e)}")
            st.info("This may be due to insufficient data or date range issues.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Portfolio weights or returns data not found. Using sample data or run the optimization pipeline.")
        
        # Create sample portfolio analysis
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Sample Portfolio Analysis (Demo Data)")
        
        # Generate sample portfolio returns
        dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='B')
        portfolio_returns = pd.Series(np.random.normal(0.0006, 0.01, len(dates)), index=dates)
        cum_returns = (1 + portfolio_returns).cumprod()
        
        # Create chart
        fig = px.line(
            cum_returns,
            title="Sample Portfolio Cumulative Returns"
        )
        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("This is a demonstration visualization with sample data.")
        st.markdown('</div>', unsafe_allow_html=True)

# Backtesting Page
elif page == "📈 Backtesting":
    st.markdown('<p class="subtitle">Strategy Backtesting Results</p>', unsafe_allow_html=True)
    
    if 'backtest_values' in data and 'backtest_results' in data:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Strategy Comparison - Performance")
        
        # Create interactive performance chart
        fig = px.line(
            data['backtest_values'],
            title="Portfolio Value Over Time"
        )
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            legend_title="Strategy",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Create normalized performance
        normalized_values = data['backtest_values'].div(data['backtest_values'].iloc[0])
        
        fig = px.line(
            normalized_values,
            title="Growth of $1 Investment"
        )
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Growth Multiple",
            legend_title="Strategy",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Strategy selection for detailed view
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Strategy Details")
        
        strategy = st.selectbox("Select Strategy", list(data['backtest_results'].keys()))
        
        # Display strategy metrics
        metrics = data['backtest_results'][strategy]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
            st.metric("Annualized Return", f"{metrics.get('annualized_return', 0):.2%}")
        
        with col2:
            st.metric("Annualized Volatility", f"{metrics.get('annualized_volatility', 0):.2%}")
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        
        with col3:
            st.metric("Maximum Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
            st.metric("Final Value", f"${metrics.get('final_value', 0):,.2f}")
        
        # Configuration details
        st.markdown("### Strategy Configuration")
        st.markdown(f"**Rebalancing Frequency:** {metrics.get('rebalance_frequency', 'N/A')}")
        st.markdown(f"**Initial Investment:** ${metrics.get('initial_investment', 0):,.2f}")
        
        if 'weights' in metrics and metrics['weights']:
            st.markdown("### Top Holdings")
            weights = metrics['weights']
            sorted_weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}
            
            # Create bar chart for top holdings
            top_n = min(10, len(sorted_weights))
            top_holdings = pd.DataFrame({
                'Asset': list(sorted_weights.keys())[:top_n],
                'Weight': list(sorted_weights.values())[:top_n]
            })
            
            fig = px.bar(
                top_holdings,
                x='Asset',
                y='Weight',
                color='Weight',
                color_continuous_scale='Blues',
                text_auto='.1%'
            )
            fig.update_layout(
                height=400,
                xaxis_title="Asset",
                yaxis_title="Weight",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate drawdowns for comparison
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Drawdown Comparison")
        
        drawdowns = {}
        for strat, values in data['backtest_values'].items():
            peak = values.cummax()
            drawdowns[strat] = (values / peak) - 1
        
        # Create drawdown DataFrame
        drawdown_df = pd.DataFrame(drawdowns)
        
        # Create drawdown chart
        fig = px.line(
            drawdown_df,
            title="Portfolio Drawdowns"
        )
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Drawdown",
            legend_title="Strategy",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Backtest results not found. Using sample data or run the optimization pipeline.")
        
        # Create sample backtest visualization
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Sample Backtest Results (Demo Data)")
        
        # Generate sample backtest data
        dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='B')
        strategies = ['Equal Weight', 'Max Sharpe', 'Min Volatility', 'Risk Parity']
        
        backtest_values = pd.DataFrame(index=dates)
        for i, strategy in enumerate(strategies):
            # Different performance characteristics
            mu = 0.0005 + i * 0.0001
            sigma = 0.010 - i * 0.001
            
            values = [100000]
            for j in range(1, len(dates)):
                values.append(values[-1] * (1 + np.random.normal(mu, sigma)))
            
            backtest_values[strategy] = values
        
        # Create performance chart
        fig = px.line(
            backtest_values,
            title="Sample Strategy Comparison (Demo Data)"
        )
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            legend_title="Strategy",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("This is a demonstration visualization with sample data.")
        st.markdown('</div>', unsafe_allow_html=True)

# Risk Metrics Page
elif page == "⚠️ Risk Metrics":
    st.markdown('<p class="subtitle">Risk Analysis</p>', unsafe_allow_html=True)
    
    if 'returns' in data:
        # Asset selection for risk analysis
        default_assets = list(data['returns'].columns[:min(4, len(data['returns'].columns))])
        assets = st.multiselect(
            "Select Assets for Risk Analysis",
            options=data['returns'].columns.tolist(),
            default=default_assets
        )
        
        if assets:
            # Calculate basic risk metrics directly
            risk_metrics_dict = {}
            for asset in assets:
                asset_returns = data['returns'][asset].dropna()
                
                # Basic statistics
                mean_return = asset_returns.mean() * 252  # Annualized
                volatility = asset_returns.std() * np.sqrt(252)  # Annualized
                sharpe = (mean_return - 0.035) / volatility  # Assuming 3.5% risk-free rate
                
                # Downside statistics
                downside_returns = asset_returns[asset_returns < 0]
                downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                sortino = (mean_return - 0.035) / downside_vol if downside_vol > 0 else float('inf')
                
                # VaR and CVaR
                var_95 = np.percentile(asset_returns, 5)
                cvar_95 = asset_returns[asset_returns <= var_95].mean()
                
                # Drawdowns
                asset_prices = data['prices'][asset]
                peak = asset_prices.cummax()
                drawdowns = (asset_prices / peak) - 1
                max_drawdown = drawdowns.min()
                
                # Store metrics
                risk_metrics_dict[asset] = {
                    'annual_volatility': volatility,
                    'annual_return': mean_return,
                    'sharpe_ratio': sharpe,
                    'sortino_ratio': sortino,
                    'downside_volatility': downside_vol,
                    'var_95_daily': -var_95,
                    'cvar_95_daily': -cvar_95,
                    'maximum_drawdown': max_drawdown
                }
            
            # Create DataFrames
            vol_df = pd.DataFrame({asset: metrics['annual_volatility'] for asset, metrics in risk_metrics_dict.items()}, index=['annual_volatility']).T
            return_df = pd.DataFrame({asset: metrics['annual_return'] for asset, metrics in risk_metrics_dict.items()}, index=['annual_return']).T
            sharpe_df = pd.DataFrame({asset: metrics['sharpe_ratio'] for asset, metrics in risk_metrics_dict.items()}, index=['sharpe_ratio']).T
            dd_df = pd.DataFrame({asset: metrics['maximum_drawdown'] for asset, metrics in risk_metrics_dict.items()}, index=['maximum_drawdown']).T
            var_df = pd.DataFrame({
                asset: [metrics['var_95_daily'], metrics['cvar_95_daily']] 
                for asset, metrics in risk_metrics_dict.items()
            }, index=['var_95_daily', 'cvar_95_daily']).T
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Risk Metrics Dashboard")
            
            # Create tabs for different risk metrics
            tabs = st.tabs(["Risk-Return", "Volatility", "Value at Risk", "Drawdowns"])
            
            with tabs[0]:
                # Create DataFrame for scatter plot
                risk_return_df = pd.DataFrame({
                    'Asset': [asset for asset in risk_metrics_dict.keys()],
                    'Return': [metrics['annual_return'] for metrics in risk_metrics_dict.values()],
                    'Volatility': [metrics['annual_volatility'] for metrics in risk_metrics_dict.values()],
                    'Sharpe': [metrics['sharpe_ratio'] for metrics in risk_metrics_dict.values()]
                })
                
                # Create scatter plot
                fig = px.scatter(
                    risk_return_df,
                    x='Volatility',
                    y='Return',
                    color='Sharpe',
                    size=[40] * len(risk_return_df),
                    text='Asset',
                    color_continuous_scale='viridis',
                    hover_data=['Sharpe']
                )
                
                fig.update_traces(textposition='top center')
                fig.update_layout(
                    height=600,
                    xaxis_title="Annualized Volatility",
                    yaxis_title="Annualized Return",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:
                # Volatility metrics
                st.subheader("Volatility Metrics")
                st.dataframe(vol_df.style.format('{:.4f}'), use_container_width=True)
                
                # Create a comparison of total volatility and downside volatility
                vol_compare = pd.DataFrame({
                    'Asset': list(risk_metrics_dict.keys()) * 2,
                    'Metric': ['Total Volatility'] * len(risk_metrics_dict) + ['Downside Volatility'] * len(risk_metrics_dict),
                    'Value': [metrics['annual_volatility'] for metrics in risk_metrics_dict.values()] + 
                            [metrics['downside_volatility'] for metrics in risk_metrics_dict.values()]
                })
                
                # Create grouped bar chart
                fig = px.bar(
                    vol_compare,
                    x='Asset',
                    y='Value',
                    color='Metric',
                    barmode='group',
                    text_auto='.3f'
                )
                fig.update_layout(
                    height=500,
                    xaxis_title="Asset",
                    yaxis_title="Volatility",
                    legend_title="Metric",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[2]:
                # VaR and CVaR metrics
                st.subheader("Value at Risk (VaR) and Conditional VaR (CVaR)")
                st.dataframe(var_df.style.format('{:.4f}'), use_container_width=True)
                
                # Create bar chart for VaR and CVaR
                var_long = pd.DataFrame({
                    'Asset': list(risk_metrics_dict.keys()) * 2,
                    'Metric': ['Value at Risk (95%)'] * len(risk_metrics_dict) + ['Conditional VaR (95%)'] * len(risk_metrics_dict),
                    'Value': [metrics['var_95_daily'] for metrics in risk_metrics_dict.values()] + 
                            [metrics['cvar_95_daily'] for metrics in risk_metrics_dict.values()]
                })
                
                fig = px.bar(
                    var_long,
                    x='Asset',
                    y='Value',
                    color='Metric',
                    barmode='group',
                    text_auto='.3f'
                )
                fig.update_layout(
                    height=500,
                    xaxis_title="Asset",
                    yaxis_title="Daily Loss (%)",
                    legend_title="Risk Measure",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution plots
                st.subheader("Return Distributions")
                
                # Create distribution plots for each asset
                fig = make_subplots(rows=1, cols=len(assets))
                
                for i, asset in enumerate(assets):
                    asset_returns = data['returns'][asset].dropna()
                    
                    # Create histogram
                    fig.add_trace(
                        go.Histogram(
                            x=asset_returns,
                            name=asset,
                            histnorm='probability density',
                            marker_color='rgba(0, 114, 178, 0.7)'
                        ),
                        row=1,
                        col=i+1
                    )
                    
                    # Add normal distribution curve
                    x = np.linspace(asset_returns.min(), asset_returns.max(), 100)
                    y = stats.norm.pdf(x, asset_returns.mean(), asset_returns.std())
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode='lines',
                            name='Normal',
                            line=dict(color='red', dash='dash'),
                            showlegend=False
                        ),
                        row=1,
                        col=i+1
                    )
                
                fig.update_layout(
                    height=400,
                    title_text="Return Distributions vs. Normal",
                    template="plotly_white"
                )
                
                for i in range(len(assets)):
                    fig.update_xaxes(title_text=assets[i], row=1, col=i+1)
                    if i == 0:
                        fig.update_yaxes(title_text="Density", row=1, col=i+1)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[3]:
                # Drawdown metrics
                st.subheader("Maximum Drawdown")
                st.dataframe(dd_df.style.format('{:.4%}'), use_container_width=True)
                
                # Create bar chart for maximum drawdowns
                fig = px.bar(
                    dd_df.sort_values('maximum_drawdown'),
                    y='maximum_drawdown',
                    color='maximum_drawdown',
                    color_continuous_scale='Reds_r',
                    text_auto='.2%'
                )
                fig.update_layout(
                    height=500,
                    xaxis_title="Asset",
                    yaxis_title="Maximum Drawdown",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show historical drawdowns
                drawdowns = {}
                for asset in assets:
                    try:
                        price_series = data['prices'][asset]
                        peak = price_series.cummax()
                        drawdowns[asset] = (price_series / peak) - 1
                    except Exception as e:
                        st.warning(f"Could not calculate drawdowns for {asset}: {str(e)}")
                
                if drawdowns:
                    drawdown_df = pd.DataFrame(drawdowns)
                    
                    fig = px.line(
                        drawdown_df,
                        title="Historical Drawdowns"
                    )
                    fig.update_layout(
                        height=500,
                        xaxis_title="Date",
                        yaxis_title="Drawdown",
                        legend_title="Asset",
                        template="plotly_white",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Please select at least one asset for risk analysis.")
    else:
        st.warning("Returns data not found. Using sample data or run the optimization pipeline.")
        
        # Create sample risk metrics visualization
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Sample Risk Metrics (Demo Data)")
        
        # Generate sample data
        assets = ['SPY', 'QQQ', 'AAPL', 'GLD']
        vol = [0.15, 0.20, 0.25, 0.12]  # Sample volatilities
        ret = [0.10, 0.12, 0.14, 0.06]  # Sample returns
        sharpe = [ret[i]/vol[i] for i in range(len(assets))]  # Sample Sharpe ratios
        
        # Create scatter plot
        risk_return_df = pd.DataFrame({
            'Asset': assets,
            'Return': ret,
            'Volatility': vol,
            'Sharpe': sharpe
        })
        
        fig = px.scatter(
            risk_return_df,
            x='Volatility',
            y='Return',
            color='Sharpe',
            size=[40] * len(risk_return_df),
            text='Asset',
            color_continuous_scale='viridis'
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(
            height=500,
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("This is a demonstration visualization with sample data.")
        st.markdown('</div>', unsafe_allow_html=True)

# Monte Carlo Page
elif page == "🔮 Monte Carlo":
    st.markdown('<p class="subtitle">Monte Carlo Simulation</p>', unsafe_allow_html=True)
    
    if 'monte_carlo_stats' in data:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Simulation Results")
        
        # Convert monte carlo stats to DataFrame
        stats_list = []
        for method, stats in data['monte_carlo_stats'].items():
            stats_df = pd.DataFrame.from_dict(stats, orient='index')
            stats_df['method'] = method
            stats_list.append(stats_df)
        
        mc_stats = pd.concat(stats_list, axis=0)
        mc_stats = mc_stats.reset_index().pivot(index='index', columns='method', values=0)
        
        # Filter for key statistics
        key_stats = ['mean', 'median', 'min', 'max', 'std', 'percentile_5', 'percentile_95']
        if all(stat in mc_stats.index for stat in key_stats):
            display_stats = mc_stats.loc[key_stats]
            
            # Display as a table
            st.dataframe(display_stats.style.format('${:,.2f}'), use_container_width=True)
            
            # Create interactive simulation visualization
            method = st.selectbox("Select Simulation Method", mc_stats.columns)
            
            # Plot key metrics
            fig = go.Figure()
            
            # Add range (5th to 95th percentile)
            fig.add_trace(go.Scatter(
                x=['Simulation Result'],
                y=[mc_stats.loc['mean', method]],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[mc_stats.loc['percentile_95', method] - mc_stats.loc['mean', method]],
                    arrayminus=[mc_stats.loc['mean', method] - mc_stats.loc['percentile_5', method]],
                    color="rgba(0, 114, 178, 0.3)",
                    width=30
                ),
                mode='markers',
                marker=dict(
                    color='rgba(0, 114, 178, 0.8)',
                    size=20
                ),
                name=f"{method.capitalize()} Simulation"
            ))
            
            # Add initial investment line
            if 'initial_investment' in mc_stats.index:
                initial_investment = mc_stats.loc['initial_investment', method]
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=0.5,
                    y0=initial_investment,
                    y1=initial_investment,
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash",
                    )
                )
                
                fig.add_annotation(
                    x=0,
                    y=initial_investment,
                    text=f"Initial Investment: ${initial_investment:,.0f}",
                    showarrow=False,
                    yshift=10,
                    font=dict(
                        color="red"
                    )
                )
            
            fig.update_layout(
                height=600,
                title=f"{method.capitalize()} Simulation Results Range",
                yaxis_title="Portfolio Value ($)",
                xaxis=dict(
                    showticklabels=False
                ),
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Monte Carlo statistics do not contain expected keys. Format may be incorrect.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Probability analysis
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Probability Analysis")
        
        # Extract probability metrics
        prob_metrics = [col for col in mc_stats.index if col.startswith('prob_return')]
        if prob_metrics:
            prob_data = []
            for metric in prob_metrics:
                # Extract target from metric name (prob_return_X.Y)
                target = float(metric.split('_')[-1])
                
                for sim_method in mc_stats.columns:
                    if metric in mc_stats.index:
                        prob = mc_stats.loc[metric, sim_method]
                        initial_investment = mc_stats.loc['initial_investment', sim_method] if 'initial_investment' in mc_stats.index else 100000
                        
                        prob_data.append({
                            'Target Return': f"{target*100:.0f}%",
                            'Target Value': initial_investment * (1 + target),
                            'Probability': prob,
                            'Method': sim_method.capitalize()
                        })
            
            if prob_data:
                prob_df = pd.DataFrame(prob_data)
                
                # Create grouped bar chart
                fig = px.bar(
                    prob_df,
                    x='Target Return',
                    y='Probability',
                    color='Method',
                    barmode='group',
                    text_auto='.1%',
                    hover_data=['Target Value']
                )
                fig.update_layout(
                    height=500,
                    xaxis_title="Target Return",
                    yaxis_title="Probability of Achieving",
                    legend_title="Simulation Method",
                    template="plotly_white"
                )
                
                # Format hover template
                fig.update_traces(
                    hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<br>Target Value: $%{customdata[0]:,.2f}<extra></extra>'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No probability data could be extracted from the simulation results.")
        else:
            st.info("Probability metrics not available in the simulation results.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Stress test results if available
        try:
            stress_path = monte_carlo_dir / "stress_test_summary.csv"
            if stress_path.exists():
                stress_results = pd.read_csv(stress_path)
                
                if not stress_results.empty:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Stress Test Results")
                    
                    # Create a bar chart for total returns in stress scenarios
                    fig = px.bar(
                        stress_results,
                        x='Scenario',
                        y='Total Return',
                        color='Total Return',
                        color_continuous_scale='RdYlGn',
                        text_auto='.1%'
                    )
                    fig.update_layout(
                        height=500,
                        xaxis_title="Scenario",
                        yaxis_title="Total Return",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create a table with additional metrics
                    display_cols = ['Scenario', 'Initial Value', 'Final Value', 'Total Return', 'Max Drawdown']
                    if all(col in stress_results.columns for col in display_cols):
                        # Format the table
                        st.dataframe(stress_results[display_cols].style.format({
                            'Initial Value': '${:,.2f}',
                            'Final Value': '${:,.2f}',
                            'Total Return': '{:.2%}',
                            'Max Drawdown': '{:.2%}'
                        }), use_container_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        except:
            pass
    else:
        st.warning("Monte Carlo simulation results not found. Using sample data or run the optimization pipeline.")
        
        # Create sample Monte Carlo visualization
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Sample Monte Carlo Simulation (Demo Data)")
        
        # Generate sample Monte Carlo results
        initial_investment = 100000
        final_values = np.random.normal(150000, 40000, 1000)
        
        # Plot histogram
        fig = px.histogram(
            final_values,
            nbins=50,
            title="Sample Distribution of Final Portfolio Values",
            labels={'value': 'Portfolio Value ($)', 'count': 'Frequency'},
            opacity=0.7
        )
        
        # Add vertical lines for statistics
        fig.add_vline(x=initial_investment, line_dash="dash", line_color="red", 
                    annotation_text="Initial Investment", annotation_position="top right")
        fig.add_vline(x=np.mean(final_values), line_color="green", 
                    annotation_text="Mean", annotation_position="top right")
        fig.add_vline(x=np.percentile(final_values, 5), line_color="orange", 
                    annotation_text="5th Percentile", annotation_position="top left")
        fig.add_vline(x=np.percentile(final_values, 95), line_color="orange", 
                    annotation_text="95th Percentile", annotation_position="top right")
        
        fig.update_layout(
            height=500,
            template="plotly_white",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("This is a demonstration visualization with sample data.")
        st.markdown('</div>', unsafe_allow_html=True)

# About Page
elif page == "ℹ️ About":
    st.markdown('<p class="subtitle">About This Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Modern Portfolio Optimization Toolkit")
    
    st.markdown("""
    This interactive dashboard visualizes the results of the portfolio optimization pipeline. It provides insights into:
    
    * **Portfolio Performance**: Track historical performance and risk metrics
    * **Efficient Frontier**: Visualize the risk-return tradeoff and optimal portfolios
    * **Risk Analysis**: Examine volatility, Value at Risk, and drawdowns
    * **Monte Carlo Simulations**: Project future performance probabilities
    * **Strategy Backtesting**: Compare different investment strategies
    
    The underlying toolkit uses Modern Portfolio Theory and advanced techniques including Factor-Based Optimization and the Black-Litterman Model to create optimized investment portfolios.
    
    ### Technologies Used
    
    * **Data Collection**: Yahoo Finance, Alpha Vantage, FRED API
    * **Analysis**: NumPy, Pandas, SciPy
    * **Optimization**: PyPortfolioOpt
    * **Visualization**: Plotly, Matplotlib
    * **Dashboard**: Streamlit
    
    ### Getting Started
    
    1. Run the portfolio optimization pipeline using `python main.py`
    2. Launch this dashboard with `streamlit run streamlit_dashboard.py`
    3. Explore the different pages using the navigation menu
    
    For more information, see the project README and documentation.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 20px; padding: 10px; border-top: 1px solid #ddd;">
    <p style="color: #666; font-size: 12px;">
        Modern Portfolio Optimization Toolkit © 2025 | Created with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)