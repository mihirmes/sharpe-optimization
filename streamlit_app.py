import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import streamlit as st
import plotly.graph_objects as go

# Set Streamlit page layout to wide
st.set_page_config(layout="wide")

# Streamlit input section
st.title('Sharpe Ratio Optimization Dashboard')
st.sidebar.header('Portfolio Optimization Inputs')
st.sidebar.number_input('Number of Portfolio Iterations', min_value=10000, max_value=1000000, value=10000, step=10000, help='A higher number will yield more consistent results, but will take longer to compute')

# Predefined ticker options
default_ticker_options = {
    'Vanguard US Equities Growth ETFs': ['IVOG', 'MGK', 'VBK', 'VIOG', 'VONG', 'VOOG', 'VOT', 'VTWG', 'VUG', '^TNX'],
    'Vanguard US Equities Value ETFs': ['IVOV', 'MGV', 'VBR', 'VIOV', 'VOE', 'VONV', 'VOOV', 'VTV', 'VTWV', '^TNX'],
    'Vanguard Broad Market Global Bonds ETFs': ['BIV', 'BLV', 'BND', 'BNDW', 'BNDX', 'BSV', '^TNX'],
    'Tech Giants': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'LLY', '^TNX'],
    'Custom': 'Custom'
}

# Radio button for selecting default tickers or custom input
ticker_option = st.sidebar.radio('Select a Ticker Set', list(default_ticker_options.keys()))

# Input for custom tickers if 'Custom' is selected
if ticker_option == 'Custom':
    custom_tickers = st.sidebar.text_input('Enter tickers separated by commas (e.g., AAPL,MSFT,GOOGL)').upper()
    tickers = [ticker.strip() for ticker in custom_tickers.split(',')]
else:
    tickers = default_ticker_options[ticker_option]

# Default period is set to 5 years
period = '5y'

def get_historical_data(tickers, period='5y'):
    """
    Fetch historical market data for a list of tickers.
    """
    data = yf.download(tickers, period=period, interval='1d')['Adj Close']
    return data

def calculate_annual_returns_and_cov_matrix(data):
    """
    Calculate annualized returns and covariance matrix of returns.
    """
    daily_returns = data.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    return annual_returns, cov_matrix

def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculate the expected return, volatility, and Sharpe ratio of a portfolio.
    """
    returns = np.sum(mean_returns * weights)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = returns / volatility
    return returns, volatility, sharpe_ratio

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    """
    Calculate the negative Sharpe ratio (to be minimized).
    """
    returns, volatility, _ = portfolio_performance(weights, mean_returns, cov_matrix)
    return -((returns - risk_free_rate) / volatility)

def optimize_portfolio(mean_returns, cov_matrix, num_assets, risk_free_rate=0):
    """
    Optimize the portfolio to maximize the Sharpe ratio.
    """
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate=0.03, num_portfolios=10000):
    """
    Plot the efficient frontier with random portfolios using Plotly.
    """
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_stddev, portfolio_sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_return
        results[1,i] = portfolio_stddev
        results[2,i] = portfolio_sharpe

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[1,max_sharpe_idx], results[0,max_sharpe_idx]
    max_sharpe_allocation = weights_record[max_sharpe_idx]

    max_volatility = max(results[1,:])
    min_volatility = min(results[1,:])
    max_return = max(results[0,:])
    min_return = min(results[0,:])

    # Plot Efficient Frontier and Portfolios
    fig = go.Figure()

    # Random portfolios
    fig.add_trace(go.Scatter(
        x=results[1,:],
        y=results[0,:],
        mode='markers',
        marker=dict(
            color=results[2,:],
            colorscale='YlGnBu',
            showscale=True,
            size=5
        ),
        name='Random Portfolios'
    ))

    # Capital Market Line (CML)
    cml_x = [0, sdp]
    cml_y = [risk_free_rate, rp]

    fig.add_trace(go.Scatter(
        x=cml_x,
        y=cml_y,
        mode='lines+text',
        line=dict(color='red', width=2, dash='dash'),
        name='Capital Market Line',
        hoverinfo='none'
    ))

    # Highlight portfolio with max Sharpe ratio (Market Portfolio)
    fig.add_trace(go.Scatter(
        x=[sdp],
        y=[rp],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Market Portfolio'
    ))

    # Plot the risk-free rate
    fig.add_trace(go.Scatter(
        x=[0],
        y=[risk_free_rate],
        mode='markers',
        marker=dict(color='red', size=10),
        name=f'Risk-Free Rate: {risk_free_rate*100:.2f}%'
    ))

    # Set plot limits and legend
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility',
        yaxis_title='Return',
        showlegend=True,
        legend=dict(
            x=0,
            y=1,
            xanchor='left',
            yanchor='top',
            bordercolor="Black",
            borderwidth=1
        ),
        height=600,
        xaxis=dict(range=[min(0, min_volatility), 1.1*max_volatility]),
        yaxis=dict(range=[min(0, 1.1*min_return), max(1.1*risk_free_rate,1.1*max_return)])
    )

    st.plotly_chart(fig)

    # Display portfolio composition with a pie chart
    allocation = pd.Series(max_sharpe_allocation, index=mean_returns.index).sort_values(ascending=False)

    fig_allocation = go.Figure(go.Pie(
        labels=allocation.index,
        values=allocation.values,
        hoverinfo='label+percent',
        textinfo='value',
    ))

    fig_allocation.update_layout(
        title='Portfolio Composition',
    )
    st.divider()
    col00, col01, col02 = st.columns([15,5,15])
    col00.plotly_chart(fig_allocation)

    col02.subheader('Portfolio Details')
    col02.divider()
    col02.write(f'Max Sharpe Ratio Portfolio Allocation\n')
    col02.write(f'Annualized Return: {rp:.2%}')
    col02.write(f'Annualized Volatility: {sdp:.2%}')
    col02.write(f'Sharpe Ratio: {results[2,max_sharpe_idx]:.2f}\n')

    st.markdown("Dashboard made by Mihir Meswania")


# Fetch historical data
data = get_historical_data(tickers, period=period)

# Check if '^TNX' (10-year treasury yield) is in the data for risk-free rate
if '^TNX' in tickers:
    risk_free_rate = data['^TNX'].iloc[-1] / 100
    data = data.drop(columns=['^TNX'])
else:
    try:    
        risk_free_rate = yf.download("^TNX", period='1d')  # Updated default risk-free rate
    except Exception:
        risk_free_rate = 0.03  # Updated default risk-free rate

# Clean the data by removing tickers with missing values
data.dropna(axis=0, inplace=True)

# Calculate annual returns and covariance matrix
mean_returns, cov_matrix = calculate_annual_returns_and_cov_matrix(data)

# Optimize portfolio
num_assets = len(data.columns)
opt_result = optimize_portfolio(mean_returns, cov_matrix, num_assets, risk_free_rate)

# Plot efficient frontier and portfolio composition
plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate=risk_free_rate)
st.info("**Disclaimer:** The information generated is not to be taken as investing advice. The purpose of this dashboard is completely educational and to visualize portfolio sharpe ratio optimization")
