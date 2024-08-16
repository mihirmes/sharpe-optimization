import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import streamlit as st
import plotly.graph_objects as go

# Set Streamlit page layout to wide
st.set_page_config(page_title='Sharpe Optimization', page_icon=':bar_chart:', layout="wide")

# Streamlit input section
st.title('Sharpe Ratio Optimization Dashboard')
st.sidebar.header('Portfolio Optimization Inputs')
num_portfolios = st.sidebar.number_input('Number of Portfolio Iterations', min_value=10000, max_value=1000000, value=10000, step=10000, help='Note: A higher number will yield more consistent results, but will take longer to compute')

# Predefined ticker options
default_ticker_options = {
    'Vanguard US Equities Growth ETFs': ['IVOG', 'MGK', 'VBK', 'VIOG', 'VONG', 'VOOG', 'VOT', 'VTWG', 'VUG', '^TNX'],
    'Vanguard US Equities Value ETFs': ['IVOV', 'MGV', 'VBR', 'VIOV', 'VOE', 'VONV', 'VOOV', 'VTV', 'VTWV', '^TNX'],
    'Vanguard Broad Market Global Bonds ETFs': ['BIV', 'BLV', 'BND', 'BNDW', 'BNDX', 'BSV', '^TNX'],
    'Magnificent 7': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', '^TNX'],
}

# Radio button for selecting default tickers or custom input
ticker_option = st.sidebar.radio('Select a Ticker Set', list(default_ticker_options.keys()))
tickers = default_ticker_options[ticker_option]

# Default period is set to 5 years
period = '5y'

def get_historical_data(tickers, period='5y'):
    """
    Fetch historical market data for a list of tickers.
    """
    data = yf.download(tickers, period=period, interval='1d')['Adj Close'][1::]
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

    # Highlight portfolio with max Sharpe ratio (Market Portfolio)
    fig.add_trace(go.Scatter(
        x=[sdp],
        y=[rp],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Max Sharpe Ratio Portfolio'
    ))

    # Set plot limits and legend
    fig.update_layout(
        title='Portfolio Visualization Along Efficient Frontier',
        xaxis_title='Volatility',
        yaxis_title='Return',
        showlegend=True,
        legend=dict(
            x=0.8,
            y=1,
            xanchor='left',
            yanchor='top',
            bordercolor="white",
            borderwidth=2
        ),
        height=600
        # xaxis=dict(range=[0.9*min_volatility, 1.1*max_volatility]),
        # yaxis=dict(range=[0.9*min_return, 1.1*max_return])
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
    col02.write(f'Annualized Return: {rp:.2%}')
    col02.write(f'Annualized Volatility: {sdp:.2%}')
    col02.write(f'Sharpe Ratio: {results[2,max_sharpe_idx]:.2f}\n')


# Fetch historical data
try:
    data = get_historical_data(tickers, period=period)
    # Check if '^TNX' (10-year treasury yield) is in the data for risk-free rate
    if '^TNX' in tickers:
        risk_free_rate = data['^TNX'].iloc[-1] / 100
        data = data.drop(columns=['^TNX'])
    else:
        risk_free_rate = 0.03  # Updated default risk-free rate

    # Clean the data by removing missing tickers
    data.dropna(axis=1, inplace=True)

    # Clean the data by removing missing values
    data.dropna(axis=0, inplace=True)

    # Calculate annual returns and covariance matrix
    mean_returns, cov_matrix = calculate_annual_returns_and_cov_matrix(data)

    # Optimize portfolio
    num_assets = len(data.columns)
    opt_result = optimize_portfolio(mean_returns, cov_matrix, num_assets, risk_free_rate)

    # Plot efficient frontier and portfolio composition
    plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate=risk_free_rate, num_portfolios=num_portfolios)
    st.info("**Disclaimer:** The information generated is not to be taken as investing advice. The purpose of this dashboard is completely educational and to visualize portfolio sharpe ratio optimization")
except Exception:
    st.error('Error fetching historical stock data')

with st.expander("Understand the Sharpe Ratio"):
    st.markdown(r"""
    ## Understanding the Sharpe Ratio
    
    ### What is the Sharpe Ratio?
    The Sharpe Ratio is a crucial financial metric used to evaluate the risk-adjusted return of an investment portfolio. 
                Developed by Nobel laureate William F. Sharpe, the ratio helps investors understand how much excess return they are 
                receiving for the extra volatility they take on when investing in a riskier asset.
    ### Formula for the Sharpe Ratio
    The Sharpe Ratio is calculated using the following formula:
    $$Sharpe Ratio = \frac{R_p - R_f}{\sigma_p}$$
    Where:
    - $$R_p$$ is the expected portfolio return.
    - $$R_f$$ is the risk-free rate of return.
    - $$\sigma_p$$ is the portfolio's standard deviation (a measure of volatility).

    ### Interpretation
    A higher Sharpe Ratio indicates that a portfolio's returns are higher relative to the amount of risk taken. Conversely, a lower Sharpe Ratio suggests that the returns do not justify the risk. Generally, a Sharpe Ratio greater than 1.0 is considered good, while a ratio above 2.0 is considered very good.

    ### Example Calculation
    Suppose you have a portfolio with an expected annual return of 10% ($$R_p = 0.10$$), a risk-free rate of 3% ($$R_f = 0.03$$), and a standard deviation of 15% ($$\sigma_p = 0.15$$). The Sharpe Ratio would be calculated as follows:
    $$Sharpe Ratio = \frac{0.10 - 0.03}{0.15} = \frac{0.07}{0.15} = 0.47$$
    In this case, the Sharpe Ratio of 0.47 suggests that the portfolio's returns are relatively low compared to the amount of risk being taken. 

    ### Uses of the Sharpe Ratio
    - **Portfolio Comparison**: The Sharpe Ratio allows investors to compare different portfolios or investments on a risk-adjusted basis. A higher Sharpe Ratio indicates a more favorable risk-return tradeoff.
    - **Performance Evaluation**: Portfolio managers often use the Sharpe Ratio to assess the performance of their investments relative to a benchmark or other portfolios.
    - **Risk Management**: By focusing on risk-adjusted returns, the Sharpe Ratio helps investors make more informed decisions about which investments provide the best returns for the level of risk assumed.

    ### Limitations
    While the Sharpe Ratio is widely used, it has some limitations:
    - It assumes that returns are normally distributed, which may not always be the case.
    - It does not differentiate between upward and downward volatility, treating all volatility as a negative factor.

    Despite these limitations, the Sharpe Ratio remains one of the most popular and useful tools for evaluating investment performance.
    """)

st.markdown("""
<a href="https://www.linkedin.com/in/m-meswania/" style="text-decoration:none;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width:25px; height:25px; vertical-align:middle; margin-right:5px; margin-bottom:5px"> 
    Meswania, Mihir
</a>
""", unsafe_allow_html=True)
