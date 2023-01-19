import numpy
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize


def getdata(names, interv):
    tickers = check_tickers(names)

    df, labels = stock_information(tickers, interv)

    prices = data_for_price_chart(df)

    ret = stock_returns(df)
    sd = stock_std(df)
    cov = cov_matrix(pd.pivot_table(df, index=['Date'], columns='Name', values='Return').reset_index())

    ret_sd, color = random_portfolios(tickers, ret, cov)

    weights = []
    for j in range(len(tickers)):
        weights.append(1 / len(tickers))
    weights = np.array(weights)

    sharpe_portfolio = minimize(
        lambda x: penalized_function(x, optimization, ret, cov.values, -2, 100), weights, method='Nelder-Mead',
        options={'disp': False})  # calculates portfolio with highest sharpe ratio

    min_var_portfolio = minimize(
        lambda x: penalized_function(x, optimization, ret, cov.values, -1, 100), weights, method='Nelder-Mead',
        options={'disp': False})

    min_var_port_ret = portfolio_return_yearly(ret, min_var_portfolio.x)
    max_ret = max(ret) * 12

    final_weights, eff_frontier = efficient_frontier(ret, cov, min_var_port_ret, max_ret, weights)


    sharpe_port_ret_arr = portfolio_return_monthly(pd.pivot_table(df, index=['Date'], columns='Name', values='Adj Close').reset_index().set_index("Date").pct_change(), sharpe_portfolio.x)

    sharpe_port_year_returns = year_returns(sharpe_port_ret_arr)
    sharpe_port_year_returns = sharpe_port_year_returns[0].values.tolist()

    info = {"Weight": sharpe_portfolio.x, "Company": tickers, "Return": ret.to_list(), "Std.": sd.to_list(), "Sharpe": (ret / sd).to_list()}
    info = pd.DataFrame(info)
    info = info.set_index("Company")

    return df, labels, prices, tickers, ret_sd, color, eff_frontier, final_weights, info, sharpe_port_year_returns


def stock_information(tickers, interval):
    df = yf.download(tickers=tickers, interval=interval, group_by="ticker", rounding=True, auto_adjust=False,
                     prepost=False, threads=10)
    df.reset_index(inplace=True)
    df.dropna(inplace=True)

    labels = df.iloc[:, 0].values
    labels = np.datetime_as_string(labels, unit="D")
    labels = labels.tolist()

    df = pd.melt(df, id_vars=['Date'])
    df = pd.pivot_table(df, index=['Date', 'variable_0'], columns='variable_1', values='value').reset_index()
    df.rename(columns={'variable_0': 'Name'}, inplace=True)
    df["Return"] = df.groupby("Name")["Adj Close"].pct_change(1)
    df.dropna(inplace=True)

    return df, labels


def data_for_price_chart(df):
    df2 = pd.pivot_table(df, index=['Date'], columns='Name', values='Adj Close').reset_index()
    df2 = df2.iloc[:, df2.columns != 'Date']
    prices = df2.values
    prices = np.transpose(np.asmatrix(prices, dtype=str)).tolist()

    return prices


def random_portfolios(tickers, ret, cov):
    d = []
    for i in range(1000):
        wght = random_weights(len(tickers))

        temp_ret = portfolio_return_yearly(ret, wght)
        temp_sd = portfolio_std(cov, wght)

        d.append([temp_sd, temp_ret, portfolio_sharpe(temp_ret, temp_sd)])
    ret_sd = np.array(d)[:, 0:2].tolist()
    color = color_codes(np.array(d)[:, 2].tolist())

    return ret_sd, color


def efficient_frontier(ret, cov, min_var_port_ret, max_ret, weights):
    array = []
    final_weights = []
    for n in range(20):
        res = minimize(
            lambda x: penalized_function(x, optimization, ret, cov.values,
                                         (min_var_port_ret + ((n + 1) / 20) ** 2 * (max_ret - min_var_port_ret)),
                                         100), weights, method='Nelder-Mead',
            options={'disp': False})  # calculates portfolios for efficient frontier
        array.append(res)

        # could be that despite restriction slightly negative weight is returned
        res.x[numpy.where(res.x < 0)] = 0
        res.x =  np.round(res.x, 2)
        res.x = res.x / np.sum(res.x)
        final_weights.append(res.x)


    final_weights = np.array(final_weights).T.tolist()

    eff_frontier = []
    for k in array:
        eff_frontier.append([portfolio_std(cov, k.x), portfolio_return_yearly(ret, k.x)])
    eff_frontier = np.array(eff_frontier).tolist()

    return final_weights, eff_frontier


def stock_returns(df):
    return df.groupby("Name")["Return"].mean()


def stock_std(df):
    return df.groupby("Name")["Return"].std()


def cov_matrix(df):
    return df.cov()


'''
    Calculates mean portfolio return. Basically weighted sum of the means of the asset returns.
'''

def portfolio_return_yearly(returns, weights):
    return np.dot(returns, weights) * 12


def portfolio_return_monthly(returns, weights):
    temp = np.dot(returns, weights)
    temp[0] = 0
    temp = pd.DataFrame(temp)
    temp = temp.set_index(returns.index)
    return temp


def year_returns(prices):
    prices.index = pd.to_datetime(prices.index)
    prices[0] = (1 + prices[0]).cumprod()
    return prices[prices.index.month == 1].pct_change().dropna()


def portfolio_std(cov_mat, weights):
    return np.sqrt(np.dot(weights.T, (np.dot(cov_mat, weights))) * 12)


def random_weights(n):
    array = np.random.uniform(0, 1, n)
    return array / np.sum(array)


def portfolio_sharpe(port_ret, port_sd):
    return port_ret / port_sd


def color_codes(array):
    d = []
    for x in array:
        x = ((x - min(array)) / (max(array) - min(array))) * (57 - 0)
        d.append('hsl(204, 82%, ' + str(x) + '%)')
    return d


'''
    Method that is used to decide what is optimized. If target value is -1 we try to find portfolio that minimizes 
    variance. If target value is -2 we want to find portfolio that has greatest trade of between return and variance 
    aka sharpe ratio. If target value is something else we want to find portfolio that has that return but minimum
    variance.

    Parameters
    ----------
    profits = array of the logarithmic returns
    weight  = array of the weights of different assets
    cov     = covariance matrix
    target  = indicates what we want to optimize

    Returns 
    ----------
    Variance or sharpe ratio of the function depending on value of target
'''


def optimization(profits, weights, cov, target):
    if target == -1:
        return portfolio_std(cov, weights), weights, [np.sum(weights) - 1]
    if target == -2:
        return -portfolio_return_yearly(profits, weights) / portfolio_std(cov, weights), weights, [np.sum(weights) - 1]
    else:
        return portfolio_std(cov, weights), weights, [np.sum(weights) - 1, portfolio_return_yearly(profits, weights) - target]


'''
    Calculates value of the penalized function. For each set of weights penalty term is added if weights violate
    constraints.

    Parameters
    ----------
    x       = array of weights
    f       = function that is optimized
    profits = array of the logarithmic returns
    cov     = covariance matrix
    target  = indicates what we want to optimize
    r       = penalty coefficient

    Returns 
    -----------
    Value of the penalized function
'''


def penalized_function(x, f, profits, cov, target, r):
    return f(profits, x, cov, target)[0] + r * alpha(profits, x, f, cov, target)


'''
    Calculates the values of the penalty function. If the given point violates the constraints, value of the penalty 
    function is increased.

    Parameters
    ----------
    profits = array of the logarithmic returns
    x       = array of weights
    f       = function that is optimized
    cov     = covariance matrix
    target  = indicates what we want to optimize

    Returns 
    ----------
    Value of the penalty term
'''


def alpha(profits, x, f, cov, target):
    (_, ieq, eq) = f(profits, x, cov, target)
    return sum([min([0, ieq_j]) ** 2 for ieq_j in ieq]) + sum([eq_k ** 2 for eq_k in eq])


def check_tickers(tickers):
    existing_tickers = []
    for ticker in tickers:
        try:
            if yf.Ticker(ticker).isin == '-':
                raise ValueError()
            existing_tickers.append(ticker)
        except:
            continue

    existing_tickers = ' '.join(existing_tickers).split()
    existing_tickers.sort()
    return existing_tickers


# dt, lab, val, tick, rand, sharpe, eff_frontier, jk, info, year = getdata(["", "AAPL", "NLFX", "MSFT", "META"], "1mo")
arr = random_weights(5)
arr
