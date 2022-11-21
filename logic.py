import pandas as pd
import yfinance as yf
import numpy as np


def getdata(names, interv):
    tickers = ' '.join(names)
    df = yf.download(tickers=tickers, interval=interv, group_by="ticker", auto_adjust=False, prepost=False, threads=10)
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

    df2 = pd.pivot_table(df, index=['Date'], columns='Name', values='Adj Close').reset_index()
    df2 = df2.iloc[:, df2.columns != 'Date']
    prices = df2.values
    prices = np.transpose(np.asmatrix(prices, dtype=str)).tolist()

    dr_rounded = df.round(2).astype(object)

    ret = stock_returns(df)
    sd = stock_std(df)
    cov = cov_matrix(pd.pivot_table(df, index=['Date'], columns='Name', values='Return').reset_index())

    d = []
    for i in range(1000):
        wght = random_weights(len(names))

        temp_ret = portfolio_return(ret, wght)
        temp_sd = portfolio_std(cov, wght)

        d.append([temp_sd, temp_ret, portfolio_sharpe(temp_ret, temp_sd)])
    ret_sd = np.array(d)[:, 0:2].tolist()
    sharpe = color_codes(np.array(d)[:, 2].tolist())


    return dr_rounded, labels, prices, tickers, ret_sd, sharpe


def stock_returns(df):
    return df.groupby("Name")["Return"].mean()


def stock_std(df):
    return df.groupby("Name")["Return"].std()


def cov_matrix(df):
    return df.cov()


def portfolio_return(returns, weights):
    return np.dot(returns, weights)


def portfolio_std(cov_mat, weights):
    return np.sqrt(np.dot(weights.T, (np.dot(cov_mat, weights))))


def random_weights(n):
    array = np.random.uniform(0, 1, n)
    return array / np.sum(array)


def portfolio_sharpe(port_ret, port_sd):
    return port_ret/port_sd


def color_codes(array):
    d=[]
    for x in array:
        x = ((x - min(array)) / (max(array) - min(array))) * (57 - 0)
        d.append('hsl(204, 82%, ' + str(x) + '%)')
    return d


dt, lab, val, tick, rand, sharpe = getdata(["AAPL", "MSFT", "META", "AMZN", "NFLX"], "1mo")
arr = random_weights(5)
arr

