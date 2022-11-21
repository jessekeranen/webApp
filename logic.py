import pandas as pd
import yfinance as yf
import numpy as np

def getData(names, interv):
    tickers = ' '.join(names)
    df = yf.download(tickers=tickers, interval=interv, group_by="ticker", auto_adjust=False, prepost=False, threads=10)
    df.reset_index(inplace=True)

    df.dropna(inplace=True)

    labels = df.iloc[:, 0].values
    labels = np.datetime_as_string(labels, unit="D")
    labels = labels.tolist()

    df = pd.melt(df, id_vars=['Date'])
    df = pd.pivot_table(df, index=['Date', 'variable_0'], columns='variable_1', values='value').reset_index()
    df.rename(columns={'variable_0':'Name'}, inplace=True)

    df2 = pd.pivot_table(df, index=['Date'], columns='Name', values='Adj Close').reset_index()
    df2 = df2.iloc[:, df2.columns != 'Date']

    dr_rounded = df.round(2).astype(object)

    prices = df2.values
    prices = np.transpose(np.asmatrix(prices, dtype=str)).tolist()

    return dr_rounded, labels, prices, tickers

dt, lab, val, tick = getData(["AAPL", "MSFT", "META", "AMZN"], "1mo")
dt