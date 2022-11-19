import yfinance as yf
import numpy as np

def getData(ticker, interv):
    df = yf.download(tickers=ticker, interval=interv, group_by="ticker", auto_adjust=False, prepost=False, threads=10)
    df.reset_index(inplace=True)
    labels = df.iloc[:, 0].values
    labels = np.datetime_as_string(labels, unit="D")
    labels = labels.tolist()

    values = df.iloc[:, 5].values.tolist()
    return df, labels, values


df, labels, values = getData("AAPL", "1mo")
labels
values