import pandas as pd

def simple_moving_avg(interval, prices):
    return prices.rolling(interval).mean()



