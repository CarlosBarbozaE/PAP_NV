
import pandas as pd
import yfinance as yf
import datetime

now = datetime.datetime.now()
tickers = pd.read_csv('CNDX_holdings.csv', skiprows=2).dropna()
tickers = tickers['Ticker']

precios = {}
for i in tickers:
    pr = yf.download(i, start='2015-01-01', end=now, interval='1d')
    pr = pr.drop(['Adj Close', 'Volume'], axis=1)
    pr = pr.reset_index()
    precios[i] = pr
