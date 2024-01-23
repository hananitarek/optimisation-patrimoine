import csv
import random
from os import path
import pandas as pd

import numpy as np
from tqdm import tqdm
from precise_asset import BetterAsset
from asset import Asset

#set a seed for reproducibility



def loadUniverse(file):
    stocks = pd.read_csv(file)
    print(stocks)

    # stock_symbol = stocks[stocks.date == '2013-02-08'].symbol.values
    # get the list of stock symbols
    stock_symbol = stocks.symbol.unique()
    

    stocks['return'] = np.log(stocks['close']) - np.log(stocks['close'].shift())
    drop = np.array([0])

    for i in tqdm(range(1, stocks.shape[0]), "Traitement des données"):
        if stocks['symbol'][i] != stocks['symbol'][i-1]:
            drop = np.append(drop, i)

    stocks.drop(index = drop, inplace=True)

    prices = stocks.copy().loc[:,['date','symbol','close']]
    prices = prices.pivot(index='date', columns='symbol', values='close')
    prices = prices[stock_symbol].copy()

    returns = stocks.copy().loc[:,['date','symbol','return']]
    returns = returns.pivot(index='date', columns='symbol', values='return')
    returns = returns[stock_symbol].copy()

    opens = stocks.copy().loc[:,['date','symbol','open']]
    opens = opens.pivot(index='date', columns='symbol', values='open')
    opens = opens[stock_symbol].copy()

    highs = stocks.copy().loc[:,['date','symbol','high']]
    highs = highs.pivot(index='date', columns='symbol', values='high')
    highs = highs[stock_symbol].copy()

    lows = stocks.copy().loc[:,['date','symbol','low']]
    lows = lows.pivot(index='date', columns='symbol', values='low')
    lows = lows[stock_symbol].copy()

    closes = stocks.copy().loc[:,['date','symbol','close']]
    closes = closes.pivot(index='date', columns='symbol', values='close')
    closes = closes[stock_symbol].copy()

    volumes = stocks.copy().loc[:,['date','symbol','volume']]
    volumes = volumes.pivot(index='date', columns='symbol', values='volume')
    volumes = volumes[stock_symbol].copy()

    ethicGrades = stocks.copy().loc[:,['date', 'symbol','note_ethique']]

    ethicGrades = ethicGrades.pivot(index='date', columns='symbol', values='note_ethique')
    ethicGrades = ethicGrades[stock_symbol].copy()

    outData = []
    for i in range(len(stock_symbol)):
        curAsset = BetterAsset()
        curAsset.symbol = stock_symbol[i]
        curAsset.open = opens.iloc[:,i].values
        curAsset.high = highs.iloc[:,i].values
        curAsset.low = lows.iloc[:,i].values
        curAsset.close = closes.iloc[:,i].values
        curAsset.volume = volumes.iloc[:,i].values

        curAsset.ethicGrade = ethicGrades.iloc[:,i].values[0]

        maximumOpens = max(curAsset.open)
        minimumLows = min(curAsset.low)
        firstOpen = curAsset.open[0]

        curAsset.crisisYield = (minimumLows - maximumOpens) / firstOpen

        outData.append(curAsset)

    
    return outData, returns, prices, stock_symbol

