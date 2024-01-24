import pandas as pd
import yfinance as yf
import numpy as np
from tqdm import tqdm
from precise_asset import BetterAsset
from pandas_datareader import data as wb

#set a seed for reproducibility


def get_Universe(file):
    stocks = pd.read_csv(file)
    stocks = stocks[['date', 'symbol', 'close', 'note_ethique']]

    stocks['return'] = np.log(stocks['close']) - np.log(stocks['close'].shift())

    drop = np.array([0])

    for i in tqdm(range(1, len(stocks)), "Traitement des données"):
        if stocks['symbol'][i] != stocks['symbol'][i-1]:
            drop = np.append(drop, i)


    stocks.drop(index = drop, inplace = True)

    

    dailyReturns = stocks.copy().loc[:,['date','symbol','return']]
    dailyReturns = stocks.pivot(index='date', columns='symbol', values='return')
    stock_name = dailyReturns.columns.values
    dailyReturns = dailyReturns[stock_name].copy()

    ethicGrades = stocks.copy().loc[:,['date', 'symbol','note_ethique']]
    ethicGrades = ethicGrades.pivot(index='date', columns='symbol', values='note_ethique')
    ethicGrades = ethicGrades[stock_name].copy()
    
    return dailyReturns, ethicGrades, stock_name

def get_index(symbol):
    yf.pdr_override()
    data_index = pd.DataFrame()
    data_index = wb.get_data_yahoo(symbol,start='2013-04-16', end='2023-04-13', interval='1d')
    data_index['return'] = np.log(data_index['Close']) - np.log(data_index['Close'].shift())
    return data_index


def get_newdata(stocks, symbols, data_index):
    data = stocks.copy()

    # supprimer les colonnes des indices qui ont trop de valeurs manquantes
    data.dropna(axis=1, thresh=0.8*len(data), inplace=True)


    # si le symbole est déjà connu, on retire la colonne correspondante
    for sym in tqdm(symbols, "Checking for duplicates"):
        if sym in data.columns:
            data.drop(sym, axis=1, inplace=True)
        data[sym] = data_index['return'].copy()


    data.dropna(inplace=True) # drop lines with NaN values

    print(data)

    return data


def get_sets(data, stock_name, symbol):
    X = data[stock_name].values
    returns = data[symbol].values
    return X, returns



def loadUniverse(file):
    stocks = pd.read_csv(file)
    # stock_symbol = stocks[stocks.date == '2013-02-08'].symbol.values
    # get the list of stock symbols
    stock_symbol = stocks.symbol.unique()
    

    stocks['return'] =  np.log(stocks['close']) - np.log(stocks['close'].shift())

    # stocks.dropna(inplace=True)

    prices = stocks.copy().loc[:,['date','symbol','close']]
    prices = prices.pivot(index='date', columns='symbol', values='close')
    prices = prices[stock_symbol].copy()

    dailyReturns = stocks.copy().loc[:,['date','symbol','return']]
    dailyReturns = dailyReturns.pivot(index='date', columns='symbol', values='return')
    dailyReturns = dailyReturns[stock_symbol].copy()

    opens = stocks.copy().loc[:,['date','symbol','open']]
    opens = opens.pivot(index='date', columns='symbol', values='open')
    opens = opens[stock_symbol].copy()

    highs = stocks.copy().loc[:,['date','symbol','high']]
    highs = highs.pivot(index='date', columns='symbol', values='high')
    highs = highs[stock_symbol].copy()

    lows = stocks.copy().loc[:,['date','symbol','low']]
    lows = lows.pivot(index='date', columns='symbol', values='low')
    lows = lows[stock_symbol].copy()

    volumes = stocks.copy().loc[:,['date','symbol','volume']]
    volumes = volumes.pivot(index='date', columns='symbol', values='volume')
    volumes = volumes[stock_symbol].copy()

    ethicGrades = stocks.copy().loc[:,['date', 'symbol','note_ethique']]
    ethicGrades = ethicGrades.pivot(index='date', columns='symbol', values='note_ethique')
    
    outData = []
    for i in range(len(stock_symbol)):
        curAsset = BetterAsset()
        curAsset.symbol = stock_symbol[i]
        curAsset.open = opens.iloc[:,i].values
        curAsset.low = lows.iloc[:,i].values
        curAsset.DailyPrices = prices.iloc[:,i].values
        
        curAsset.ethicGrade = np.nanmax(ethicGrades.iloc[:,i].values)

        maximumOpens = max(curAsset.open)
        minimumLows = min(curAsset.low)
        firstOpen = curAsset.open[0]

        curAsset.crisisYield = (minimumLows - maximumOpens) / firstOpen

        # count the number of prices non nan
        curAsset.numPrices = 0
        for price in curAsset.DailyPrices:
            if price != "nan":
                curAsset.numPrices += 1


        outData.append(curAsset)

    
    return outData, dailyReturns, prices, stock_symbol

