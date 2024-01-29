import pandas as pd
import yfinance as yf
import numpy as np
from tqdm import tqdm
from pandas_datareader import data as wb
import streamlit as st


def get_Universe(file, esg_file):
    stocks = pd.read_csv(file)
    esg = pd.read_csv(esg_file)

    stocks = stocks[['date', 'symbol', 'Close']]
    esg = esg[['symbol', 'esg_score']]

    # stocks['return'] = stocks['Close'].pct_change()
    stocks['return'] = np.log(stocks['Close'].pct_change() + 1)

    drop = np.array([0])
    progress_bar = st.sidebar.progress(0, text="Traitement des données")
    for i in tqdm(range(1, len(stocks)), "Traitement des données"):
        if stocks['symbol'][i] != stocks['symbol'][i-1]:
            progress_bar.progress(int(i / len(stocks) * 100), text="Traitement des données")
            drop = np.append(drop, i)
    progress_bar.empty()
    stocks.drop(index = drop, inplace = True)

    dailyReturns = stocks.copy().loc[:,['date','symbol','return']]
    dailyReturns = stocks.pivot(index='date', columns='symbol', values='return')
    stock_name = dailyReturns.columns.values
    dailyReturns = dailyReturns[stock_name].copy()

    ethicGrades = esg.copy().loc[:,['symbol','esg_score']]

    DailyPrices = stocks.copy().loc[:,['date','symbol','Close']]
    DailyPrices = stocks.pivot(index='date', columns='symbol', values='Close')
    DailyPrices = DailyPrices[stock_name].copy()
    
    return dailyReturns, ethicGrades, DailyPrices, stock_name

def get_index(symbol):
    yf.pdr_override()
    data_index = pd.DataFrame()
    data_index = wb.get_data_yahoo(symbol,start='2013-04-16', end='2023-04-13', interval='1d')
    # data_index['return'] = np.log(data_index['Close']) - np.log(data_index['Close'].shift())
    # data_index['return'] = data_index['Close'].pct_change()
    data_index['return'] = np.log(data_index['Close'].pct_change() + 1)
    return data_index


def get_newdata(stocks, esg_data, dailyprices, symbols, data_index):
    data = stocks.copy()
    esg_d = esg_data.copy()
    dailyprices_d = dailyprices.copy()

    # supprimer les colonnes des indices qui ont trop de valeurs manquantes
    data.dropna(axis=1, thresh=0.8*len(data), inplace=True)
    # supprimer les lignes de esg_data qui ne sont pas dans data
    esg_d = esg_d[esg_d['symbol'].isin(data.columns)]
    # supprimer les colonnes de dailyprices qui ne sont pas dans data
    dailyprices_d = dailyprices_d[data.columns]



    # si le symbole est déjà connu, on retire la colonne correspondante
    for sym in tqdm(symbols, "Checking for duplicates"):
        if sym in data.columns:
            data.drop(sym, axis=1, inplace=True)
            # drop the corresponding esg score
            esg_d = esg_d[esg_d['symbol'] != sym]
            # drop the corresponding dailyprices
            dailyprices_d.drop(sym, axis=1, inplace=True)
            
        data[sym] = data_index['return'].copy()
        dailyprices_d[sym] = data_index['Close'].copy()

    


    data.dropna(inplace=True) # drop lines with NaN values
    dailyprices_d.dropna(inplace=True) # drop lines with NaN values
    
    # sort the companies by their name
    esg_d = esg_d.sort_values(by=['symbol'])
    esg_d.reset_index(inplace=True, drop=True)


    return data, dailyprices_d, esg_d


def get_sets(data, stock_name, symbols):
    X = data[stock_name].values
    returns = data[symbols].values
    return X

