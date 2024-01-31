import pandas as pd
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import *

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

def get_index(symbols):
    data_index = pd.DataFrame()
    daily_prices_index = pd.DataFrame()
    to_date = datetime.today().date()
    from_date = to_date - relativedelta(years=20)

    for symbol in tqdm(symbols, "Getting index data"):
        yf.pdr_override()
        current_index = pd.DataFrame()
        current_index = wb.get_data_yahoo(symbol,start=from_date, end=to_date, interval='1d')
        current_index['symbol'] = symbol
        current_index['date'] = current_index.index
        current_index['return'] = np.log(current_index['Close'].pct_change() + 1)

        data_index = pd.concat([data_index, current_index])

    data_index_return = data_index.copy()[['date', 'symbol', 'return']]
    data_index_return = data_index_return.pivot(index='date', columns='symbol', values='return')
    data_index_return = data_index_return[symbols].copy()
    data_index_return.dropna(inplace=True)

    daily_prices_index = data_index.copy()[['date', 'symbol', 'Close']]
    daily_prices_index = daily_prices_index.pivot(index='date', columns='symbol', values='Close')
    daily_prices_index = daily_prices_index[symbols].copy()
    daily_prices_index.dropna(inplace=True)
    return data_index_return, daily_prices_index


def get_newdata(stocks, esg_data, dailyprices, symbols, data_index, daily_prices_index):
    data = stocks.copy()
    esg_d = esg_data.copy()
    dailyprices_d = dailyprices.copy()

    # supprimer les colonnes des indices qui ont trop de valeurs manquantes
    data.dropna(axis=1, thresh=0.95*len(data), inplace=True)
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


    for i in tqdm(range(len(symbols)), "Adding index data"):
        data[symbols[i]] = data_index[symbols[i]].copy()
        dailyprices_d[symbols[i]] = daily_prices_index[symbols[i]].copy()

    


    data.dropna(inplace=True) # drop lines with NaN values
    dailyprices_d.dropna(inplace=True) # drop lines with NaN values
    
    # sort the companies by their name
    esg_d = esg_d.sort_values(by=['symbol'])
    esg_d.reset_index(inplace=True, drop=True)


    return data, dailyprices_d, esg_d


def get_sets(data, stock_name):
    X = data[stock_name].values
    return X

