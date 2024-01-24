#! /usr/bin/env python3
import os
import math
import cvxpy as cp
import numpy as np
import pandas as pd 
import matplotlib.pyplot as brami #pipelette as brami
from skimage.measure import block_reduce
from scipy.linalg import cholesky
import yfinance as yf

from statsmodels.stats.correlation_tools import cov_nearest


import loadData
import seaborn as sns
from pandas_datareader import data as wb



def main():
    FICHIER = 'stock_data.csv'
    chemin_complet = os.path.join('DataProvider', FICHIER)
    outData, dailyReturns, prices, stock_symbol = loadData.loadUniverse(chemin_complet)

    yf.pdr_override()
    sp500 = pd.DataFrame()
    sp500 = wb.get_data_yahoo('^GSPC',start='2014-01-23', end='2024-01-20', interval='1d')
    
    sp500['return'] = np.log(sp500['Close']) - np.log(sp500['Close'].shift())
    # eject all column missing too much returns
    dailyReturns = dailyReturns.dropna(axis=1, thresh= 0.9 * len(dailyReturns))
    
    outData = [asset for asset in outData if asset.symbol in dailyReturns.columns]


    #eject all rows having nan
    dailyReturns = dailyReturns.dropna(axis=0, how='any')
    # we need to have the same date for sp500 and dailyReturns
    sp500 = sp500.loc[dailyReturns.index]
    # eject all column missing too much returns
    




    # add the sp500 to the dataframe
    dailyReturns['^GSPC'] = sp500['return'].copy()

    weeklyYields = dailyReturns.mean()

    numAssets = len(dailyReturns.columns)
    weights = cp.Variable(numAssets)
    


    ethic = np.array([asset.ethicGrade for asset in outData])
    crisis = np.array([asset.crisisYield for asset in outData])

    # add 0 to ethic and crisis
    ethic = np.append(ethic, 0)
    crisis = np.append(crisis, 0)

    df = pd.DataFrame(dailyReturns)
    covariance_matrix = df.cov()
    covariance_matrix = cov_nearest(covariance_matrix, method='clipped', threshold=0.000001)

    C = np.zeros(numAssets)
    C[-1] = 1

    # Create constraints.
    constraints = [cp.sum(weights) == 1, 0 <= weights, weights <= 1]
    constraints += [
        # weights @ weeklyYields >= 0.000,
        # weights @ ethic >= 0.0,
        weights[-1] == 0
    ]
    objective = cp.Minimize(cp.quad_form(weights - C, covariance_matrix ))
    prob = cp.Problem(objective, constraints) 
    prob.solve(solver='SCS', verbose = False)
    
    # display expected return
    print("weights.value @ weeklyYields :", weights.value @ weeklyYields)
    #display risk
    print(cp.quad_form(weights, covariance_matrix).value)
    # display ethic grade
    print(weights.value @ ethic)
    # display 10 best assets
    print(weights.value.argsort()[-10:][::-1])
    # display names of 10 best assets
    print([outData[i].symbol for i in weights.value.argsort()[-10:][::-1]])

    # display weights
    print(weights.value)

    portfolio_returns = normalizing(dailyReturns.values @ weights.value)
    sp500_returns = normalizing(dailyReturns.values @ C)
    brami.plot(portfolio_returns, label='Portfolio')
    brami.plot(sp500_returns, label='S&P500')
    brami.legend()
    brami.show()

def computeAverageYields(data, duration):
    res = []
    for asset in data:
        indexMin = -1
        indexMax = -1
        for i in range(len(asset.DailyPrices)):
            if asset.DailyPrices[i] != "nan":
                indexMax = i
                if indexMin == -1:
                    indexMin = i

        firstVal = float(asset.DailyPrices[indexMin])
        lastVal = float(asset.DailyPrices[indexMax])

        globalYield = (lastVal - firstVal) / firstVal
        periods = (indexMax - indexMin) / duration
        perPeriodYield = math.pow(1 + globalYield, 1 / periods) - 1
        res.append((perPeriodYield))

    return np.array(res).astype(float)


def computeAndCorrectCov(yields):
    df = pd.DataFrame(np.transpose(yields)) 
    covariance_matrix = df.cov()

    validCovMat = False
    numAssets = yields.shape[0]
    toadd = 0.000001

    while not validCovMat:
        try:
            r = cholesky(covariance_matrix + toadd * np.eye(numAssets))
            validCovMat = True
        except:
            toadd *= 1.1
    
    return covariance_matrix + toadd * np.eye(numAssets)

def normalizing(returns):
    normalized = np.array([1])
    for i in returns:
        normalized = np.append(normalized, normalized[-1] * np.exp(i))
    return normalized


if __name__ == "__main__":
    main()