#! /usr/bin/env python3
import os
import math
import cvxpy as cp
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.linalg import cholesky


from statsmodels.stats.correlation_tools import cov_nearest

import OptimizationEngine.loadData as loadData


def solver(esg_max, symbol = 'MC.PA'):
    FICHIER = 'stock_data_french.csv'
    ESG_FILE = 'stock_data_french_esg.csv'
    chemin_complet = os.path.join('DataProvider', FICHIER)
    chemin_complete_esg = os.path.join('DataProvider', ESG_FILE)

    dailyReturns, ethicGrades, DailyPrices, stock_name = loadData.get_Universe(chemin_complet, chemin_complete_esg)
    
    data_index_tracked = loadData.get_index([symbol])
    data, dailyprices, esg_data = loadData.get_newdata(dailyReturns, ethicGrades, DailyPrices, [symbol], data_index_tracked)

    ethic = esg_data['esg_score'].values
    # add 0 to the end of the array
    ethic = np.append(ethic, 0)
    
    stock_name = data.columns.values

    X = loadData.get_sets(data, stock_name, [symbol])
    cov = np.cov(X, rowvar=False)
    cov = cov_nearest(cov, method="nearest", threshold= 1e-15)
    

    # calculate the yield over the period of time : need to take into account the last prices and the first prices of the dataframe
    yields = pd.DataFrame()
    for symbol in dailyprices.columns.values:
        current_row = pd.DataFrame()
        current_row['symbol'] = [symbol]
        current_row['yield'] = (dailyprices[symbol].iloc[-1] - dailyprices[symbol].iloc[0]) / dailyprices[symbol].iloc[0]
        yields = pd.concat([yields, current_row], ignore_index=True)

    # last yield is the yield of the index
    yield_index = yields['yield'].iloc[-1]

    yields = yields['yield'].values


    numAssets = np.shape(X)[1]

    x = cp.Variable(numAssets)
    C = np.zeros(numAssets)
    C[-1] = 1

    
    
    # Create constraints.
    constraints = [
        cp.sum(x) == 1,
        0 <= x,
        x <= 1,
        # min(yields) <= x @ yields,
        x @ ethic <= esg_max
    ]
    indexConstraints = [
        x[-1] == 0
    ]
    constraints = constraints + indexConstraints
    objective = cp.Minimize(cp.quad_form(x - C, cov))
    prob = cp.Problem(objective, constraints) 
    prob.solve(solver="SCS", verbose = False)

    # if x.value are very small negative numbers, we set them to 0
    x.value[x.value < 0] = 0

    # rendement moyen du portefeuille
    portfolio_return = x.value @ yields
    dailyprices_index = dailyprices.values @ C
    dailyprices_x = dailyprices.values @ x.value

    # dailyprices_index = min_max_normalization(dailyprices_index)
    # dailyprices_x = min_max_normalization(dailyprices_x)

    dailyprices_index = normalizing(dailyprices_index)
    dailyprices_x = normalizing(dailyprices_x)

    # extract all dates 
    # create a dataframe containing the returns of the index and the returns of the portfolio
    df = pd.DataFrame({'index': dailyprices_index, 'portfolio': dailyprices_x, 'date': data.index})


    # create a dictionary containing the weights of 10 assets with the highest weights and sixth asset is the sum of the others
    weights = {}
    for i in x.value.argsort()[-15:][::-1]:
        weights[stock_name[i]] = x.value[i]
    weights['others'] = 1 - sum(weights.values())
    
    # if some weights are negative, we set them to 0
    for key, value in weights.items():
        if value < 0:
            weights[key] = 0

    # create a dictionary containing the tracking error and the ethic score
    performance = {}
    performance['tracking_error'] = abs(dailyprices_x[-1] - dailyprices_index[-1]) / dailyprices_index[0]
    performance['esg_score'] = x.value @ ethic
    performance['portfolio_return'] = portfolio_return
    performance['portfolio_risk'] = x.value @ cov @ x.value

    index_performance = {}
    index_performance['index_return'] = yield_index
    index_performance['index_risk'] = C @ cov @ C




    return df, performance, index_performance, weights


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
    return (returns / returns[0])

def min_max_normalization(returns):
    normalized = (returns - returns.mean()) / (returns.max() - returns.min())
    return normalized

if __name__ == "__main__":
    solver()