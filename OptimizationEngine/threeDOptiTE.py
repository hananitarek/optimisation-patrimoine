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
# import loadData as loadData

def solver(esg_max = 100, symbols = ['MC.PA', 'ORAN', 'MDM.PA'], weights = [0.5, 0.3, 0.2]):
    FICHIER = 'stock_data_french.csv'
    ESG_FILE = 'stock_data_french_esg.csv'
    chemin_complet = os.path.join('DataProvider', FICHIER)
    chemin_complete_esg = os.path.join('DataProvider', ESG_FILE)

    dailyReturns, ethicGrades, DailyPrices, stock_name = loadData.get_Universe(chemin_complet, chemin_complete_esg)
    
    data_index_tracked, dailyprices_index = loadData.get_index(symbols)
    data, dailyprices, esg_data = loadData.get_newdata(dailyReturns, ethicGrades, DailyPrices, symbols, data_index_tracked, dailyprices_index)


    
    stock_name = data.columns.values

    X = loadData.get_sets(data, stock_name)
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


    # yield_index_krach which uses a beta law to simulate the yield of the index in case of a krach around 20%
    yield_index_krach = yields - (0.2*np.random.beta(2, 5, len(yields)) * np.pi / 2)
    

    numAssets = np.shape(X)[1]

    x = cp.Variable(numAssets)
    C = np.zeros(numAssets)
    # complete the C matrix with the weights of the assets we want to track at the end
    for i in range(len(symbols)):
        C[-i-1] = weights[i]
    
    ethic = esg_data['esg_score'].values
    # add 0 to the end of the array
    for i in range(len(symbols)):
        ethic = np.append(ethic, 0)
    
    
    # Create constraints.
    constraints = [
        cp.sum(x) == 1,
        0 <= x,
        x <= 1,
        # x @ ethic <= esg_max,
        (x - C) @ yield_index_krach >= 15
    ]
    indexConstraints = []
    for i in range(len(symbols)):
        indexConstraints.append(x[-i-1] == 0)


    constraints = constraints + indexConstraints
    objective1 = cp.Maximize((x - C) @ yields)
    objective = cp.Minimize(cp.quad_form(x - C, cov))
    prob = cp.Problem(objective1, constraints) 
    prob.solve(solver="SCS", verbose = False)

    # if x.value are very small negative numbers, we set them to 0
    x.value[x.value < 0] = 0

    # rendement moyen du portefeuille
    portfolio_return = x.value @ yields
    dailyprices_index = dailyprices.values @ C
    dailyprices_x = dailyprices.values @ x.value

    dailyprices_index = normalizing(dailyprices_index)
    dailyprices_x = normalizing(dailyprices_x)

    # extract all dates 
    # create a dataframe containing the returns of the index and the returns of the portfolio
    df = pd.DataFrame({'index': dailyprices_index, 'portfolio': dailyprices_x, 'date': dailyprices.index})


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

    # create a dataframe containing the composition of the portfolio
    composition = pd.DataFrame({'symbol': stock_name, 'weight (%)': x.value * 100})
    composition = composition.sort_values(by='weight (%)', ascending=False)

    # print the krach yield of the portfolio
    print("Krach yield : ", x.value @ yield_index_krach)


    return df, performance, index_performance, weights, composition



def normalizing(returns):
    return (returns / returns[0])

def min_max_normalization(returns):
    normalized = (returns - returns.mean()) / (returns.max() - returns.min())
    return normalized

if __name__ == "__main__":
    solver()