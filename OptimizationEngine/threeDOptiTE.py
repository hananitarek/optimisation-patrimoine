#! /usr/bin/env python3
import os
import math
import cvxpy as cp
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.linalg import cholesky


from statsmodels.stats.correlation_tools import cov_nearest

import loadData



def solver(symbol = 'NVDA'):
    FICHIER = 'stock_data.csv'
    chemin_complet = os.path.join('DataProvider', FICHIER)
    dailyReturns, ethicGrades, stock_name = loadData.get_Universe(chemin_complet)
    data_index_tracked = loadData.get_index([symbol])
    data = loadData.get_newdata(dailyReturns, [symbol], data_index_tracked)
    
    stock_name = data.columns.values

    X, returns = loadData.get_sets(data, stock_name, symbol)
    cov = np.cov(X, rowvar=False)
    cov = cov_nearest(cov, method="nearest", threshold= 1e-15)
    

    weeklyYields = data.mean()
    print(weeklyYields)
    print(returns.mean())

    numAssets = np.shape(X)[1]

    x = cp.Variable(numAssets)
    C = np.zeros(numAssets)
    C[-1] = 1

    # outData = [asset for asset in outData if asset.symbol in dailyReturns.columns]

    # ethic = np.array([asset.ethicGrade for asset in outData])
    # crisis = np.array([asset.crisisYield for asset in outData])

    # # add 0 to ethic and crisis
    # ethic = np.append(ethic, 0)
    # crisis = np.append(crisis, 0)


    # Create constraints.
    constraints = [
        cp.sum(x) == 1,
        0 <= x,
        x <= 1,
    ]
    indexConstraints = [
        # x @ weeklyYields >= 0.000,
        # x @ ethic >= 0.0,
        x[-1] == 0
    ]
    constraints = constraints + indexConstraints
    objective = cp.Minimize(cp.quad_form(x - C, cov))
    prob = cp.Problem(objective, constraints) 
    prob.solve(solver="SCS", verbose = False)
    
    # # display expected return
    # print("weights.value @ weeklyYields :", x.value @ weeklyYields)
    # #display risk
    # print(cp.quad_form(x, covariance_matrix).value)
    # # display ethic grade
    # print(x.value @ ethic)
    # # display 10 best assets
    # print(x.value.argsort()[-10:][::-1])
    # # display names of 10 best assets
    # print([outData[i].symbol for i in x.value.argsort()[-10:][::-1]])
    
    # rendement moyen du portefeuille
    portfolio_return = x.value @ weeklyYields
    # rendement moyen de l'indice
    index_return = returns.mean()
    # tracking error
    tracking_error = portfolio_return - index_return

    print('Portfolio return: ', portfolio_return * 100)
    print('Index return: ', index_return * 100)
    print('Tracking error: ', tracking_error * 100)

    returns_index = returns
    returns_x = X @ x.value

    returns_index = normalizing(returns_index)
    returns_x = normalizing(returns_x)

    # extract all dates 
    # create a dataframe containing the returns of the index and the returns of the portfolio
    df = pd.DataFrame({'index': returns_index, 'portfolio': returns_x, 'date': data.index})
    # df.set_index('date', inplace=True)
    # # plot 
    # plt.plot(returns_index, color='red')
    # plt.plot(returns_x, color='blue')
    # plt.legend(['index_tracked', 'portfolio'])

    # plt.show()

    # create a dictionary containing the weights of 10 assets with the highest weights and sixth asset is the sum of the others
    weights = {}
    for i in x.value.argsort()[-10:][::-1]:
        weights[stock_name[i]] = x.value[i]
    weights['others'] = 1 - sum(weights.values())



    return df, tracking_error, weights


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
    normalized = np.array([0])
    for i in returns[1 :]:
        normalized = np.append(normalized, normalized[-1] + i)
    return normalized

def min_max_normalization(returns):
    normalized = (returns - returns.mean()) / (returns.max() - returns.min())
    return normalized

if __name__ == "__main__":
    solver()