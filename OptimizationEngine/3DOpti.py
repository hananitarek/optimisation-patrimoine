#! /usr/bin/env python3
import os
import math
import cvxpy as cp
import numpy as np
import pandas as pd 
import matplotlib.pyplot as brami #pipelette as brami
from skimage.measure import block_reduce

import loadData
import seaborn as sns



def main():
    FICHIER = 'stock_data.csv'
    chemin_complet = os.path.join('DataProvider', FICHIER)
    outData, returns, prices, stock_symbol = loadData.loadUniverse(chemin_complet)

    weeklyLogYields = computeWeeklyLogYields(outData)

    print(weeklyLogYields)
    numAssets = len(outData)
    weights = cp.Variable(numAssets)

    ethic = np.array([asset.ethicGrade for asset in outData])
    crisis = np.array([asset.crisisYield for asset in outData])

    df = pd.DataFrame(returns)
    covariance_matrix = df.cov()
    # sns.heatmap(covariance_matrix)
    # brami.show()

    # Create constraints.
    constraints = [cp.sum(weights) == 1, weights >= 0]
    constraints += [
        weights @ ethic >= 0.3,
        cp.quad_form(weights, covariance_matrix) <= 0.1
    ]
    objective = cp.Maximize(weights @ weeklyLogYields)
    prob = cp.Problem(objective, constraints) 
    result = prob.solve(solver="SCS")

    print(result)


def computeWeeklyLogYields(data):
    # compute weekly logarithmic yields

    close = np.array([asset.close for asset in data]).astype(float)
    for i in range(close.shape[0]):
        prevVal = np.nan
        for j in range(close.shape[1]):
            if not np.isnan(close[i][j]):
                prevVal = close[i][j]
            else :
                close[i][j] = prevVal # Fill the missing values with the previous value
    
    weeklyPrices = block_reduce(close, block_size=(1, 7), func=np.nanmean)
    df = pd.DataFrame(weeklyPrices)
    df.ffill(axis=1)
    weeklyPrices = df.to_numpy()

    logYields = np.log(np.divide(weeklyPrices[:, 1:], weeklyPrices[:, :-1]))
    return logYields

if __name__ == "__main__":
    main()