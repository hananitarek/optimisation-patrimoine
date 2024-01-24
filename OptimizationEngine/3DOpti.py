#! /usr/bin/env python3
import os
import math
import cvxpy as cp
import numpy as np
import pandas as pd 
import matplotlib.pyplot as brami #pipelette as brami
from skimage.measure import block_reduce
from scipy.linalg import cholesky
from statsmodels.stats.correlation_tools import cov_nearest


import loadData
import seaborn as sns



def main():
    FICHIER = 'stock_data.csv'
    chemin_complet = os.path.join('DataProvider', FICHIER)
    outData, dailyReturns, prices, stock_symbol = loadData.loadUniverse(chemin_complet)

    outData = [asset for asset in outData if asset.numPrices > 3 * 356]  # we want at least 3 years of data
    

    weeklyYields = prices.pct_change().dropna().mean()

    print(weeklyYields)
    numAssets = len(outData)
    weights = cp.Variable(numAssets)

    ethic = np.array([asset.ethicGrade for asset in outData])
    print(ethic)
    crisis = np.array([asset.crisisYield for asset in outData])

    df = pd.DataFrame(dailyReturns)
    covariance_matrix = df.cov()
    covariance_matrix = cov_nearest(covariance_matrix, method='clipped', threshold=0.000001)

    # print True if the matrix is positive definite
    print(np.all(np.linalg.eigvals(covariance_matrix) > 0))


    # covariance_matrix = computeAndCorrectCov(dailyReturns)
    # sns.heatmap(covariance_matrix)
    # brami.show()

    # print the dimensions of the covariance matrix
    print(covariance_matrix.shape)
    print(weights.shape)

    # Create constraints.
    constraints = [cp.sum(weights) == 1, weights >= 0]
    constraints += [
        # weights @ ethic <= 0.3,
        cp.quad_form(weights, covariance_matrix) <= 1
    ]
    objective = cp.Maximize(weights @ weeklyYields)
    prob = cp.Problem(objective, constraints) 
    result = prob.solve(solver='SCS', verbose = True)

    print(weights.value)
    # print the ethic grade of the portfolio
    print(weights.value @ ethic)
    # print the variance of the portfolio
    print(weights.value @ covariance_matrix @ weights.value)


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

if __name__ == "__main__":
    main()