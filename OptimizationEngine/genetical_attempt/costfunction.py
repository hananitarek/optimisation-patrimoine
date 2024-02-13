import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import chromosome as ch
from time import time as t
class IndexTracker:
    ''' 
    This class is used to track the index of the portfolio
    '''
    
    def __init__(self, taille_portefeuille, min_weight, periode, cout_transaction, stocks_name, stocks, index, param_mult, lmbda = 1):
        ''' 
        stocks : list of stocks
        
        taille_portefeuille : int : number of stocks in the portfolio

        min_weight : float : minimum weight a stock can take in the portfolio

        periode : int : number of days in the period

        cout_transaction : float : transaction cost as a percentage of the portfolio value

        param_mult : float : parameter used to multiply the fitness function
        '''

        self.taille_portefeuille = taille_portefeuille
        self.min_weight = min_weight
        self.periode = periode
        self.cout_transaction = cout_transaction
        self.stocks = stocks
        self.dictionary = {}
        self.stocks_name = stocks_name
        self.index_returns = index
        self.lmbda = lmbda
        self.param_mult = param_mult

        for i in range(len(self.stocks_name)):
            self.dictionary[self.stocks_name[i]] = self.stocks[self.stocks_name[i]].values

    def CalculPoids(self, chromo):
        ''' 
        chromo : np.array : chromosome of the individual

        Cette fonction calcule les poids des actions dans le portefeuille en fonction du chromosome de l'individu 

        '''
        assets = np.zeros(self.taille_portefeuille)
        weights = np.zeros(self.taille_portefeuille)

        for i in range(len(chromo.chromosome)):
            weights[i] = max(self.min_weight, chromo.chromosome[i])
            
            assets[i] = chromo.assets[i]

        weights = weights / np.sum(weights)
        

        return weights, assets





    def evaluation(self, chromo, evaluation_range, param_mult):
        '''
        chromo  : chromosome of the individual
        Cette fonction calcule le rendement du portefeuille en fonction du chromosome de l'individu

        '''
        
        weights, assets = self.CalculPoids(chromo)

        index_returns = self.index_returns

        shares = [0 for i in range(len(assets))]
        #currentweights = np.zeros(len(weights))

        for k in range(len(shares)):
            shares[k] = self.stocks_name[int(assets[k])]
        
            #currentweights[k] = weights[k]

        portfolioreturns = np.ones(evaluation_range[1])
        tracking_error = np.zeros(evaluation_range[1])
        # delweights = np.zeros(len(weights))

        for j in range(evaluation_range[0], evaluation_range[1]):
            
            current_periodreturn = 0
            for k in range(len(shares)):
                
                try:
                    # prix de l'action k le jour j à partir de la dataframe stocks
                   
                    value = self.dictionary[shares[k]][j-1]
                    #value = self.stocks.iat[j-1, k-1]

                    current_periodreturn += value * weights[k]

                except:
                    current_periodreturn += 0.0*weights[k]

            portfolioreturns[j] = current_periodreturn

            tracking_error[j] = current_periodreturn - index_returns[j]



        portfolio_mean = np.mean(portfolioreturns)
        portfolio_std = np.std(portfolioreturns)
        sharp_ratio = portfolio_mean / portfolio_std * np.sqrt(252) # 252 = nombre de jours de trading par an

        truetracking_error = np.sqrt(np.var(tracking_error[evaluation_range[0]:evaluation_range[1]]))

        # fit = self.lmbda * truetracking_error + (1 - self.lmbda) * er
        
        fit =  truetracking_error  / 100 
        # fit = 0
        # fit =  self.lmbda * sharp_ratio

        return [fit, portfolioreturns, weights, assets]


