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
    
    def __init__(self, taille_portefeuille, min_weight, rebalancing_period, transaction_cost, stocks, index):
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
        self.rebalancing_period = rebalancing_period
        self.transaction_cost = transaction_cost
        self.stocks = stocks
        self.index = index

    def CalculPoids(self, chromo):
        ''' 
        chromo : np.array : chromosome of the individual

        Cette fonction calcule les poids des actions dans le portefeuille en fonction du chromosome de l'individu 

        '''
        assets = np.zeros(self.taille_portefeuille)
        weights = np.zeros(self.taille_portefeuille)
        sorted_genome = np.sort(chromo)
        count = 0 # compteur pour les poids nuls
        for i in range(len(chromo)):
            # weights[i] = max(self.min_weight, chromo.chromosome[i])
            
            # assets[i] = chromo.assets[i]
            if count >= self.taille_portefeuille:
                break
            if sorted_genome[-self.taille_portefeuille] <= chromo[i]:
                # si le poids le plus faible est supérieur à la valeur du chromosome
                assets[count] = i
                weights[count] = max(0.001,
                                     chromo[i] - (1 - self.taille_portefeuille / len(self.stocks))
                                     )
            
                count += 1

        weights = (1 - self.taille_portefeuille * self.min_weight) * weights / np.sum(weights)
        weights += self.min_weight
        # weights = weights / np.sum(weights)
        

        return weights, assets





    def evaluation(self, chromo, evaluation_range):
        '''
        chromo  : chromosome of the individual
        Cette fonction calcule le rendement du portefeuille en fonction du chromosome de l'individu

        '''
        
        weights, assets = self.CalculPoids(chromo)

        index_returns = self.index.prices

        shares = [0 for _ in range(len(assets))]
        currentweights = np.zeros(len(weights))

        for k in range(len(shares)):
            shares[k] = self.stocks[int(assets[k])]
        
            currentweights[k] = weights[k]

        portfolioreturns = np.ones(evaluation_range[1])
        tracking_error = np.zeros(evaluation_range[1])
        delweights = np.zeros(len(weights))

        for j in range(evaluation_range[0], evaluation_range[1]):
            
            current_periodreturn = 0
            for k in range(len(shares)):
                
                try:
                    # prix de l'action k le jour j à partir de la dataframe stocks
                    value = self.dictionary[shares[k]][j-1]

                    current_periodreturn += value * currentweights[k]

                except:
                    current_periodreturn += 0.0*weights[k]

            for k in range(len(shares)):
                try:
                    currentweights[k] *= (self.dictionary[shares[k]][j-1] / current_periodreturn)
                    delweights[k] = weights[k] - currentweights[k]
                except:
                    delweights[k] = 0


            if np.mod(j - evaluation_range[0], self.rebalancing_period) == 0:
                cost = sum(
                    abs(delweights)
                ) * self.transaction_cost
                current_periodreturn = current_periodreturn - cost

            portfolioreturns[j] = current_periodreturn

            tracking_error[j] = current_periodreturn - index_returns[j]



        # portfolio_mean = np.mean(portfolioreturns)
        # portfolio_std = np.std(portfolioreturns)
        # sharp_ratio = portfolio_mean / portfolio_std * np.sqrt(252) # 252 = nombre de jours de trading par an

        truetracking_error = np.sqrt(np.sum(np.power(tracking_error[evaluation_range[0]:evaluation_range[1]],2))/(evaluation_range[1]-evaluation_range[0]))
        # fit = self.lmbda * truetracking_error + (1 - self.lmbda) * er
        
        fit =  truetracking_error / np.sqrt(252)
        # fit = 0
        # fit =  self.lmbda * sharp_ratio

        return [fit, portfolioreturns, weights, assets]


