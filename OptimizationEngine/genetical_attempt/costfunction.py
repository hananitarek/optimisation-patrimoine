import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import chromosome as ch
sys.path.append(r"/home/tarek/Projet-spe/DS-DEMO")

import two_step_withmoredata as ts

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

        #sorted_chromo = np.sort(chromo) # On trie le chromosome pour pouvoir prendre les indices des valeurs les plus élevées
        #count = 0

        #for i in range(len(chromo)):
        #    if count >= self.taille_portefeuille:
        #        break
        #    if sorted_chromo[-self.taille_portefeuille] <= chromo[i]:
        #        # if the weight is greater than the minimum weight
        #        assets[count] = i
        #        
        #        # weights[count] = max(0.001, chromo[i] - (1 - self.taille_portefeuille / len(self.stocks))) # to avoid negative weights
        #        weights[count] = max(self.min_weight, chromo[i])
        #        # weights[count] = chromo[i]- (1 - self.taille_portefeuille / len(self.stocks)) 
        #        # if weights[count] < 0:
        #        #     weights[count] = 0
#
        #        count += 1

        for i in range(len(chromo.chromosome)):
            weights[i] = max(self.min_weight, chromo.chromosome[i])
            
            assets[i] = chromo.assets[i]

        
        # weights = (1 - self.taille_portefeuille * self.min_weight) * (weights / np.sum(weights)) # somme =  (1 - self.taille_portefeuille * self.min_weight)
        # weights += self.min_weight
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
#

            #for k in range(len(shares)):
            #    try:
            #        value = self.dictionary[shares[k]][j-1]
#
            #        currentweights[k] *= (value/current_periodreturn)
            #        delweights[k] = weights[k] - currentweights[k]
            #    except:
            #        delweights[k] = 0


            # TODO : gérer le coût de transaction, sum(abs(delweights)) est trop grand ! 
            #if np.mod(j- evaluation_range[0], self.periode) == 0:
            #    
            #    cost = sum(abs(delweights)) * self.cout_transaction
            #    current_periodreturn = current_periodreturn - cost 
            
            portfolioreturns[j] = current_periodreturn

            tracking_error[j] = current_periodreturn - index_returns[j]



        portfolio_mean = np.mean(portfolioreturns)
        portfolio_std = np.std(portfolioreturns)
        sharp_ratio = portfolio_mean / portfolio_std * np.sqrt(252) # 252 = nombre de jours de trading par an


        
        #truetracking_error = np.sqrt(
        #    np.sum(
        #        np.power(
        #            tracking_error[evaluation_range[0]:evaluation_range[1]],
        #            2
        #            )
        #        )/(evaluation_range[1]-evaluation_range[0])
        #    )

        truetracking_error = np.sqrt(np.var(tracking_error[evaluation_range[0]:evaluation_range[1]]))

        # fit = self.lmbda * truetracking_error + (1 - self.lmbda) * er
        
        fit =  truetracking_error  / 100 
        # fit = 0
        # fit =  self.lmbda * sharp_ratio

        return [fit, portfolioreturns, weights, assets]



# test 
if __name__ == "__main__":

    stocks, stocks_name = ts.get_stocks()

    stocks.dropna(axis=1, thresh=0.8*len(stocks), inplace=True)
    stocks.dropna(inplace = True)


    index = ts.get_index("MSFT")

    index.dropna(inplace = True)

    stocks_name = stocks.columns.values
    index_tracker = IndexTracker(15, 0.1, 245, 0.01, stocks_name, stocks, np.array(index['return']), lmbda = 1) 

    NumAssets = len(stocks_name)

    chromo = ch.Chromosome(NumAssets)

    weights, assets = index_tracker.CalculPoids(chromo.chromosome)



    # evaluation test 
    evaluation_range = [0, 1000]
    fit, portfolioreturns, weights, assets = index_tracker.evaluation(chromo.chromosome, evaluation_range)

    print(weights)
    print(chromo.chromosome)
