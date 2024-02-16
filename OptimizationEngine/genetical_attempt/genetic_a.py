import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from os import path
import os
from tqdm import tqdm
from chromosome import Chromosome
from costfunction import IndexTracker
import yfinance as yf
from pandas_datareader import data as wb
from time import time as t
from Stock import Stock
from datetime import datetime
from dateutil.relativedelta import *

class Genetics:
    def __init__(self, population_size, numAssets, ga_type):
        self.population_size = population_size   # population_size is an integer : le nombre de portefeilles dans la population
        self.numAssets = numAssets               # numAssets is an integer : le nombre d'actions dans le dataset

        self.genes = [Chromosome(numAssets) for i in range(population_size)] # genes is a list of Chromosome objects
        
        self.fitted_genes = []
        self.unfitted_genes = []
        self.fittest_index = 0
        self.ga_type = ga_type  # ga_type is an object of the class GeneticAlgorithm
 
    def best_chromo(self):
        '''
        Returns the best chromosome in the population (the one with the lowest fitness)
        '''
        validation_scores = list(map(lambda x: x.validation_fitness, self.genes))
        self.fittest_gene = np.argmin(validation_scores) # the index of the best chromosome
        return self.genes[self.fittest_gene] # the best chromosome

    def get_population(self):
        self.tournament()
        self.crossover()
        self.mutate()


    def tournament(self):
        self.fittest_genes = []
        self.unfitted_genes = []

        # we set all the genes to be replaced
        # Initialisation
        for i in range(len(self.genes)):
            self.genes[i].to_replace = True
            self.unfitted_genes.append(self.genes[i])

        fitness_threshold = np.inf
        

        for i in range(len(self.genes)):
            if self.genes[i].fitness < fitness_threshold:
                fitness_threshold = self.genes[i].fitness
                self.fittest_index = i
        
        self.fittest_gene = self.fittest_index

        self.genes[self.fittest_index].to_replace = False # we set the fittest gene to not be replaced
        self.fittest_genes.append(self.genes[self.fittest_index]) # we add the fittest gene to the list of fittest genes
        self.unfitted_genes.remove(self.genes[self.fittest_index]) # we remove the fittest gene from the list of unfitted genes

        for i in range(int((1 - self.ga_type.replacement_rate) * self.population_size)):
            
            fitness_threshold = np.inf
            best = 0
            for j in range(self.ga_type.tournament_size):
                current_idx = int(np.random.random() * len(self.unfitted_genes))
                #self.unfitted_genes[current_idx]

                if self.unfitted_genes[current_idx].fitness < fitness_threshold:
                    best = current_idx
                    fitness_threshold = self.unfitted_genes[current_idx].fitness
            
            self.unfitted_genes[best].to_replace = False
            self.fittest_genes.append(self.unfitted_genes[best])
            self.unfitted_genes.remove(self.unfitted_genes[best])



    
    def mutate(self):
        for i in self.genes:
            if i.to_replace:
                i = self.fittest_genes[int(np.random.random() * len(self.fittest_genes))].clone()
                i.mutate(self.ga_type.prob_mutation)
    
    def crossover(self):

        for c1 in self.genes:
            if c1.to_replace:
                if np.random.random() < self.ga_type.prob_crossover:
                    pere = self.fittest_genes[int(np.random.random() * len(self.fittest_genes))]
                    mere = self.fittest_genes[int(np.random.random() * len(self.fittest_genes))]
                    fils = self.unfitted_genes[int(np.random.random() * len(self.unfitted_genes))]

                    split = int(np.random.random() * self.numAssets)

                    for j in range(self.numAssets):
                        fils.chromosome[j] = pere.chromosome[j]
                        if j < split:
                            fils.chromosome[j] = mere.chromosome[j]
        

                    







class GeneticAlgorithm:
    prob_crossover = None 
    prob_mutation = None
    replacement_rate = None
    tournament_size = None
    list_meilleures_fitness = []

    @classmethod
    def update_parameters(cls, prob_crossover, prob_mutation, replacement_rate, tournament_size):
        cls.prob_crossover = prob_crossover
        cls.prob_mutation = prob_mutation
        cls.replacement_rate = replacement_rate
        cls.tournament_size = tournament_size
    
    def __init__(self, numCycles, population_size, stocks, index, genetic_params, tracker_params):
        prob_crossover, prob_mutation, replacement_rate, tournament_size = genetic_params
        self.update_parameters(prob_crossover, prob_mutation, replacement_rate, tournament_size)
        taille_portefeuille, min_weight, periode, cout_transaction = tracker_params

        self.stocks = stocks
        self.numCycles = numCycles
        self.stocks_name = stocks_name
        self.index = index
        self.evaluateur = IndexTracker(taille_portefeuille, min_weight, periode, cout_transaction, stocks, index)
        self.Genetics = Genetics(population_size, len(stocks), self.__class__) # params : population_size, numAssets, ga_type
        self.list_meilleures_fitness = []

    def run(self, T_train, T_valid, T_test):
        
        for i in tqdm(range(self.numCycles)):
            if i > 0 :
                self.Genetics.get_population()
    
            for j in range(len(self.Genetics.genes)):
                scores = self.evaluateur.evaluation(self.Genetics.genes[j].chromosome, T_train)

                self.Genetics.genes[j].fitness = scores[0]
                self.Genetics.genes[j].portfolioreturns = scores[1]
                self.Genetics.genes[j].weights = scores[2]
                self.Genetics.genes[j].assets = scores[3]

                self.list_meilleures_fitness.append(scores[0])

        
        for k in range(len(self.Genetics.genes)):
            scores = self.evaluateur.evaluation(self.Genetics.genes[k].chromosome, T_valid)
            self.Genetics.genes[k].validation_fitness = scores[0]
            self.Genetics.genes[k].validation_portfolioreturns = scores[1]

            scores = self.evaluateur.evaluation(self.Genetics.genes[k].chromosome, T_test)
            self.Genetics.genes[k].test_fitness = scores[0]
            self.Genetics.genes[k].test_portfolioreturns = scores[1]


    def best_portfolio(self, portfolio_type = 'test'):
        
        best = self.Genetics.best_chromo()

        if portfolio_type == 'test':
            r = best.test_portfolioreturns

        if portfolio_type == 'validation':
            r = best.validation_portfolioreturns

        elif portfolio_type == 'train':
            r = best.portfolioreturns
        

        shares = [0 for i in range(len(best.assets))]

        for k in range(len(shares)):
            shares[k] = self.stocks[int(best.assets[k])].epic

        return [shares, best.weights, r]
    
    def solution(self):
        return self.Genetics.best_chromo()




# test 
if __name__ == "__main__":
    # we load the data
    FICHIER = 'stock_data.csv'
    chemin_complet = os.path.join('DataProvider', FICHIER)
    stocks = pd.read_csv(chemin_complet)
    stocks = stocks[['date', 'symbol', 'Close']]

    stocks['Adj Close'] = stocks['Close'] * 100


    drop = np.array([0])
    for i in tqdm(range(1, len(stocks)), "Traitement des données"):
        if stocks['symbol'][i] != stocks['symbol'][i-1]:
            drop = np.append(drop, i)

    stocks.drop(index = drop, inplace = True)

    stocks = stocks.copy().loc[:,['date','symbol','Adj Close']]
    stocks = stocks.pivot(index='date', columns='symbol', values='Adj Close')

    stocks.dropna(axis=1, thresh=0.8*len(stocks), inplace=True)
    stocks.dropna(inplace = True)


    yf.pdr_override()
    index = pd.DataFrame()
    to_date = datetime.today().date()
    from_date = to_date - relativedelta(years=10)
    index = wb.get_data_yahoo("^GSPC",start=from_date, end=to_date, interval='1d')

    # index['return'] = np.log(index['Close']) - np.log(index['Close'].shift())
    index['Adj Close'] = index['Close']
    
    data = index[['Adj Close']].copy()

    # rename the column
    data.columns = ['FTSE']
    stocks['FTSE'] = data['FTSE']
    stocks.dropna(inplace = True)

    
    index_returns = stocks['FTSE'].values

    # retirer la colonne FTSE du dataframe stocks
    stocks.drop(columns = ['FTSE'], inplace = True)

    stocks_name = stocks.columns.values

    list_stocks = {}
    for i in range(len(stocks_name)):
        stock = Stock(stocks_name[i], stocks[stocks_name[i]])
        list_stocks[i] = stock

    index = Stock('FTSE', data['FTSE'])

    # we define the parameters
    numCycles = 10
    population_size = 10
    genetic_params = [0.95, 0.075,0.5, 2] # prob_crossover, prob_mutation, replacement_rate, tournament_size
    tracker_params = [len(stocks_name), 0.0, 10, 0.01] # taille_portefeuille, min_weight, periode, cout_transaction

    # we create the genetic algorithm



    ga = GeneticAlgorithm(numCycles, population_size, list_stocks, index, genetic_params, tracker_params)

    # we run the genetic algorithm
    Nb_days = len(index_returns)
    T_train = [1, int(Nb_days * 0.7)]
    T_valid = [int(Nb_days * 0.7), int(Nb_days * 0.85)]
    T_test = [int(Nb_days * 0.85), Nb_days]


    ga.run(T_train, T_valid, T_test) # T_train, T_valid, T_test

    liste_fitness = ga.list_meilleures_fitness

    plt.plot(liste_fitness)
    plt.show()

    # we get the best portfolio
    best_portfolio = ga.best_portfolio()

    weights = dict(zip(best_portfolio[0], best_portfolio[1]))
    weight_vector = list(map(lambda X : weights[X], sorted(weights)))

    # we print the tracking error (fitness if lambda = 1)
    print(ga.solution().fitness)

    # we print the best portfolio

    print(weight_vector)


    # plot the returns of the solution

    # extract from stocks a new dataframe with only the stocks in the portfolio
    fdata = stocks[best_portfolio[0]]

    print(fdata.head())

    portfolioreturns = np.dot(
        fdata.sort_values(by = 'date'),
        weight_vector
    )


    # we subtract the mean and divide by the standard deviation because we want to compare the returns of the portfolio to the returns of the index
    # this comparison is not possible if the returns are not on the same scale (mean and standard deviation) 
    nreturns = (( portfolioreturns - portfolioreturns.mean() ) / (portfolioreturns.max() -  portfolioreturns.min())) 
    nindex = ((index_returns - index_returns.mean()) / (index_returns.max() - index_returns.min())) 

    plt.plot(nreturns, label = "portfolio")
    plt.plot(nindex, label = "index")

   # plot a vertical line at the end of the training period
    plt.axvline(x = T_train[1], color = 'r', linestyle = '--')

    # plot a vertical line at the end of the validation period
    plt.axvline(x = T_valid[1], color = 'g', linestyle = '--')

    # plot a vertical line at the end of the test period
    plt.axvline(x = T_test[1], color = 'b', linestyle = '--')



    plt.legend()
    plt.show()


   