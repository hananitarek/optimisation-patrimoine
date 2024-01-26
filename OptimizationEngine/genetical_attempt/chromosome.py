import numpy as np
from random import sample 

class Chromosome:
    def __init__(self, numAssets, nb_stocks_available):
        self.numAssets = numAssets
        self.nb_stocks_available = nb_stocks_available
        
        self.assets = sample(range(nb_stocks_available), numAssets) # liste des actifs choisis
        self.weights = 0

        self.chromosome = np.random.rand(numAssets)
        self.to_replace = False

        self.fitness = -np.inf 
        self.validation_fitness = -np.inf
        self.test_fitness = -np.inf

        self.portefolio_prices = 0
        self.validation_prices = 0
        self.test_prices = 0

    def mutate(self, mutation_rate):
        for i in range(self.numAssets):
            if np.random.random() < mutation_rate:
               
                self.chromosome[i] = np.random.random()

    def clone(self):
        clone = Chromosome(self.numAssets, self.nb_stocks_available)
        
        for i in range(self.numAssets):
            clone.chromosome[i] = self.chromosome[i]
        
        if len(self.weights) > 0:
            clone.weights = np.zeros(len(self.weights))
            clone.assets = np.zeros(len(self.assets))

        for i in range(len(self.weights)):
            clone.weights[i] = self.weights[i]
            clone.assets[i] = self.assets[i]
        
        clone.to_replace = self.to_replace

        return clone
