import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from datetime import datetime
from dateutil.relativedelta import *
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
parentDirectoryPath = path.dirname (path.dirname(path.abspath(__file__)))


from tqdm import tqdm
from chromosome import Chromosome
from costfunction import IndexTracker
import yfinance as yf
from pandas_datareader import data as wb
from time import time as t


# Define the fitness function to maximize the Sharpe Ratio
def fitness_function(weights, data):
    data_returns = np.log(data) - np.log(data.shift(1))
    data_returns = data_returns.dropna()
    
    portfolio_returns = np.dot(data_returns, weights)
    portfolio_mean = np.mean(portfolio_returns)
    portfolio_std = np.std(portfolio_returns)
    sharpe_ratio = (portfolio_mean / portfolio_std) * np.sqrt(252)
    return sharpe_ratio




# Define the genetic algorithm
def genetic_algorithm(data, population_size=200, num_generations=50, mutation_rate=0.05, elitism=0.1):
    # Initialize the population
    population = np.random.rand(population_size, len(data.columns))
    population = population / np.sum(population, axis=1)[:, np.newaxis]

    # Calculate fitness of initial population
    fitness = np.array([fitness_function(individual, data) for individual in population])
    
    for generation in tqdm(range(num_generations)):
        # Sort the population by fitness
        sorted_idx = np.argsort(fitness)[::-1]
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Determine number of individuals to carry over via elitism
        num_elites = int(elitism * population_size)
        
        # Create the next generation, starting with the elites
        offspring = population[:num_elites]
        
        # Select parents for reproduction using tournament selection
        parent1_idx = np.random.randint(num_elites, population_size, size=population_size-num_elites)
        parent2_idx = np.random.randint(num_elites, population_size, size=population_size-num_elites)
        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]

        # Perform crossover and mutation to create the next generation
        crossover_prob = np.random.rand(population_size-num_elites, len(data.columns))
        crossover_mask = crossover_prob <= 0.5
        offspring_crossover = np.where(crossover_mask, parent1, parent2)

        mutation_prob = np.random.rand(population_size-num_elites, len(data.columns))
        mutation_mask = mutation_prob <= mutation_rate
        mutation_values = np.random.rand(population_size-num_elites, len(data.columns))
        mutation_direction = np.random.choice([-1, 1], size=(population_size-num_elites, len(data.columns)))
        offspring_mutation = np.where(mutation_mask, offspring_crossover + mutation_direction * mutation_values, offspring_crossover)

        # Ensure the offspring population has valid weights
        offspring_mutation = offspring_mutation / np.sum(offspring_mutation, axis=1)[:, np.newaxis]

        # Combine elites and offspring to create the next generation
        population = np.vstack((population[:num_elites], offspring_mutation))
        
        # Calculate fitness of new population
        fitness = np.array([fitness_function(individual, data) for individual in population])
        
    # Return the best individual from the final population
    best_idx = np.argmax(fitness)
    best_individual = population[best_idx]
    print('Best Sharpe Ratio: ', np.max(fitness)) 

    return best_individual



if __name__ == "__main__":
    FICHIER = 'stock_data_french.csv'
    chemin_complet = os.path.join('DataProvider', FICHIER)
    stocks = pd.read_csv(chemin_complet)
    stocks = stocks[['date', 'symbol', 'close']]
    stocks['return'] = stocks['Close'] * 100

    drop = np.array([0])

    for i in tqdm(range(1, len(stocks)), "Traitement des données"):
        if stocks['symbol'][i] != stocks['symbol'][i-1]:
            drop = np.append(drop, i)

    stocks.drop(index = drop, inplace = True)
    stocks = stocks.copy().loc[:,['date','symbol','Close']]
    stocks = stocks.pivot(index='date', columns='symbol', values='Close')





    yf.pdr_override()
    index = pd.DataFrame()
    to_date = datetime.today().date()
    from_date = to_date - relativedelta(years=10)
    index = wb.get_data_yahoo("^FTSE",start=from_date, end=to_date, interval='1d')

    index['return'] = index['Close']
    
    #Split the data into train and test sets
    train_data = stocks.iloc[:int(len(stocks)*0.8)]
    test_data = stocks.iloc[int(len(stocks)*0.8):]

    

    weights = genetic_algorithm(data = stocks, population_size=50, num_generations=10, mutation_rate=0.01, elitism=0.01)

    weights[weights < 0] = 0

    weights = weights / np.sum(weights)

    portfolio_returns = np.dot(
        test_data.pct_change().dropna(),
        weights
    )

    portfolio_cum_returns = np.cumprod(portfolio_returns + 1)

    benchmark_returns = index.iloc[int(len(index)*0.8):].pct_change().dropna()
    benchmark_cum_returns = np.cumprod(benchmark_returns + 1)


    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(portfolio_cum_returns, label='Portfolio')
    ax.plot(benchmark_cum_returns, label='Benchmark')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.set_title('Backtesting Results')
    ax.legend()
    plt.show()