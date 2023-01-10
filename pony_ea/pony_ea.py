#! /usr/bin/env python

import time
import random
import math
import copy
import itertools
import argparse
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
#import tsp
import defender
import attacker
import numpy as np
import pandas as pd

# The MIT License (MIT)

# Copyright (c) 2013, 2018 ALFA Group, Erik Hemberg

# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Evolutionary Algorithm

The source code design is for teaching the concept of how evolution
inspires computational intelligence, not for fast portable use. 

This version finds solutions for the Travelling Salesman Problem.
For more on TSP see e.g. 
Annals of Operations Research 63(1996)339-370 339 Genetic algorithms for the traveling salesman problem
"""

__author__ = "Erik Hemberg"

DEFAULT_FITNESS = -float("inf")
VERBOSE = False

train_pca_scores = {} # labels = n_components, 

def map_genome(genome, X, is_defender=True, y=np.empty(0)):
    """
    Map a genome to a phenotype (input to output) with identity.

    :param genome: List of node(city) to visit in order
    :type genome: list of integers
    :return: List of node(city) to visit in order
    :rtype: list of integers
    """
    if(is_defender and not (y.size ==0)):
        pca =PCA(n_components = int(genome[0]))
        pca_scores = pca.fit_transform(X)
        knn = KNeighborsClassifier(n_neighbors = int(genome[1])).fit(pca_scores,y)
        return pca, knn
    
    elif(not is_defender):
        mask = np.ones(X.shape[0], dtype=bool)
        mask[genome] = False
        return mask

    else:
        raise ValueError("need to have y to create defender phenotype")



def evolutionary_algorithm(population_size, generations, 
                           mutation_probability, crossover_probability,
                           tournament_size, elite_size, data):
    """
  The evolutionary algorithm (EA), performs a *stochastic parallel
  iterative* search. The algorithm:

  - Generate a population of *initial solutions*
  - *Iterate* a fixed number of times

    - *Evaluate* the fitness of the new solutions
    - *Select* solutions for a new population
    - *Vary* the solutions in the new population

      - *Mutate* a solution

      - *Crossover* two solutions

    - *Replace* the old population with the new population keeping the "elite" solutions

  The data fields are:

  - Individual, a dictionary:

    - Genome, an integer list
    - Fitness, an integer for the fitness value
    - Phenotype, a representation of a TSP tour


    :param population_size:
    :param generations:
    :param mutation_probability:
    :param crossover_probability:
    :param tournament_size:
    :param elite_size:
    :param tsp_data:
    :return:
    """
    ##########
    # Create problem
    ##########
    # Parse the  data to a cost matrix
    train_X, test_X , train_y, test_y = defender.parse_data(data)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    ##########
    # Initialize defender population
    ##########
    population = []
    for i in range(population_size):
        _genome =[0,0]
        # EH: any reason for picking the constants? I.e. the number of PCA dimensions and the neighbors.
        # Another way for the PCA could be to pick a threshold for the weights that the number of dimensions must be over when you sum them.
        _genome[0] = max(1, min(train_X.shape[1], int(np.random.normal(train_X.shape[1]*.2, train_X.shape[1]*.1))))
        _genome[1] = max(1, min(train_X.shape[0], int(np.random.normal(train_X.shape[0]*.05, train_X.shape[0]*.025))))
        solution = {'genome': _genome,
                    'fitness': DEFAULT_FITNESS,
                    'phenotype': map_genome(_genome, train_X, is_defender = True, y=train_y)}
        population.append(solution)
        if VERBOSE:
            print('Initial {}: {}'.format(solution['genome'], solution['fitness']))

    ##########
    # Evaluate defender fitness
    ##########
    for solution in population:
        solution['fitness'] = defender.evaluate_solution(solution['phenotype'], train_X, train_y)

    #########
    # Initialize attacker population
    ##########
    atk_population = []
    rng = np.random.default_rng()
    for i in range(population_size):
        _genome = rng.choice(train_X.shape[0],int(train_X.shape[0]*.05), replace= False)

        solution = {'genome': _genome,
                    'fitness': DEFAULT_FITNESS,
                    'phenotype': map_genome(_genome, train_X, is_defender=False)}
        atk_population.append(solution)
        if VERBOSE:
            print('Initial {}: {}'.format(solution['genome'], solution['fitness']))

    ##########
    # Evaluate attacker fitness
    ##########
    for solution in atk_population:
        # EH: Why not evaluate against all defenders? Here attacker is only evaluated against one defender?
        # It gets a bit messier against multiple adversaries since then you have to pick a "solution concept", e.g. mean expected utility or best-worst case
        fitnesses_ = []
        for defender_ in population:
            fitness_ = attacker.evaluate_solution(solution['phenotype'], defender_['phenotype'], train_X, train_y)
            fitnesses_.append(fitness_)
        
        # Here we use mean expected utility
        solution['fitness'] = np.mean(fitnesses_)

    ##########
    # Generation loop
    ##########
    generation = 0
    while generation < generations:
        ##########
        # Select fit solutions
        ##########
        new_population = []
        while len(new_population) < population_size:
            # Randomly select tournament size solutions
            # from the population.
            competitors = random.sample(population, tournament_size)
            # Rank the selected solutions in a competition
            sort_population(competitors)
            # Append the winner of the competition to the new population
            new_population.append(copy.deepcopy(competitors[0]))

        ##########
        # Vary the population by crossover
        ##########
        for i, solution in enumerate(new_population):
            if crossover_probability > random.random():
                # Select a mate for crossover
                mate = random.sample(new_population, 1)
                # Put the child in the population
                new_population[i] = onepoint_crossover(solution, *mate)

        ##########
        # Vary the population by mutation
        # ##########
        for i, solution in enumerate(new_population):
            if mutation_probability > random.random():
                # Mutate genes by adding noise to them
                new_population[i] = noise_mutation(solution, train_X, is_defender=True)


        ##########
        # Evaluate fitness
        ##########
        for solution in new_population:
            fitnesses_ = []
            for attack_solution in atk_population:
                frozen_X = train_X[attack_solution['phenotype']]
                frozen_y = train_y[attack_solution['phenotype']]
                solution['phenotype'] = map_genome(solution['genome'], frozen_X, is_defender=True, y=frozen_y)
                fitness_ = defender.evaluate_solution(solution['phenotype'], train_X, train_y)
                fitnesses_.append(fitness_)
               
            solution['fitness'] = np.mean(fitnesses_)
            
        ##########
        # Replace population
        ##########
        sort_population(population)
        # Add best(elite) solutions from old population
        population = population[:elite_size] + new_population
        sort_population(population)
        # Trim back to population size
        population = population[:population_size]

        # Print the stats of the population
        print_stats(generation, population, train_X, train_y)

        '''
        #############################################
        Repeat process for attacker
        #############################################
        '''
        ##########
        # Select fit solutions
        ##########
        new_population = []
        while len(new_population) < population_size:
            # Randomly select tournament size solutions
            # from the population.
            competitors = random.sample(atk_population, tournament_size)
            # Rank the selected solutions in a competition
            sort_population(competitors)
            # Append the winner of the competition to the new population
            new_population.append(copy.deepcopy(competitors[0]))

        ##########
        # Vary the population by crossover
        ##########
        for i, solution in enumerate(new_population):
            if crossover_probability > random.random():
                # Select a mate for crossover
                mate = random.sample(new_population, 1)
                # Put the child in the population
                new_population[i] = onepoint_crossover(parent_one=solution, parent_two=mate[0])

        ##########
        # Vary the population by noise
        # ##########

        for i, solution in enumerate(new_population):
            if mutation_probability > random.random():
                # Mutate genes by adding noise to them
                new_population[i] = noise_mutation(solution, train_X, is_defender=False)


        ##########
        # Evaluate fitness
        ##########
        # EH: Can you not save some mappings and only map the solutions in the new population before you evaluate the solutions?
        for solution in new_population:
            fitnesses_ = []                    
            solution['phenotype'] = map_genome(solution['genome'], train_X, is_defender=False, y=train_y)        
            for defender_solution in population:
                frozen_defender = defender_solution['phenotype']
                fitness_ = attacker.evaluate_solution(solution['phenotype'], frozen_defender, train_X, train_y)
                fitnesses_.append(fitness_)
                
            solution['fitness'] = np.mean(fitnesses_)
            
        ##########
        # Replace population
        ##########
        sort_population(atk_population)
        # Add best(elite) solutions from old population
        atk_population = atk_population[:elite_size] + new_population
        sort_population(atk_population)
        # Trim back to population size
        atk_population = atk_population[:population_size]

        # Print the stats of the population
        print_stats(generation, atk_population,train_X, train_y,is_defender = False, best_defender = population[0]['phenotype'])
        # Increase the generation counter
        generation += 1

    return population[0]


def sort_population(population):
    """
    Sort population by fitness.

    Decreasing fitness order.

    :param population:
    :return: sorted population
    """
    population.sort(reverse=True, key=lambda x: x['fitness'])


def onepoint_crossover(parent_one, parent_two):
    """Given two individuals, create one child using one-point
    crossover and return.

    A cut position is chosen at random on the first parent
    chromosome. Then, an offspring is created by selecting randomly from 
    the first or second genorme to include 
    :param parent_one: A parent
    :type parent_one: dict
    :param parent_two: Another parent
    :type parent_two: dict
    :return: A child
    :rtype: dict

    """
    child = {'genome': [], 'fitness': DEFAULT_FITNESS, 'phenotype': None}
    #print(parent_one['genome'], parent_two)
    # Pick a point for crossover
    point = random.randint(0, len(parent_one['genome']))
    # # Get temporary genome concatenate
    # try:
    #     print(parent_one['genome'].shape, parent_two['genome'].shape)
    # except:
    #     print(len(parent_one['genome']), len(parent_two['genome']))
    if random.random() >0.5:
        child['genome'] = np.append(parent_one['genome'][:point],parent_two['genome'][point:])
    else: 
        child['genome'] = np.append(parent_two['genome'][:point],parent_one['genome'][point:])
    return child


def noise_mutation(solution, X, is_defender):
    """Mutate the solution by adding noise

    :param solution:
    :type solution: dict
    :return: Mutated solution
    :rtype: dict

    """

    # Pick points for noising
    if(is_defender):
        solution['genome'][0] =  max(1,min(X.shape[1]-1,int(np.random.default_rng().normal(solution['genome'][0], solution['genome'][0]*.05))))
        solution['genome'][1] = max(1,min(X.shape[0]-1, int(np.random.default_rng().normal(solution['genome'][1], solution['genome'][1]*.05))))
        solution['fitness'] = DEFAULT_FITNESS

    else: 
        mu = np.mean(solution['genome'])
        solution['genome'] = np.random.default_rng().normal(loc=mu, scale=mu*.05, size =solution['genome'].shape)
        solution['genome'] = solution['genome'].astype('int32')
        solution['genome'] = np.clip(solution['genome'], 0, X.shape[0]) 
        solution['fitness'] = DEFAULT_FITNESS

    return solution


def print_stats(generation, population,X=np.empty(0),y=np.empty(0), is_defender = True, best_defender = None):
    """
    Print the statistics for the generation and population.

    :param generation:generation number
    :type generation: integer
    :param population: population to get statistics for
    :type population: list of population
    """

    def get_ave_and_std(values):
        """
        Return average and standard deviation.            

        :param values: Values to calculate on
        :type values: list
        :returns: Average and Standard deviation of the input values
        :rtype: Tuple of floats
        """
        _ave = float(sum(values)) / len(values)
        _std = math.sqrt(float(
            sum((value - _ave) ** 2 for value in values)) / len(values))
        return _ave, _std

    # Get the fitness values
    fitness_values = [i['fitness'] for i in population]
    # Calculate average and standard deviation of the fitness in
    # the population
    ave_fit, std_fit = get_ave_and_std(fitness_values)
    # Print the statistics, including the best solution
    if(is_defender):
        print("Gen:{}; Defender Population fitness mean:{:.2f}+-{:.3f}; Median:{:.4f}, Best solution:{}, fitness:{}, test fitness:{}".format(
            generation, ave_fit, std_fit,np.median(fitness_values), population[0]['genome'], population[0]['fitness'], defender.evaluate_solution(population[0]['phenotype'], X, y)))
    else:
        print("Gen:{}; Attacker Population fitness mean:{:.2f}+-{:.3f}; Median:{:.4f}, Best solution:{}. fitness:{}, test fitness:{}".format(
            generation, ave_fit, std_fit,np.median(fitness_values),  population[0]['genome'], population[0]['fitness'], attacker.evaluate_solution(population[0]['phenotype'], best_defender,X, y)))

def main():
    """
    Parse the command line arguments. Create the fitness
    function and the Evolutionary Algorithm. Run the
    search.
    """

    ###########
    # Parse arguments
    ###########
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--population_size", type=int, default=20,
                        help="Population size")
    parser.add_argument("-g", "--generations", type=int, default=5,
                        help="number of generations")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Random seed number")
    parser.add_argument("--crossover_probability", type=float, default=0.8,
                        help="Crossover probability")
    parser.add_argument("--mutation_probability", type=float, default=0.5,
                        help="Mutation probability")
    parser.add_argument("--tournament_size", type=int, default=2,
                        help="Tournament size")
    parser.add_argument("--elite_size", type=int, default=2,
                        help="Elite size")
    parser.add_argument("--tsp_data", type=str, default=r'C:\Users\mitadm\Downloads\triple (1)\data2.csv',
                        help="Data for ICS problem in a CSV file.")
    parser.add_argument("--verbose", action='store_true',
                        help="Verbose mode")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    global VERBOSE
    VERBOSE = args.verbose
    if VERBOSE:
        print(args)

    ###########
    # Evolutionary Algorithm search
    ###########
    start_time = time.time()
    best_solution = evolutionary_algorithm(args.population_size, args.generations,
                                           args.mutation_probability, args.crossover_probability,
                                           args.tournament_size, args.elite_size, args.tsp_data)
    execution_time = time.time() - start_time
    print("EA:\n Best tour cost is {} for path {}. Searched {} points in {:.5f} seconds".format(
        best_solution['fitness'], best_solution['genome'], args.population_size * args.generations, execution_time))

  
if __name__ == '__main__':
    main()
