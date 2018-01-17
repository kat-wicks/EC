#! /usr/bin/env python

import time
import random
import math
import copy
import itertools
import argparse

import tsp


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

Change the fitness function ot apply the algorithm to a different problem.

.. codeauthor:: Erik Hemberg <hembergerik@csail.mit.edu>

  The evolutionary algorithm, performs a *stochastic parallel
  iterative* search. The algorithm:

  - Generate a population of *initial solutions*
  - *Iterate* a fixed number of times 

    - *Evaluate* the fitness of the new solutions
    - *Select* solutions for a new population
    - *Vary* the solutions in the new population
  
      - *Mutate* a solution

      - *Crossover* two solutions

    - *Replace* the old population with the new population

  The data fields are:

  - Individual, a dictionary:

    - Genome, an integer list for representing a bitstring
    - Fitness, an integer for the fitness value

See e.g.  Annals of Operations Research 63(1996)339-370 339
Genetic algorithms for the traveling salesman problem for more on TSP

"""


DEFAULT_FITNESS = float("inf")
VERBOSE = False

def map_genome(genome):
    return genome

def ea(population_size, max_size, generations,
       mutation_probability, crossover_probability,
       tournament_size, elite_size, tsp_data):
    """
    Evolutionary search
    loop. Starting from the initial population.
    """

    #Create population
    city_data = tsp.parse_city_data(tsp_data)
    number_of_cities = len(city_data)
    base_tour = list(range(0, number_of_cities))

    # Initial population
    population = []
    for i in range(population_size):
        genome = base_tour[:]
        random.shuffle(genome)
        phenotype = map_genome(genome)
        individual = {'genome': genome, 'fitness': DEFAULT_FITNESS, 'phenotype': phenotype}
        population.append(individual)
        if VERBOSE:
            print('Initial {}: {}'.format(individual['genome'], individual['fitness']))

    # Evaluate fitness
    for ind in population:
        ind['fitness'] = tsp.get_tour_cost(ind['genome'], city_data)

    #Generation loop
    generation = 0
    while generation < generations:

        # Selection
        new_population = []
        while len(new_population) < population_size:
            # Randomly select tournament size individual solutions
            # from the population.
            competitors = random.sample(population, tournament_size)
            # Rank the selected solutions
            sort_population(competitors)
            # Append the best solution to the winners
            new_population.append(copy.deepcopy(competitors[0]))

        # Vary the population by crossover
        for ind in new_population:
            if crossover_probability < random.random():
                parents = random.sample(new_population, 2)
                ind = modified_onepoint_crossover(*parents)
        
        # Vary the population by mutation
        for ind in new_population:
            if mutation_probability < random.random():
                ind = swap_mutation(individual)
                
        # Evaluate fitness
        for ind in new_population:
            ind['fitness'] = tsp.get_tour_cost(ind['genome'], city_data)

        # Replace population
        sort_population(population)
        # Add elites
        population = population[:elite_size] + new_population
        sort_population(population)
        # Get back to population size
        population = population[:population_size]

        # Print the stats of the population
        print_stats(generation, population)

        # Increase the generation counter
        generation += 1

    return population[0]


def sort_population(population):
    population.sort(reverse=False, key=lambda x:x['fitness'])    

def modified_onepoint_crossover(parent_one, parent_two):
    """Given two individuals, create one child using one-point
    crossover and return.

    A cut position is chosen at random on the first parent
    chromosome. Then, an offspring is created by appending the second
    parent chromosome to the initial segment of the first parent
    (before the cut point), and by eliminating the duplicates.

    :param Individual: 
    :param parent_one: A parent
    :type parent_one: dict
    :param parent_two: Another parent
    :type parent_two: dict
    :return: A child
    :rtype: dict

    """
    child = {'genome': None, 'fitness': DEFAULT_FITNESS}

    point = random.randint(0, len(parent_one['genome']))
    # Get temporary genome concatenate
    _genome = parent_one['genome'][:point] + parent_two['genome'][:]
    # Do not use duplicates
    genome = []
    for gene in _genome:
        if gene not in genome:
            genome.append(gene)
                
    child['genome'] = genome
    
    return child


def swap_mutation(individual):
    """Mutate the individual by random swap.

    :param individual:
    :type individual: Individual
    :return: Mutated individual
    :rtype: dict

    """

    genome_length = len(individual['genome']) - 1
    point_one = random.randint(0, genome_length)
    point_two = random.randint(0, genome_length)
    value_one = individual['genome'][point_one]
    value_two = individual['genome'][point_two]
    # Swap
    individual['genome'][point_one] = value_two
    individual['genome'][point_two] = value_one
    # Reset fitness
    individual['fitness'] = DEFAULT_FITNESS

    return individual


def print_stats(generation, population):
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
        :type values: list of numbers
        :returns: Average and Standard deviation of the input values
        :rtype: Tuple of floats
        """
        _ave = float(sum(values)) / len(values)
        _std = math.sqrt(float(
            sum((value - _ave) ** 2 for value in values)) / len(values))
        return _ave, _std

    # Sort population
    population.sort(reverse=True, key=lambda x:x['fitness'])    
    # Get the fitness values
    fitness_values = [i['fitness'] for i in population]
    # Calculate average and standard deviation of the fitness in
    # the population
    ave_fit, std_fit = get_ave_and_std(fitness_values)
    # Print the statistics, including the best solution
    print("Gen:{} fit_ave:{:.2f}+-{:.3f} {} {}".format(generation, ave_fit, std_fit,
           population[0]['genome'], population[0]['fitness']))


def main():
    """
    Parse the command line arguments. Create the fitness
    function and the Evolutionary Algorithm. Run the
    search.
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--psize", type=int, default=30,
                        help="population size")
    parser.add_argument("-m", "--maxsize", type=int, default=5,
                        help="individual size")
    parser.add_argument("-g", "--generations", type=int, default=5,
                        help="number of generations")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="seed number")
    parser.add_argument("-cp", "--crossover_probability", type=float, default=0.8,
                        help="crossover probability")
    parser.add_argument("-mp", "--mutation_probability", type=float, default=0.2,
                        help="mutation probability")
    parser.add_argument("-t", "--tournament_size", type=int, default=2,
                        help="tournament size")
    parser.add_argument("--elite_size", type=int, default=1,
                        help="elite size")
    parser.add_argument("--tsp_data", type=str, default='tsp_costs.csv',
                        help="Data for Travelling Salesman problem.")
    parser.add_argument("--tsp_exhaustive", action='store_true',
                        help="Data for Travelling Salesman problem.")
    parser.add_argument("--verbose", action='store_true',
                        help="Verbose")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    VERBOSE = args.verbose
    if VERBOSE:
        print(args)
    
    start_time = time.time()
    best_solution = ea(args.psize, args.maxsize, args.generations,
                       args.mutation_probability, args.crossover_probability,
                       args.tournament_size, args.elite_size, args.tsp_data)
    execution_time = time.time() - start_time
    print("EA:\n Minimal tour cost is {} for path {}. Searched {} points in {:.5f} seconds".format( best_solution['fitness'], best_solution['genome'], args.psize * args.generations, execution_time))


    if args.tsp_exhaustive:
        city_data = tsp.parse_city_data(args.tsp_data)
        base_tour = range(0, len(city_data))
        min_tour = {'cost':float("inf"), 'tour': None}
        start_time = time.time()
        for tour in itertools.permutations(base_tour):
            cost = tsp.get_tour_cost(list(tour), city_data)
            if cost < min_tour['cost']:
                min_tour['cost'] = cost
                min_tour['tour'] = tour

        execution_time = time.time() - start_time
        print("EXHAUSTIVE:\n A minimal tour cost is {} for path {}. Searched {} points in {:.5f} seconds".format( min_tour['cost'], min_tour['tour'], math.factorial(len(city_data)), execution_time))


if __name__ == '__main__':
    main()

