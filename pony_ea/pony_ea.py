#! /usr/bin/env python

# The MIT License (MIT)

# Copyright (c) 2013 Erik Hemberg

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

Evolutionary Algorithm Description
----------------------------------

Components
~~~~~~~~~~

**EA** 
  The evolutionary algorithm, performs a *stochastic parallel
  iterative* search. The algorithm:

  - Generate a population of *initial solutions*
  - *Iterate* a fixed number of times 

    - *Evaluate* the fitness of the new solutions
    - *Select* solutions for a new population
    - *Vary* the solutions in the new population
  
      - *Mutate* a solution

    - *Replace* the old population with the new population

  The data fields are:

  - Individual, a dictionary:

    - Genome, an integer list for representing a bitstring
    - Fitness, an integer for the fitness value

  - Population, a list of individuals
  - Population size, integer for population size
  - Max size, the maximum size of an individual
  - Generations, the number of generations 
  - Random number generation seed, integer
  - Mutation probability, float

Running **EA**
----------------------------

::

  usage: pony_ea.py [-h] [-p PSIZE] [-m MAXSIZE] [-g GENERATIONS] [-s SEED]
                  [-mp MUTATION]

  optional arguments:
    -h, --help            show this help message and exit
    -p PSIZE, --psize PSIZE
                          population size
    -m MAXSIZE, --maxsize MAXSIZE
                          individual size
    -g GENERATIONS, --generations GENERATIONS
                          number of generations
    -s SEED, --seed SEED  seed number
    -mp MUTATION, --mutation MUTATION
                          mutation probability

"""

import random
import math
import copy

import argparse

DEFAULT_FITNESS = -1000

def ea(population_size, max_size, generations,
       mutation_probability, tournament_size):
    """
    Evolutionary search
    loop. Starting from the initial population.
    """

    #Create population
    population = []
    for i in range(population_size):
        genome = [random.randint(0, 1) for _ in range(max_size)]
        individual = {'genome': genome, 'fitness': DEFAULT_FITNESS}
        population.append(individual)
        print('Initial {}: {}'.format(individual['genome'], individual['fitness']))

    # Evaluate fitness
    for ind in population:
        ind['fitness'] = sum(ind['genome'])

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
            competitors.sort(reverse=True, key=lambda x:x['fitness'])
            # Append the best solution to the winners
            new_population.append(copy.deepcopy(competitors[0]))

        # Vary the population by mutation
        for ind in new_population:
            if random.random() < mutation_probability:
                #Pick gene
                idx = random.randint(0, len(ind['genome']) - 1)
                #Flip it
                ind['genome'][idx] = (ind['genome'][idx] + 1) % 2

        # Evaluate fitness
        for ind in new_population:
            ind['fitness'] = sum(ind['genome'])

        # Replace population
        population = new_population

        # Print the stats of the population
        print_stats(generation, population)

        # Increase the generation counter
        generation += 1


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
    parser.add_argument("-p", "--psize", type=int, default=10,
                        help="population size")
    parser.add_argument("-m", "--maxsize", type=int, default=5,
                        help="individual size")
    parser.add_argument("-g", "--generations", type=int, default=5,
                        help="number of generations")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="seed number")
    parser.add_argument("-cp", "--crossover_probability", type=float, default=0.9,
                        help="crossover probability")
    parser.add_argument("-mp", "--mutation_probability", type=float, default=0.5,
                        help="mutation probability")
    parser.add_argument("-t", "--tournament_size", type=int, default=2,
                        help="tournament size")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    print(args)
    ea(args.psize, args.maxsize, args.generations,
       args.mutation_probability, args.tournament_size)


if __name__ == '__main__':
    import time
    start_time = time.time()
    main()
    execution_time = time.time() - start_time
    print('Execution time: {} seconds'.format(execution_time))

