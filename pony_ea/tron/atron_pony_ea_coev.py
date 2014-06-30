#! /usr/env python

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
import argparse
import random

from tron_adversarial import TronAdversarial, PlayerAI
from tron_oop_pony_ea import TronNonAdversarialFitness, EA

"""
Evolutionary Algorithm for **STU Tron ALFA** adversarial with coevolution.

The source code design is for teaching the concept of how evolution
inspires computational intelligence, not for fast portable use.

.. codeauthor:: Erik Hemberg <hembergerik@csail.mit.edu>

**STU Tron ALFA** Evolutionary Algorithm Description
----------------------------------------------------

Components
~~~~~~~~~~

**TronNonAdversarialFitness**
  The fitness function. 

  - Decodes an individual
  - Evaluates the individual on **STU Tron** and assigns it a fitness score

**EA Coevolution** 
  The evolutionary algorithm, performs a *stochastic parallel
  iterative* search. The algorithm:

  - Generate a population of *initial solutions*
  - *Iterate* a fixed number of times 

    - *Select* solutions for a new population
    - *Vary* the solutions in the new population
  
      - *Mutate* a solution
      - *Crossover* two solutions

    - *Evaluate* the fitness of the new solutions against each other
    - *Replace* the old population with the new population

Running **STU Tron ALFA EA**
----------------------------


::

  usage: atron_pony_ea_coev.py [-h] [-p PSIZE] [-m MAXSIZE] [-e ESIZE]
                             [-g GENERATIONS] [-s SEED] [-cp CROSSOVER]
                             [-mp MUTATION] [-r ROWS] [-d] [-a]

optional arguments:
  -h, --help            show this help message and exit
  -p PSIZE, --psize PSIZE
                        population size
  -m MAXSIZE, --maxsize MAXSIZE
                        individual size
  -e ESIZE, --esize ESIZE
                        elite size
  -g GENERATIONS, --generations GENERATIONS
                        number of generations
  -s SEED, --seed SEED  seed number
  -cp CROSSOVER, --crossover CROSSOVER
                        crossover probability
  -mp MUTATION, --mutation MUTATION
                        mutation probability
  -r ROWS, --rows ROWS  # of board rows
  -d, --draw_board      draw the board or not
  -a, --ai_vs_ai        Ai vs ai evolution
"""


class TronAdversarialFitnessInteractive(TronNonAdversarialFitness):
    """

    **STU Tron ALFA** adversarial interactive fitness function. Evaluates an
    individual interactively and assigns fitness.

    Attributes:
      - Rows -- The number of rows on the **STU Tron** board
      - Bike Width -- Width of the bike in the **STU Tron** game
      - Draw board -- Indicates if the **STU Tron** GUI should be used
      - Strategy template -- Strategy template used :ref:`example of encoding a
      strategy <encoding_ex>`
      - Strategy -- The decoded strategy
      - Game Repetition -- The number of times the Games should be repeated
      between the individuals
    """

    def __init__(self, rows, draw_board, max_size, game_repetitions=2):
        """
        Constructor

        :param rows: Number of rows on the **STU Tron** board
        :type rows: integer
        :param draw_board: Display GUI
        :type draw_board: bool
        :param max_size: the maximum size of a solution
        :type max_size: int
        :param game_repetitions: the number of times a game should be repeated
        :type game_repetitions: int
        """
        super(TronAdversarialFitnessInteractive, self).__init__(rows,
                                                                draw_board,
                                                                max_size)
        self.game_repetitions = game_repetitions

    def __call__(self, individual_0):
        """
        Function call operator. Starts a Tron game and sets the
        strategies of the two players according to the Individual
        function arguments. The fitness of the winner increases by 1.

        :param individual_0: Individual solution
        :type individual_0: Individual
        """
        print(self.__class__.__name__, '.__call__', individual_0.fitness)
        strategy_0 = self.decode_individual(individual_0)
        for game in range(self.game_repetitions):
            #Create Tron game instance
            #Pass in the AI player
            tron_game = TronAdversarial(rows=self.rows,
                                        bike_width=self.bike_width,
                                        draw_board=self.draw_board,
                                        strategy=strategy_0)
            tron_game.run()
            winner = tron_game.winner
            if winner[0] == 1:
                individual_0.fitness += 1

            print(self.__class__.__name__, ".__call__ winner", winner,
                  individual_0.fitness)


class TronAdversarialFitness(TronAdversarialFitnessInteractive):
    """

    **STU Tron ALFA** adversarial fitness function. Evaluates two
    individuals and assigns fitness.

    Attributes:
      - Rows -- The number of rows on the **STU Tron** board
      - Bike Width -- Width of the bike in the **STU Tron** game
      - Draw board -- Indicates if the **STU Tron** GUI should be used
      - Strategy template -- Strategy template used :ref:`example of encoding a
      strategy <encoding_ex>`
      - Strategy -- The decoded strategy
      - Game Repetition -- The number of times the Games should be repeated
      between the individuals
    """

    def __init__(self, rows, draw_board, max_size, game_repetitions=2):
        """
        Constructor

        :param rows: Number of rows on the **STU Tron** board
        :type rows: int
        :param draw_board: Display GUI
        :type draw_board: bool
        :param max_size: the maximum size of a solution
        :type max_size: int
        :param game_repetitions: the number of times a game should be repeated
        :type game_repetitions: int
        """
        super(TronAdversarialFitness, self).__init__(rows, draw_board, max_size)
        self.game_repetitions = game_repetitions

    def __call__(self, individual_0, individual_1):
        """
        Function call operator. Starts a Tron game and sets the
        strategies of the two players according to the Individual
        function arguments. The fitness of the winner increases by 1.

        :param individual_0: Individual solution
        :type individual_0: Individual
        :param individual_1: Individual solution
        :type individual_1: Individual
        """
        # Decode the strategies
        print(self.__class__.__name__, '.__call__', individual_0.fitness,
              individual_1.fitness)
        strategy_0 = self.decode_individual(individual_0)
        strategy_1 = self.decode_individual(individual_1)
        # TODO for 2 ai players the outcome for each repetion will be the same
        # since there is no stochastic element
        for game in range(self.game_repetitions):
            #Create Tron game instance
            tron_game = TronAdversarial(rows=self.rows,
                                        bike_width=self.bike_width,
                                        draw_board=self.draw_board,
                                        strategy=None)
            #Set players in Tron
            tron_game.players[0] = PlayerAI(x=self.rows / 2,
                                            y=self.rows / 2,
                                            direction=(0, 1),
                                            color="Blue",
                                            alive=True,
                                            id_=0,
                                            canvas=tron_game.canvas,
                                            board=tron_game.board,
                                            strategy=strategy_0)
            tron_game.players[1] = PlayerAI(x=self.rows / 2 + 1,
                                            y=self.rows / 2 + 1,
                                            direction=(0, 1),
                                            color="Green",
                                            alive=True,
                                            id_=1,
                                            canvas=tron_game.canvas,
                                            board=tron_game.board,
                                            strategy=strategy_1)

            # Run the Tron game
            tron_game.run()
            # Get the winner
            winner = tron_game.winner
            # Increase the fitness of the winner
            if winner[0] == 0:
                individual_0.fitness += 1
            elif winner[0] is not None:
                individual_1.fitness += 1

            print(self.__class__.__name__, ".__call__ winner", winner,
                  individual_0.fitness, individual_1.fitness)


class EACoevolution(EA):
    """
    Evolutionary Algorithm which uses coevolution as a fitness function.
    """

    def __init__(self, population_size, max_size, generations, elite_size,
                 crossover_probability, mutation_probability, fitness_function):
        """
        Constructor

        :param population_size: Size of population
        :type population_size: int
        :param max_size: Bitstring size for an individual solution
        :type max_size: int
        :param generations: Number of iterations of the search loop
        :type generations: int
        :param elite_size: Number of individuals preserved between generations
        :type elite_size: int
        :param crossover_probability: Probability of crossing over two solutions
        :type crossover_probability: float
        :param mutation_probability: Probability of mutating a solution
        :type mutation_probability: float
        :param fitness_function: Method used to evaluate fitness of a solution
        :type fitness_function: Object
        """
        EA.__init__(self, population_size, max_size, generations, elite_size,
                    crossover_probability, mutation_probability,
                    fitness_function)

    def evaluate_fitness(self, individuals, fitness_function):
        """
        Perform the coevolutionary fitness evaluation for each
        individual. Each individual competes against each other.

        :param individuals: Population to evaluate
        :type individuals: list
        :param fitness_function: Fitness function to evaluate the population on
        :type fitness_function: Object
        """
        # Reset the fitness of all solutions. The fitness is relative to the
        # opponent's
        for individual in individuals:
            individual.fitness = 0

        # Iterate over all the individual solutions
        for i, ind_0 in enumerate(individuals):
            # Count variable for evaluating each individual against all others
            cnt = min(i + 1, len(individuals))
            for ind_1 in individuals[cnt:]:
                fitness_function(ind_0, ind_1)


def main():
    """
    Parse the command line arguments. Create the **STU Tron** fitness
    function and the Genetic Algorithm. Run the
    search.

    :return: Best individual
    :rtype: Individual
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Population size
    parser.add_argument("-p", "--psize", type=int, default=10,
                        help="population size")
    # Size of an individual
    parser.add_argument("-m", "--maxsize", type=int, default=14,
                        help="individual size")
    # Number of elites, i.e. the top solution from the old population
    # transferred to the new population
    parser.add_argument("-e", "--esize", type=int, default=0, help="elite size")
    # Generations is the number of times the EA will iterate the search loop
    parser.add_argument("-g", "--generations", type=int, default=5,
                        help="number of generations")
    # Random seed. Used to allow replication of runs of the EA. The search is
    # stochastic and and replication of the results can be guaranteed by using
    # the same random seed
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="seed number")
    # Probability of crossover
    parser.add_argument("-cp", "--crossover", type=float, default=0.1,
                        help="crossover probability")
    # Probability of mutation
    parser.add_argument("-mp", "--mutation", type=float, default=0.8,
                        help="mutation probability")
    # Number of rows on the *STU Tron* board
    parser.add_argument("-r", "--rows", type=int, default=10,
                        help="# of board rows")
    # Turn on or of the display of the board
    parser.add_argument("-d", "--draw_board", action='store_true',
                        help="draw the board or not")
    # Turn on AI vs AI evolution or have interactive evolution of human versus
    # AI
    parser.add_argument("-a", "--ai_vs_ai", action='store_true',
                        help="Ai vs ai evolution")
    args = parser.parse_args()
    # Set arguments
    population_size = args.psize
    max_size = args.maxsize
    generations = args.generations
    elite_size = args.esize
    seed = args.seed
    crossover_probability = args.crossover
    mutation_probability = args.mutation
    rows = args.rows
    draw_board = args.draw_board
    ai_vs_ai = args.ai_vs_ai

    # Print EA settings
    print(args)

    # Set random seed if not 0 is passed in as the seed
    if seed != 0:
        random.seed(seed)

    #Interactive or AI vs AI game
    if ai_vs_ai:
        fitness_function = TronAdversarialFitness(rows, draw_board, max_size)
        pony_ea = EACoevolution(population_size, max_size, generations,
                                elite_size, crossover_probability,
                                mutation_probability, fitness_function)
    else:
        fitness_function = TronAdversarialFitnessInteractive(rows, draw_board,
                                                             max_size)
        pony_ea = EA(population_size, max_size, generations, elite_size,
                     crossover_probability,
                     mutation_probability, fitness_function)

    return pony_ea.run()

if __name__ == '__main__':
    import time

    start_time = time.time()
    best_solution = main()
    print('Best solution: %s' % best_solution)
    execution_time = time.time() - start_time
    print('Execution time: %f seconds' % execution_time)
