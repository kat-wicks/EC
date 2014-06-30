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

import random
import math
import copy
import argparse
import time

from atron_pony_ea_coev import EACoevolution, EA
from tron_adversarial import TronAdversarial, PlayerAI


"""
PonyGP

Implementation of GP to describe how the algorithm works.

The population is an instance of the class List
and contains individuals

Individual has fields
- genome
- fitness

Genome is object of class Tree

Tree has a root which is an object of class TreeNode

TreeNode has instance variables
- parent
- children (a list of TreeNodes)
- label

.. codeauthor:: Erik Hemberg <hembergerik@csail.mit.edu>

"""


class Tree(object):
    """
    A Tree has a root which is an object of class TreeNode

    Attributes:

    - Root -- The root node of the tree
    - Node count -- The number of nodes in the tree
    - Depth -- The maximum depth of the tree
    """

    def __init__(self, root):
        """
        Constructor

        :param root: Root node of the tree
        :type root: TreeNode
        """
        # Root of tree
        self.root = root
        # Number of nodes in the tree
        self.node_cnt = 1
        # Largest depth of the tree
        self.depth = 1

    def grow(self, node, depth, max_depth, full, _symbols):
        """
        Recursively grow a node to max depth in a pre-order, i.e. depth-first
        left-to-right traversal.

        :param node: Root node of subtree
        :type node: TreeNode
        :param depth: Current tree depth
        :type depth: int
        :param max_depth: Maximum tree depth
        :type max_depth: int
        :param full: grows the tree to max depth when true
        :type full: bool
        :param _symbols: set of symbols to chose from
        :type _symbols: Symbols
        """

        # grow is called recursively in the loop. The loop iterates arity number
        # of times. The arity is given by the node symbol
        for _ in range(_symbols.arities[node.symbol]):
            # Get a random symbol
            symbol = _symbols.get_rnd_symbol(depth, max_depth, full)
            # Increase the node count
            self.node_cnt += 1
            # Create a child node object of the current node
            child = TreeNode(node, symbol)
            # Append the child node to the current node
            node.children.append(child)
            # Call grow with the child node as the current node
            self.grow(child, depth + 1, max_depth, full, _symbols)

    def calculate_depth(self):
        """
        Return the maximum depth of the tree.

        :returns: Maximum depth of the tree
        :rtype: int
        """

        # Get a list of all nodes
        all_nodes = self.depth_first(self.root)
        # Find the depth of each node
        # TODO improve depth calculation by not iterating over all nodes. Let
        # get_depth or depth_first have side-effects
        node_depths = [self.get_depth(node) for node in all_nodes]
        # The maximum depth of the tree is the node with the greatest depth
        self.depth = max(node_depths)

        return self.depth

    def depth_first(self, root):
        """
        Return a list of nodes of recursively collected by pre-order
        depth-first
        left-to-right traversal.

        :param root: Start of traversal
        :type root: TreeNode
        :return: list of nodes in pre-order
        :rtype: list
        """

        # Add the root node to the list of nodes
        nodes = [root]
        # Iterate over the children of the root node. If the node is a
        # leaf node then it has no children and there will be no more
        # calls to depth_first.
        for child in root.children:
            # Append the nodes returned from the child node
            nodes += (self.depth_first(child))

        # Return the list of nodes
        return nodes

    @staticmethod
    def get_depth(node):
        """
        Return depth of node by counting the number of parents when
        traversing up the tree.

        :param node: Node which depth is calculated
        :type node: TreeNode
        :return: depth of node in tree
        :rtype: int
        """

        # Set the starting depth
        depth = 0
        # If the current node has a parent then set the parent to be
        # the current node
        while node.parent:
            # The parent of the current node is set to be the current
            # node
            node = node.parent
            # Depth is increased by one
            depth += 1

        # Return the depth after following all the parent references
        return depth

    def __str__(self):
        """
        Return string representation of tree

        :return: string of tree
        :rtype: str
        """
        # String representation
        _str = 'node_cnt:%d depth:%d root:%s' % \
               (self.node_cnt, self.depth, self.root.str_as_tree())
        # Return string representation
        return _str


class TreeNode(object):
    """
    A node in a tree.

    Attributes:
      - Parent -- The parent of the node. None indicates no parent, i.e. root
      node
      - Symbol -- The label of the node
      - Children -- The children of the node
    """

    def __init__(self, parent=None, symbol=None):
        """
        Constructor

        :param parent: Parent node
        :type parent: TreeNode
        :param symbol: Node symbol
        :type symbol: str
        """
        # The parent of the tree node
        self.parent = parent
        # The symbol of the node (a.k.a label)
        self.symbol = symbol
        # The children of the node
        self.children = []

    def str_as_tree(self):
        """
        Return an s-expression for the node and its descendants.

        :return: S-expression
        :rtype: str
        """

        # The number of children determines if it is a internal or
        # leaf node
        if len(self.children):
            # Append a ( before the symbol to denote the start of a subtree
            _str = "(" + str(self.symbol)
            # Iterate over the children
            for child in self.children:
                # Append a " " between the child symbols
                _str += " " + child.str_as_tree()

            # Append a ) to close the subtree
            _str += ")"

            # Return the subtree string
            return _str
        else:
            # Return the symbol
            return str(self.symbol)

    def __repr__(self):
        """
        Return a detailed string representation of the node
        itself.

        :return: String representation of node
        :rtype: str
        """
        # Check if root node
        if self.parent is None:
            # The root has no parent
            parent = str(self.parent)
        else:
            # parent is set to its symbol
            parent = self.parent.symbol

        # Return parent and symbol
        return "p:%s label:%s" % (parent, self.symbol)


class Symbols(object):
    """
    Symbols are functions (internal nodes) or terminals (leaves)

    Attributes:
      - Arities -- Dictionary with the symbol as a key and the arity of symbol
      the symbol as the value
      - Terminals -- List of the terminal symbols, arity is 0
      - Functions -- List of the function symbols, arity greater than 0
    """

    def __init__(self, arities):
        """
        Constructor

        :param arities: symbol names and their arities
        :type arities: dict
        """

        # Arities dictionary with symbol as key and arity as value
        self.arities = arities
        # List of terminal symbols
        self.terminals = []
        # List of function symbols
        self.functions = []

        # Append symbols to terminals or functions by looping over the
        # arities items
        for key, value in self.arities.items():
            # A symbol with arity 0 is a terminal
            if value == 0:
                # Append the symbols to the terminals list
                self.terminals.append(key)
            else:
                # Append the symbols to the functions list
                self.functions.append(key)

    def get_rnd_symbol(self, depth, max_depth, full=False):
        """
        Get a random symbol. The depth determines if a terminal
        must be chosen. If full is specified a function will be chosen
        until max_depth.

        :param depth: current depth
        :type depth: int
        :param max_depth: maximum allowed depth
        :type max_depth: int
        :param full: grow to full depth
        :type full: bool
        :return: symbol
        :rtype: str
        """

        # Pick a terminal if max depth has been reached
        if depth >= max_depth:
            # Pick a random terminal
            symbol = random.choice(self.terminals)
        else:
            # Can it be a terminal before the max depth is reached
            # then there is 50% chance that it is a terminal
            if not full and bool(random.getrandbits(1)):
                # Pick a random terminal
                symbol = random.choice(self.terminals)
            else:
                # Pick a random function
                symbol = random.choice(self.functions)

        # Deal with random numbers. Pick a random number and attach to the
        # symbol and save the terminal.
        # TODO Improve efficiency, the terminals and arities structures can
        # grow very large
        if symbol == 'C_X':
            # Pick a random float in [0.0, 1.0]
            constant = random.random()
            # Create a string symbol
            symbol = '%s(%1.6f)' % (symbol, constant)
            # Append the constant symbol to terminals
            self.terminals.append(symbol)
            # Add the constant symbol to arities
            self.arities[symbol] = 0

        # Return the picked symbol
        return symbol


class Individual(object):
    """
    A GP Individual.

    Attributes:
      - Genome -- A tree
      - Fitness -- The fitness value of the individual
      - Wins -- Number of wins for an individual
      - Games -- NUmber of games played

    DEFAULT_FITNESS
      Default fitness value of an unevaluated individual

    """

    DEFAULT_FITNESS = -1000

    def __init__(self, genome):
        """
        Constructor

        :param genome: genome of the individual
        :type genome: Tree
        """
        # Set the genome (a.k.a input) of the individual
        self.genome = genome
        # Set the fitness to the default value
        self.fitness = Individual.DEFAULT_FITNESS

        # Values used for *STU Tron* to calculate fitness
        # Number of wins
        self.wins = 0
        # Number of games
        self.games = 0

    def __lt__(self, other):
        """
        Returns the comparison of fitness values between two individuals.

        :param other: other individual to compare against
        :type other: Individual
        :returns: if the fitness is lower than the other individual
        :rtype: bool
        """
        # Compare the fitness of this and the other individual
        return self.fitness < other.fitness

    def __str__(self):
        """
        Returns a string representation of fitness and genome

        :returns: String with fitness and genome
        :rtype: bool
        """
        # String representation by calling the root node of the genome
        # as a s-expression
        _str = 'Individual: %f, %s' % \
               (float(self.fitness), self.genome.root.str_as_tree())
        # Return string representation
        return _str


class PlayerAIGP(PlayerAI):
    """
    PlayerAIGP extends PlayerAI. Overrides the evaluate strategy function to
    call the evaluator function defined in the fitness function.

    Attributes:
      - Evaluator -- References the function used to evaluate the Player
    """

    def __init__(self, x, y, direction, color, alive, id_, canvas, board,
                 strategy, evaluator):
        """
        Constructor

        :param x: x coordinate
        :type x: int
        :param y: y coordinate
        :type y: int
        :param direction: coordinates for the current direction
        :type direction: tuple
        :param color: color of the player
        :type color: str
        :param alive: player life indicator
        :type alive: bool
        :param id_: unique identifier
        :type id_ int
        :param canvas: canvas to draw player on
        :type canvas: TronCanvas
        :param board: game board
        :type board: list
        :param strategy: player strategy
        :type strategy: str
        :param evaluator: Reference to evaluator function
        :type evaluator: function
        """
        super(PlayerAIGP, self).__init__(x, y, direction, color, alive, id_,
                                         canvas, board, strategy)
        # Function used to evaluate a player
        self.evaluator = evaluator

    def evaluate_strategy(self):
        """
        Check the environment and call the evaluator function. Pass in the
        PlayerAIGP instance as an argument.
        """
        # Check the environment
        self.check_environment()
        # Enable access to the PlayerAIGP methods and fields by
        # passing it as an argument
        self.evaluator(self)


class TronAdversarialFitnessInteractiveGP(object):
    """
    STU Tron Non-Adversarial player fitness function. Evaluates the fitness
    of an
    individual.

    Attributes:
      - Rows -- The number of rows on the **STU Tron** board
      - Bike Width -- Width of the bike in the **STU Tron** game
      - Draw board -- Indicates if the **STU Tron** GUI should be used
      - Strategy -- The strategy, i.e. the individual tree
      - Player -- A reference to the object which has the functions which are
      evaluated
      - Game repetitions -- Number of times the game is repeated
    """

    def __init__(self, rows, draw_board, game_repetitions=2):
        """Constructor

        :param rows: Number of rows on the **STU Tron** board
        :type rows: integer
        :param draw_board: Display GUI
        :type draw_board: integer
        :param game_repetitions: Number of times the game is repeated
        :type game_repetitions: int
        """
        # Rows in tron game
        self.rows = rows
        # Bike width in tron game
        self.bike_width = 4
        # Display the board or not. The evaluation is significantly
        # faster if the board is not displayed
        self.draw_board = draw_board
        # Reference to the object which has the functions evaluated
        self.player = None
        # Number of times the game is repeated
        self.game_repetitions = game_repetitions

    def __call__(self, individual_0):
        """
        Function call operator. Starts a Tron game and sets the
        strategy of the player according to the Individual
        function arguments. The fitness is the length of the tail.

        :param individual_0: Individual solution
        :type individual_0: Individual
        """

        # The strategy is the executable GP-tree
        strategy = individual_0.genome
        # Play a number of games
        for game in range(self.game_repetitions):
            # Create Tron game instance
            tron_game = TronAdversarial(rows=self.rows,
                                        bike_width=self.bike_width,
                                        draw_board=self.draw_board,
                                        strategy=strategy)
            # Create an AI GP player
            tron_game.players[1] = \
                PlayerAIGP(x=tron_game.players[1].x,
                           y=tron_game.players[1].y,
                           direction=tron_game.players[1].direction,
                           color=tron_game.players[1].color,
                           alive=tron_game.players[1].alive,
                           id_=tron_game.players[1].ID,
                           canvas=tron_game.players[1].canvas,
                           board=tron_game.players[1].board,
                           strategy=tron_game.players[1].strategy,
                           # Set the strategy
                           # evaluation function in
                           # the tron player
                           evaluator=self.evaluate_strategy)

            # Run the tron game
            tron_game.run()
            # Get the winner of the game
            winner = tron_game.winner
            # Get the ID of the winner
            if winner[0] == 1:
                # Increase the fitness of the AI if it is the winner
                individual_0.fitness += 1

            print(self.__class__.__name__, ".__call__ winner", winner,
                  individual_0.fitness)

    def evaluate_strategy(self, player):
        """
        Wrapper function for evaluating a tree. Start evaluating from the root.

        TODO Enforcing closure by abusing typing by catching TypeError and
        ignoring it. A nicer solution is to use Strongly Typed GP

        :param player: Player strategy to evaluate
        :type player: PlayerAIGP
        """
        # Set the current player
        self.player = player

        try:
            # Evaluate the player strategy
            self.evaluate(player.strategy.root)
        except TypeError:
            # Ignore type errors, this handles closure in a crude manner
            pass

    def evaluate(self, node):
        """
        Return the value from the node. Recursively evaluates the nodes of a
        tree in a depth-first left-to-right manner

        :param node: Node to evaluate
        "type node: TreeNode
        """

        # The current token is the symbol of the node
        token = node.symbol
        # Check what the token is
        if token == 'ifleq':
            # Check if the first child node is less than the second child node
            if self.evaluate(node.children[0]) < self.evaluate(
                    node.children[1]):
                # Evaluate the third child node
                self.evaluate(node.children[2])
            else:
                # Evaluate the fourth child node
                self.evaluate(node.children[3])
        elif token.startswith('direction'):
            # Get the direction integer
            direction = int(token[-2:-1])
            # Return the distance to an obstacle in that direction
            return self.player.distance(direction)
        elif token == 'left()':
            # Turn the player left
            self.player.left()
        elif token == 'right()':
            # Turn the player right
            self.player.right()
        elif token == 'ahead()':
            # Let the player go ahead
            self.player.ahead()
        elif token.startswith('C_X'):
            # Get the constant number
            constant = float(token[-9:-1])
            return constant
        elif token == '*':
            # Multiply the first child node with the second child node
            return self.evaluate(node.children[0]) * self.evaluate(
                node.children[1])
        elif token == '+':
            # Add the first child node to the second child node
            return self.evaluate(node.children[0]) + self.evaluate(
                node.children[1])
        elif token == '-':
            # Subtract the first child node from the second child node
            return self.evaluate(node.children[0]) - self.evaluate(
                node.children[1])
        elif token == '/':
            # Divide the first child node with the second child node
            try:
                val = self.evaluate(node.children[0]) / self.evaluate(
                    node.children[1])
                return val
            except ZeroDivisionError:
                # Return a number for zero division
                return 1000
        else:
            # Raise an error if there is an unknown token
            raise ValueError('Unknown token: %s' % token)


class TronAdversarialFitnessGP(TronAdversarialFitnessInteractiveGP):
    """
    STU Tron Non-Adversarial player fitness function. Evaluates the fitness
    of an individual.

    Attributes:
      - Rows -- The number of rows on the **STU Tron** board
      - Bike Width -- Width of the bike in the **STU Tron** game
      - Draw board -- Indicates if the **STU Tron** GUI should be used
      - Strategy -- The strategy, i.e. the individual tree
      - Player -- A reference to the object which has the functions which are
      evaluated

    """

    def __init__(self, rows, draw_board):
        """Constructor

        :param rows: Number of rows on the **STU Tron** board
        :type rows: integer
        :param draw_board: Display GUI
        :type draw_board: integer
        """
        # Only play 1 game
        super(TronAdversarialFitnessGP, self).__init__(rows, draw_board, 1)
        # List of strategies
        self.strategies = []

    def __call__(self, individual_0, individual_1):
        """
        Function call operator. Starts a Tron game and sets the
        strategy of the player according to the Individual
        function arguments. The fitness is the length of the tail.

        :param individual_0: Individual solution
        :type individual_0: Individual
        :param individual_1: Individual solution
        :type individual_1: Individual
        """
        # Set the strategies as the GP tree of the individuals
        self.strategies.append(individual_0.genome)
        self.strategies.append(individual_1.genome)
        # Play a number of games
        for game in range(self.game_repetitions):
            # Create Tron game instance
            tron_game = TronAdversarial(rows=self.rows,
                                        bike_width=self.bike_width,
                                        draw_board=self.draw_board,
                                        strategy=self.strategies[0])
            # Create a list of players
            players = []
            # Loop over the strategies and the players in he Tron game
            for player, strategy in zip(tron_game.players, self.strategies):
                # Append the players to the player list
                players.append(PlayerAIGP(x=player.x,
                                          y=player.y,
                                          direction=player.direction,
                                          color=player.color,
                                          alive=player.alive,
                                          id_=player.ID,
                                          canvas=player.canvas,
                                          board=player.board,
                                          strategy=strategy,
                                          # Set the strategy evaluation function
                                          # in the tron player
                                          evaluator=self.evaluate_strategy))

            # Set the players in the tron game
            tron_game.players = players
            # Run the tron game
            tron_game.run()
            # Get the winner
            winner = tron_game.winner
            # Check ID of the winner
            if winner[0] == 0:
                # Assign fitness
                individual_0.wins += 1
            elif winner[0] is not None:
                # Assign fitness
                individual_1.wins += 1
            else:
                # Punish behavior for drawing
                individual_0.wins -= 2
                individual_1.wins -= 2

            # Increase the number of games played
            individual_0.games += 1
            individual_1.games += 1

            print(self.__class__.__name__, ".__call__ winner", winner,
                  individual_0.wins, individual_1.wins)


class GP(EA):
    """
    Genetic Programming implementation.

    Attributes:

    - Population size -- Size of the population
    - Solution size -- Max size of the nodes which represents an individual
      solution
    - Max depth -- Max depth of a tree, this is a function of the solution size
    - Generations -- Number of iterations of the search loop
    - Elite size -- Number of individuals preserved between generations
    - Crossover probability -- Probability of crossing over two solutions
    - Mutation probability -- Probability of mutating a solution
    - Fitness function -- Method used to evaluate fitness, e.g.
      :ref:`**STU Tron ALFA** non-adversarial <tron_non_adversarial_fitness>`

    POPULATION_FILE
      File where population is saved
    """

    POPULATION_FILE = 'gp_population.dat'

    def __init__(self, population_size, max_size, generations,
                 elite_size, crossover_probability,
                 mutation_probability, fitness_function, symbols):
        """Constructor
        
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
        :param symbols: Symbols used to build trees
        :type symbols: Symbols
        """
        super(GP, self).__init__(population_size, max_size, generations,
                                 elite_size, crossover_probability,
                                 mutation_probability, fitness_function)

        # Max depth is a function of the max_size
        self.max_depth = GP.get_max_depth(self.max_size)
        # The symbols used in the GP Trees
        self.symbols = symbols

    @classmethod
    def get_max_depth(cls, size):
        """
        Return the max depth of a binary tree given a size. The size is
        the number of nodes.

        :param size: Number of tree nodes
        :type size: int
        :returns: Max depth of the binary tree
        :rtype: int
        """
        return int(math.log(size, 2))

    def initialize_population(self):
        """
        Ramped half-half initialization. The individuals in the
        population are initialized using the grow or the full method for
        each depth value (ramped) up to max_depth.

        :returns: List of individuals
        :rtype: list
        """

        individuals = []
        for i in range(self.population_size):
            #Pick full or grow method
            full = bool(random.getrandbits(1))
            #Ramp the depth
            max_depth = (i % self.max_depth) + 1
            #Create root node
            symbol = self.symbols.get_rnd_symbol(1, max_depth)
            root = TreeNode(None, symbol)
            tree = Tree(root)
            #Grow the tree if the root is a function symbol
            if tree.depth < max_depth and symbol in self.symbols.functions:
                tree.grow(tree.root, 1, max_depth, full, self.symbols)
            individuals.append(Individual(tree))
            print('Initial tree %d: %s' % (i, tree.root.str_as_tree()))

        return individuals

    def search_loop(self, population):
        """
        Return the best individual from the evolutionary search
        loop. Starting from the initial population.
        
        :param population: Initial population of individuals
        :type population: list
        :returns: Best individual
        :rtype: Individual
        """

        # Evaluate fitness
        self.evaluate_fitness(population, self.fitness_function)
        best_ever = None

        #Generation loop
        generation = 0
        while generation < self.generations:
            new_population = []
            # Selection
            parents = self.tournament_selection(population)

            # Crossover
            while len(new_population) < self.population_size:
                # Vary the population by crossover
                new_population.extend(
                    # Pick 2 parents and pass them into crossover.
                    self.subtree_crossover(*random.sample(parents, 2))
                )
            # Select population size individuals. Handles uneven population
            # sizes, since crossover returns 2 offspring
            new_population = new_population[:self.population_size]

            # Vary the population by mutation
            new_population = list(map(self.subtree_mutation, new_population))

            # Evaluate fitness
            self.evaluate_fitness(new_population, self.fitness_function)

            # Replace population
            population = self.generational_replacement(new_population,
                                                       population)
            # Print the stats of the population
            self.print_stats(generation, population)

            # Set best solution
            population.sort(reverse=True)
            best_ever = population[0]

            # Increase the generation counter
            generation += 1

        return best_ever

    def print_stats(self, generation, individuals):
        """
        Print the statistics for the generation and population.
       
        :param generation:generation number
        :type generation: int
        :param individuals: population to get statistics for
        :type individuals: list
        """

        def get_ave_and_std(values):
            """
            Return average and standard deviation.            

            :param values: Values to calculate on
            :type values: list
            :returns: Average and Standard deviation of the input values
            :rtype: tuple
            """
            _ave = float(sum(values)) / len(values)
            _std = math.sqrt(float(
                sum((value - _ave) ** 2 for value in values)) / len(values))
            return _ave, _std

        # Make sure individuals are sorted
        individuals.sort(reverse=True)
        # Get the fitness values
        fitness_values = [i.fitness for i in individuals]
        # Get the number of nodes
        size_values = [i.genome.node_cnt for i in individuals]
        # Get the max depth
        depth_values = [i.genome.calculate_depth() for i in individuals]
        # Get average and standard deviation of fitness
        ave_fit, std_fit = get_ave_and_std(fitness_values)
        # Get average and standard deviation of size
        ave_size, std_size = get_ave_and_std(size_values)
        # Get average and standard deviation of max depth
        ave_depth, std_depth = get_ave_and_std(depth_values)
        # Print the statistics
        print(
            "Gen:%d evals:%d fit_ave:%.2f+-%.3f size_ave:%.2f+-%.3f "
            "depth_ave:%.2f+-%.3f %s" %
            (generation, (self.population_size * generation),
             ave_fit, std_fit,
             ave_size, std_size,
             ave_depth, std_depth,
             individuals[0]))

    def subtree_mutation(self, individual):
        """
        Return a new individual by randomly picking a node and growing a
        new subtree from it.
        
        :param individual: Individual to mutate
        :type individual: Individual
        :returns: Mutated individual
        :rtype: Individual
        """

        new_individual = Individual(copy.deepcopy(individual.genome))
        # Check if mutation should be applied
        if random.random() < self.mutation_probability:
            # Pick node
            node = random.choice(
                new_individual.genome.depth_first(new_individual.genome.root))
            # Clear children
            node.children[:] = []
            # Get depth of the picked node
            node_depth = new_individual.genome.get_depth(node)
            # Set a new symbol for the picked node
            node.symbol = self.symbols.get_rnd_symbol(node_depth,
                                                      self.max_depth)
            # Grow tree if it was a function symbol
            if node.symbol in self.symbols.functions:
                # Grow subtree
                new_individual.genome.grow(node, node_depth, self.max_depth,
                                           bool(random.getrandbits(1)),
                                           self.symbols)

            # Set the new node count in the tree
            node_cnt = len(
                new_individual.genome.depth_first(new_individual.genome.root))
            new_individual.genome.node_cnt = node_cnt
            # Get the new max depth of the tree
            new_individual.genome.calculate_depth()

        # Return the individual
        return new_individual

    def subtree_crossover(self, parent1, parent2):
        """
        Returns two individuals. The individuals are created by
        selecting two random nodes from the parents and swapping the
        subtrees.

        :param parent1: Individual to crossover
        :type parent1: Individual
        :param parent2: Individual to crossover
        :type parent2: Individual
        :returns: Two new individuals
        :rtype: tuple
        """

        # Copy the parents to make offsprings
        offsprings = (Individual(copy.deepcopy(parent1.genome)),
                      Individual(copy.deepcopy(parent2.genome)))

        # Check if offspring will be crossed over
        if random.random() < self.crossover_probability:
            #Pick a crossover point
            offspring_0_node = random.choice(
                offsprings[0].genome.depth_first(offsprings[0].genome.root))
            #Only crossover internal nodes, not only leaves
            if offspring_0_node.symbol in self.symbols.functions:
                # Get the nodes from the second offspring
                nodes = offsprings[1].genome.depth_first(
                    offsprings[1].genome.root)
                # List to store possible crossover nodes
                possible_nodes = []
                #Find possible crossover points
                for node in nodes:
                    # If there is a matching arity the nodes can be crossed over
                    matching_type = self.symbols.arities[node.symbol] == \
                                    self.symbols.arities[
                                        offspring_0_node.symbol]
                    # Append the node to the possible crossover nodes
                    if matching_type:
                        possible_nodes.append(node)

                # Pick a crossover point in the second offspring
                if possible_nodes:
                    #Pick the second crossover point
                    offspring_1_node = random.choice(possible_nodes)
                    #Swap the children of the nodes
                    node_children = (
                        offspring_0_node.children, offspring_1_node.children)
                    # Copy the children from the subtree of the first offspring
                    # to the chosen node of the second offspring
                    offspring_1_node.children = copy.deepcopy(node_children[0])
                    # Copy the children from the subtree of the second offspring
                    # to the chosen node of the first offspring
                    offspring_0_node.children = copy.deepcopy(node_children[1])

        # Return the offsprings
        return offsprings


class GPCoevolution(GP, EACoevolution):
    """
    Genetic Programming implementation for coevolutionary fitness evaluation. 

    """

    def evaluate_fitness(self, individuals, fitness_function):
        """
        Perform the coevolutionary fitness evaluation for each
        individual. Each individual competes against each other.

        :param individuals: Population to evaluate
        :type individuals: list
        :param fitness_function: Fitness function to evaluate the population on
        :type fitness_function: Object
        """
        # Pick a number of opponents
        nr_opponents = 4
        # Assert that the number of opponents are smaller than the population
        # size
        assert nr_opponents < (self.population_size - 1)
        # Iterate over the population
        for ind_0 in individuals:
            # Set the current individual in the opponent list to avoid competing
            #  against one self
            opponents = [ind_0]
            # Compete against the given number of opponents
            while len(opponents) < nr_opponents:
                # Pick a random opponent.
                opponent = random.choice(individuals)
                # Check if the opponent has been played
                if opponent not in opponents:
                    # Play the opponent
                    fitness_function(ind_0, opponent)
                    # Append the opponent to the list of played opponents
                    opponents.append(opponent)

        # For each individual set the fitness based on the wins and games played
        for ind in individuals:
            ind.fitness = float(ind.wins) / float(ind.games)

        print('FIT %s' % ' '.join(map(str, [_.fitness for _ in individuals])))


def get_symbols():
    """
    Return symbols object. 

    :returns: Symbols object
    :rtype: Symbols
    """

    # Dictionary of symbols and their arity
    arities = {"left()": 0,
               "right()": 0,
               "direction[0]": 0,
               "direction[1]": 0,
               "direction[2]": 0,
               "direction[3]": 0,
               "C_X": 0,
               "+": 2,
               "-": 2,
               "*": 2,
               "/": 2,
               "ifleq": 4}
    # Create a symbols object
    symbols = Symbols(arities)

    # Return the Symbols
    return symbols


def main():
    """
    Parse the command line arguments. Create the **STU Tron** fitness
    function and the Genetic Algorithm. Run the
    search.

    :return: Best individual
    :rtype: Individual
    """
    # Command line arguments
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
    # Parse the command line arguments
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
    # Get the symbols
    symbols = get_symbols()

    # Interactive or AI vs AI game
    if ai_vs_ai:
        # Create the AI vs AI fitness function
        fitness_function = TronAdversarialFitnessGP(rows, draw_board)
        # Create the coevolutionary GP algorithm
        gp = GPCoevolution(population_size, max_size,
                           generations, elite_size, crossover_probability,
                           mutation_probability, fitness_function, symbols)
    else:
        # Create the interactive fitness function
        fitness_function = TronAdversarialFitnessInteractiveGP(rows, draw_board)
        # Create the GP algorithm
        gp = GP(population_size, max_size,
                generations, elite_size, crossover_probability,
                mutation_probability, fitness_function, symbols)
    # Start the GP run
    best_individual = gp.run()

    # Return the best solution
    return best_individual


if __name__ == '__main__':
    # Time the execution
    start_time = time.time()
    # Start the run
    best_solution = main()
    # Print the best solution
    print('Best solution: %s' % best_solution)
    # Get the execution time
    execution_time = time.time() - start_time
    # Print the execution time
    print('Execution time: %f seconds' % execution_time)    
