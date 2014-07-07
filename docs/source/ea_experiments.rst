Running experiments with **Pony GP**
====================================

Description of how to run experiments with **Pony GP**

Stochastic Search Heuristics
----------------------------

A *heuristic* is a technique for solving a problem:

- quickly when other methods are too inefficient
- for finding an approximate solution when other methods fail to
  find any exact solution

The *heuristic* achieves this by relaxing the:

- optimality, an optimal solution is not guaranteed
- completeness, all the optimal solutions might not be found
- accuracy, what are the accuracy of the solutions
- execution time, how fast is a solution returned

When to use Genetic Programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Non-linear optimization and modelling

  - No analytic method

Key components
~~~~~~~~~~~~~~

- Representation

- Fitness function

- Implementation

Search space size
~~~~~~~~~~~~~~~~~

The search space of **Pony GP** is all the possible combinations of tree
sizes and shapes up until max depth

Design of a Pony GP Experiment
------------------------------

**Pony GP** run
~~~~~~~~~~~~~~~

A run refers to executing the Genetic Program once.

- Quality of best solution

- Statistics from the population as the search progresses over
  generations

An example of output from a **pony_gp.py** run shows:

- the generation number `Gen`
- average fitness plus/minus the standard deviation of the population
  `fit_ave`
- the best individual's fitness and bitstring `Individual`

>>> pony_gp.py
Reading: fitness_cases.csv headers: ['#x0', ' x1', ' y']
(Namespace(crossover_probability=1.0, elite_size=1, fitness_cases='', generations=10, max_depth=3, mutation_probability=1.0, population_size=10, seed=0, test_train_split=0.7, tournament_size=2), {'terminals': ['1', 'x0', 'x1'], 'arities': {'+': 2, '*': 2, '-': 2, '/': 2, '1': 0, 'x0': 0, 'x1': 0}, 'functions': ['+', '*', '-', '/']})
Initial tree 0: ['1']
Initial tree 1: ['x0']
Initial tree 2: ['/', ['x1'], ['x0']]
Initial tree 3: ['x0']
Initial tree 4: ['-', ['-', ['1'], ['1']], ['+', ['x1'], ['x0']]]
Initial tree 5: ['/', ['+', ['*', ['x1'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x0'], ['x1']], ['*', ['x0'], ['1']]]]
Initial tree 6: ['x0']
Initial tree 7: ['1']
Initial tree 8: ['x1']
Initial tree 9: ['x0']
Gen:10 fit_ave:-118.28+-60.645 size_ave:5.60+-4.104 depth_ave:1.40+-1.020 {'genome': ['/', ['+', ['*', ['x1'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x0'], ['x1']], ['*', ['x0'], ['1']]]], 'fitness': -4.444444444444445}
Gen:10 fit_ave:-131.41+-77.873 size_ave:6.40+-5.869 depth_ave:1.50+-1.285 {'genome': ['/', ['+', ['*', ['x1'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x0'], ['x1']], ['*', ['x0'], ['1']]]], 'fitness': -4.444444444444445}
Gen:10 fit_ave:-133.52+-88.737 size_ave:8.80+-4.936 depth_ave:2.10+-0.943 {'genome': ['/', ['+', ['*', ['x0'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x0'], ['x1']], ['*', ['x0'], ['1']]]], 'fitness': -1.8888888888888888}
Gen:10 fit_ave:-116.36+-97.278 size_ave:10.60+-4.454 depth_ave:2.40+-0.917 {'genome': ['/', ['+', ['*', ['x0'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x0'], ['x1']], ['*', ['x0'], ['1']]]], 'fitness': -1.8888888888888888}
Gen:10 fit_ave:-74.52+-76.723 size_ave:14.20+-3.487 depth_ave:3.10+-0.539 {'genome': ['/', ['+', ['*', ['x0'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x0'], ['x1']], ['*', ['x0'], ['1']]]], 'fitness': -1.8888888888888888}
Gen:10 fit_ave:-99.64+-73.270 size_ave:14.20+-4.665 depth_ave:3.00+-0.894 {'genome': ['/', ['+', ['*', ['x0'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x0'], ['x1']], ['*', ['x0'], ['1']]]], 'fitness': -1.8888888888888888}
Gen:10 fit_ave:-109.39+-51.433 size_ave:17.80+-5.381 depth_ave:3.70+-1.005 {'genome': ['/', ['+', ['*', ['x0'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x0'], ['x1']], ['*', ['x0'], ['1']]]], 'fitness': -1.8888888888888888}
Gen:10 fit_ave:-52.65+-43.125 size_ave:16.40+-1.800 depth_ave:3.50+-0.671 {'genome': ['/', ['+', ['*', ['x0'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x0'], ['x1']], ['*', ['x0'], ['1']]]], 'fitness': -1.8888888888888888}
Gen:10 fit_ave:-34.14+-43.750 size_ave:15.40+-2.332 depth_ave:3.20+-0.400 {'genome': ['/', ['+', ['*', ['x0'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x1'], ['x1']], ['*', ['x0'], ['1']]]], 'fitness': -1.8888888888888888}
Gen:10 fit_ave:-103.69+-80.140 size_ave:15.80+-7.277 depth_ave:3.30+-1.100 {'genome': ['/', ['+', ['*', ['x0'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x0'], ['x1']], ['*', ['x0'], ['1']]]], 'fitness': -1.8888888888888888}
Best train:{'genome': ['/', ['+', ['*', ['x0'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x0'], ['x1']], ['*', ['x0'], ['1']]]], 'fitness': -1.8888888888888888}
Best test:{'genome': ['/', ['+', ['*', ['x0'], ['x1']], ['*', ['x0'], ['x0']]], ['/', ['-', ['x0'], ['x1']], ['*', ['x0'], ['1']]]], 'fitness': -0.0}


**Pony GP** experiment
~~~~~~~~~~~~~~~~~~~~~~

- Pick symbols and operators

- Set parameters

  - Number of fitness evaluations - magnitude of search

    - Population size - how many solutions to evaluate in parallel

    - Max iterations - how many generations the population will be
      modified

  - Number of variations - variation frequency,

    - Mutation probability - amplitude and search bias is determined by the operator

    - Crossover probability - amplitude and search bias is determined by the operator

  - Selection pressure - convergence speed of search

    - Selection operator
  
      - Tournament size

      - Elite size - how many solutions are preserved between generations

- Determine the stability of search parameters, the evolutionary search is
  stochastic. Perform a number of independent runs to gain confidence
  in the results
