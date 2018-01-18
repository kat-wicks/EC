# pony_ea.py


Implementation of an Evolutionary Algorithm to solve __Travelling Salesman Problem__. 

Run with default settings
```
python pony_ea.py
```


# Examples

5 city tour with exhaustive solution shown
```
python pony_ea.py --tsp_data tsp_costs_5.csv --tsp_exhaustive

Gen:0; Population fitness mean:11.47+-2.291; Best solution:[0, 4, 1, 3, 2], fitness:16.0
Gen:1; Population fitness mean:10.90+-2.135; Best solution:[0, 4, 1, 2, 3], fitness:15.0
Gen:2; Population fitness mean:9.77+-1.430; Best solution:[4, 3, 2, 0, 1], fitness:14.0
Gen:3; Population fitness mean:9.83+-1.551; Best solution:[3, 4, 1, 2, 0], fitness:14.0
Gen:4; Population fitness mean:10.83+-2.570; Best solution:[4, 1, 3, 2, 0], fitness:16.0
EA:
 Best tour cost is 16.0 for path [4, 1, 3, 2, 0]. Searched 150 points in 0.00570 seconds
```

10 city tour
```
python pony_ea.py --tsp_data tsp_costs_10.csv

Gen:0; Population fitness mean:22.63+-3.146; Best solution:[1, 3, 7, 0, 6, 5, 8, 2, 9, 4], fitness:27.0
Gen:1; Population fitness mean:23.10+-3.350; Best solution:[5, 7, 2, 0, 6, 1, 9, 8, 3, 4], fitness:31.0
Gen:2; Population fitness mean:21.70+-3.057; Best solution:[5, 7, 1, 9, 6, 8, 2, 0, 4, 3], fitness:28.0
Gen:3; Population fitness mean:19.70+-2.854; Best solution:[5, 2, 7, 9, 4, 0, 8, 1, 3, 6], fitness:25.0
Gen:4; Population fitness mean:19.30+-3.121; Best solution:[9, 4, 0, 3, 5, 8, 1, 2, 7, 6], fitness:29.0
EA:
 Best tour cost is 29.0 for path [9, 4, 0, 3, 5, 8, 1, 2, 7, 6]. Searched 150 points in 0.00756 seconds
 ```

# Usage

```
usage: pony_ea.py [-h] [-p POPULATION_SIZE] [-g GENERATIONS] [-s SEED]
                  [-cp CROSSOVER_PROBABILITY] [-mp MUTATION_PROBABILITY]
                  [-t TOURNAMENT_SIZE] [--elite_size ELITE_SIZE]
                  [--tsp_data TSP_DATA] [--tsp_exhaustive] [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  -p POPULATION_SIZE, --population_size POPULATION_SIZE
                        Population size
  -g GENERATIONS, --generations GENERATIONS
                        number of generations
  -s SEED, --seed SEED  Random seed number
  -cp CROSSOVER_PROBABILITY, --crossover_probability CROSSOVER_PROBABILITY
                        Crossover probability
  -mp MUTATION_PROBABILITY, --mutation_probability MUTATION_PROBABILITY
                        Mutation probability
  -t TOURNAMENT_SIZE, --tournament_size TOURNAMENT_SIZE
                        Tournament size
  --elite_size ELITE_SIZE
                        Elite size
  --tsp_data TSP_DATA   Data for Travelling Salesman problem in a CSV file.
  --tsp_exhaustive      Perform exhaustive search of TSP.
  --verbose             Verbose mode
```
