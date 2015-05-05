var Tree = function (root) {

    this.root = root;
    this.nodeCnt = 1;
    this.depth = 1;
};

Tree.prototype.copy = function () {
    var root = this.root.copy();
    return new Tree(root);
};

Tree.prototype.calculateDepth = function () {
    var allNodes = [];
    this.depthFirst(this.root, 0, allNodes);
    var nodeDepths = [];
    for (var i = 0; i < allNodes.length; i = i + 1) {
        nodeDepths.push(allNodes[i].depth);
    }
    this.depth = max(nodeDepths);
    this.nodeCnt = nodeDepths.length;
    return this.depth
};

Tree.prototype.depthFirst = function (root, depth, nodes) {
    nodes.push({node: root, depth: depth});
    for (var i = 0; i < root.children.length; i = i + 1) {
        this.depthFirst(root.children[i], depth + 1, nodes);
    }
};

var TreeNode = function (parent, symbol) {
    this.parent = parent;
    this.symbol = symbol;
    this.children = [];
};

TreeNode.prototype.copy = function () {
    var copy = new TreeNode(this.parent, this.symbol);
    for (var i = 0; i < this.children.length; i = i + 1) {
        copy.children.push(this.children[i].copy());
    }
    return copy;
};

TreeNode.prototype.strAsTree = function () {
    var str_ = "";
    if (this.children.length > 0) {
        str_ = "(" + this.symbol;
        for (var i = 0; i < this.children.length; i = i + 1) {
            str_ = str_ + " " + this.children[i].strAsTree();
        }
        str_ = str_ + ")";
    } else {
        str_ = this.symbol;
    }
    return str_;
};

TreeNode.prototype.grow = function (node, depth, maxDepth, full, symbols) {
    for (var i = 0; i < symbols.arities[node.symbol]; i = i + 1) {
        var symbol = symbols.getRandomSymbol(depth, maxDepth, full);
        var child = new TreeNode(node, symbol);
        node.children.push(child);
        this.grow(child, depth + 1, maxDepth, full, symbols);
    }
};

var Symbols = function (arities) {
    this.arities = arities;
    this.terminals = [];
    this.functions = [];
    for (var symbol in this.arities) {
        if (this.arities[symbol] == 0) {
            this.terminals.push(symbol);
        } else {
            this.functions.push(symbol);
        }
    }
};

Symbols.prototype.getRandomSymbol = function (depth, maxDepth, full) {
    var symbol;
    if (depth == maxDepth) {
        symbol = RNG.getRandomChoice(this.terminals);
    } else {
        if (full || RNG.getRandomBoolean()) {
            symbol = RNG.getRandomChoice(this.functions);
        } else {
            symbol = RNG.getRandomChoice(this.terminals);
        }
    }
    //TODO random numbers symbols
    return symbol
};

var Individual = function (genome) {
    this.genome = genome;
    this.fitness = DEFAULT_FITNESS;
};

Individual.prototype.copy = function () {
    var genome = this.genome.copy();
    return new Individual(genome);
};

var SymbolicRegression = function (data) {
    this.data = data;
    this.labelCol = this.data[0].length - 1;
};

SymbolicRegression.prototype.evaluateIndividual = function (individual) {
    var fitness = 0.0;
    for (var i = 0; i < this.data.length; i++) {
        var output = this.evaluate(individual.genome.root, this.data[i]);
        var target = this.data[i][this.labelCol];
        var error = output - target;
        fitness = fitness + error * error;
    }
    fitness = -fitness / this.data.length;
    return fitness;
};

SymbolicRegression.prototype.evaluatePopulation = function (population) {
    for (var i = 0; i < population.length; i = i + 1) {
        population[i]["fitness"] = this.evaluateIndividual(population[i]);
    }
    return population;
};

SymbolicRegression.prototype.evaluate = function (node, inputData) {
    // Get the symbol of the node
    var symbol = node.symbol;
    var value;
    if (symbol == "+") {
        value = this.evaluate(node.children[0], inputData) +
        this.evaluate(node.children[1], inputData);
    } else if (symbol == "*") {
        value = this.evaluate(node.children[0], inputData) *
        this.evaluate(node.children[1], inputData);
    } else if (symbol == "-") {
        value = this.evaluate(node.children[0], inputData) -
        this.evaluate(node.children[1], inputData);
    } else if (symbol == "/") {
        var numerator = this.evaluate(node.children[0], inputData);
        var denominator = this.evaluate(node.children[1], inputData);
        if (denominator != 0) {
            value = numerator / denominator;
        } else {
            value = DIV_BY_ZERO_VALUE;
        }
    } else if (symbol == "0") {
        value = Number(symbol);
    } else if (symbol == "1") {
        value = Number(symbol);
    } else {
        var split_ = symbol.split(DATA_VARIABLE);
        if (split_.length == 2) {
            var idx = parseInt(split_[1]);
            if (idx > -1 && idx < inputData.length) {
                value = inputData[idx];
            } else {
                throw "Bad index for symbol:" + symbol + ' idx:' + idx;
            }
        } else {
            throw "Unknown symbol:" + symbol;
        }
    }
    return value;
};

var GP = function (params) {
    this.params = params;
};

GP.prototype.initializePopulation = function () {
    var population = [];
    var populationSize = this.params["population_size"];
    var symbols = this.params["symbols"];
    for (var i = 0; i < populationSize; i++) {
        var full = RNG.getRandomBoolean();
        var maxDepth = (i % this.params.maxDepth) + 1;
        var symbol = symbols.getRandomSymbol(1, maxDepth, full);
        var root = new TreeNode(null, symbol);
        var tree = new Tree(root);
        if (maxDepth > 0 && contains(symbols["functions"], root.symbol)) {
            root.grow(root, 1, maxDepth, full, symbols);
        }
        population.push(new Individual(tree));
        console.log(i, population[i].genome.root.strAsTree());
        console.log(population[i]);
    }
    return population;
};

function compareIndividuals(individual0, individual1) {
    if (individual0.fitness < individual1.fitness) {
        return 1;
    } else {
        if (individual0.fitness > individual1.fitness) {
            return -1;
        } else {
            return 0;
        }
    }
}

GP.prototype.searchLoop = function (population) {
    var population = this.evaluateFitness(population);
    var generation = 0;
    var elites = [];
    while (generation < this.params["generations"]) {
        // Elite solutions
        population.sort(compareIndividuals);
        for (var i = 0; i < this.params["elite_size"]; i++) {
            elites.push(population[i]);
        }
        // Selection
        var newPopulation = this.tournamentSelection(this.params['tournament_size'],
            population);
        newPopulation = this.crossover(this.params['crossover_probability'],
            newPopulation);
        newPopulation = this.mutation(this.params['mutation_probability'],
            newPopulation, this.params['maxDepth'], this.params["symbols"]);

        // Evaluate the new population
        newPopulation = this.evaluateFitness(newPopulation);

        // Replace the population with the new population
        population = newPopulation;
        // Add elites
        for (var i = 0; i < this.params["elite_size"]; i++) {
            population[population.length - 1 - i] = elites.pop();
        }

        var bestSolution = this.printStats(generation, newPopulation);
        //TODO good to break here?
        if (Math.abs(bestSolution["fitness"]) < this.params["error_cutoff"]) {
            break;
        }
        // Increase the generation
        generation = generation + 1;
    }
    // Test on out-of-sample data
    window.best_solution = bestSolution;
    gp_params.onComplete();
    return bestSolution;

};

GP.prototype.evaluateFitness = function (population) {
   population = this.params["fitnessFunction"].evaluatePopulation(population);
    return population;
};

GP.prototype.tournamentSelection = function (tournamentSize, population) {
    var new_population = [];
    while (new_population.length < population.length) {
        var competitors = [];
        // Randomly select competitors for the tournament
        for (var i = 0; i < tournamentSize; i++) {
            var idx = RNG.getRandomInt(0, population.length - 1);
            competitors.push(population[idx]);
        }
        // Sort the competitors by fitness
        competitors.sort(compareIndividuals);
        // Push the best competitor to the new population
        var winner = competitors[0].copy();
        new_population.push(winner);
    }
    return new_population;
};

GP.prototype.printStats = function (generation, population) {

    function getAverageAndStd(values) {

        function sum(array) {
            return array.reduce(function (previous_value, current_value) {
                return previous_value + current_value;
            });
        }

        var ave = sum(values) / values.length;
        var std = 0;
        for (var val in values) {
            std = std + Math.pow((val - ave), 2);
        }
        std = Math.sqrt(std / values.length);
        return [ave, std];
    }

    var fitnessValues = [];
    var sizes = [];
    var depths = [];
    for (var i = 0; i < population.length; i++) {
        fitnessValues.push(population[i]["fitness"]);
        population[i]["genome"].calculateDepth();
        sizes.push(population[i]["genome"].nodeCnt);
        depths.push(population[i]["genome"].depth);
    }
    var aveAndStdFitness = getAverageAndStd(fitnessValues);
    var aveAndStdSize = getAverageAndStd(sizes);
    var aveAndStdDepth = getAverageAndStd(depths);

    population.sort(compareIndividuals);
    var bestSolution = population[0];

    console.log("Gen:" + generation + " fit_ave:" + aveAndStdFitness[0] + "+-" + aveAndStdFitness[1] +
    " size_ave:" + aveAndStdSize[0] + "+-" + aveAndStdSize[1] +
    " depth_ave:" + aveAndStdDepth[0] + "+-" + aveAndStdDepth[1] +
    " " + bestSolution["fitness"] + " " + bestSolution["genome"].root.strAsTree());
    console.log("min_fit:" + min(fitnessValues) + " max_fit:" + max(fitnessValues) +
    " min_size:" + min(sizes) + " max_size:" + max(sizes) +
    " min_depth:" + min(depths) + " max_depth:" + max(depths));
    return bestSolution;
};

GP.prototype.mutation = function (mutationProbability, individuals, maxDepth, symbols) {
    var newIndividuals = [];
    for (var i = 0; i < individuals.length; i = i + 1) {
        var newIndividual = individuals[i].copy();
        if (RNG.getRandom() < mutationProbability) {
            var nodes = [];
            newIndividual.genome.depthFirst(newIndividual.genome.root, 0, nodes);
            var node_ = RNG.getRandomChoice(nodes);
            var nodeDepth = node_.depth;
            var node = node_.node;
            node.children = [];
            var full = RNG.getRandomBoolean();
            node.symbol = symbols.getRandomSymbol(nodeDepth, maxDepth, full);
            node.grow(node, nodeDepth + 1, maxDepth, full, symbols);
       }
        newIndividuals.push(newIndividual);
    }
    return newIndividuals;
};

GP.prototype.crossover = function (crossoverProbability, population) {
    var CHILDREN = 2;
    var newPopulation = [];
    for (var i = 0; i < population.length; i = i + CHILDREN) {
        var children = [];
        for (var j = 0; j < CHILDREN; j++) {
            var idx = RNG.getRandomInt(0, population.length - 1);
            var child = population[idx].copy();
            children.push(child);
        }
        if (RNG.getRandom() < crossoverProbability) {
            var xoNodes = [];
            var compatibleNodes = true;
            for (var j = 0; j < children.length; j++) {
                var nodes = [];
                children[j]["genome"].depthFirst(children[j]["genome"].root, 0, nodes);
                var node_ = RNG.getRandomChoice(nodes);
                var nodeDepth = node_.depth;
                var node = node_.node;
                var subTree = new Tree(node);
                subTree.calculateDepth();
                xoNodes.push({node: node, subTreeSize: subTree.depth, nodeDepth: nodeDepth});
                //TODO tidy up compatible nodes
                compatibleNodes = node.children.length > 0 && compatibleNodes;
                if (j > 0) {
                    compatibleNodes = xoNodes[j].node.children.length == xoNodes[j - 1].node.children.length && compatibleNodes;
                    compatibleNodes = xoNodes[j].subTreeSize + xoNodes[j-1].nodeDepth < this.params.maxDepth && compatibleNodes;
                    compatibleNodes = xoNodes[j-1].subTreeSize + xoNodes[j].nodeDepth < this.params.maxDepth && compatibleNodes;
                }
            }
            //Only crossover at function nodes with same arity
            if (compatibleNodes) {
                var tmpChildren = xoNodes[0].node.children;
                xoNodes[0].node.children = xoNodes[1].node.children;
                xoNodes[1].node.children = tmpChildren;
            }
        }
        for (var j = 0; j < children.length; j++) {
            newPopulation.push(children[j]);
        }
    }
    return newPopulation;
};

function contains(array, obj) {
    for (var i = 0; i < array.length; i++) {
        if (array[i] === obj) {
            return true;
        }
    }
    return false;
}

var RNG = {seed: 711};

// From http://indiegamr.com/generate-repeatable-random-numbers-in-js/
// the initial seed
RNG.seededRandom = function (min, max) {
    RNG.seed = (RNG.seed * 9301 + 49297) % 233280;
    var rnd = RNG.seed / 233280;
    var value = min + rnd * (max - min);
    return value;
};

RNG.getRandom = function () {
    return RNG.seededRandom(0, 1);
};

RNG.getRandomInt = function (min, max) {
    var value = RNG.seededRandom(min, max);
    var intValue = Math.floor(value);
    return intValue;
};

RNG.getRandomBoolean = function () {
    return RNG.seededRandom(0, 1) < 0.5;
};

RNG.getRandomChoice = function (array) {
    var idx = RNG.getRandomInt(0, array.length);
    return array[idx];
};

function max(list) {
    return list.reduce(function(previous, current) {
        return previous > current ? previous : current;
    });
}

function min(list) {
    return list.reduce(function(previous, current) {
        return previous < current ? previous : current;
    });
}

var arities = {
    1: 0,
    0: 0,
    x_0: 0,
    "+": 2,
    "*": 2,
    "-": 2,
    "/": 2
};

var DIV_BY_ZERO_VALUE = 10000000;
var DEFAULT_FITNESS = -1000;
var DATA_VARIABLE = "x_";

var gp_params = {
    population_size: 40,
    maxDepth: 3,
    generations: 2,
    mutation_probability: 1.0,
    tournament_size: 2,
    crossover_probability: 1.0,
    elite_size: 1,
    error_cutoff: 0.01,
    symbols: new Symbols(arities)
};


