import csv

__author__ = "Erik Hemberg"
__version__ = "2"
__date__ = "17/01/2018"


def parse_city_data(file_name):
    """Cost matrix for cities from CSV file.

    :param file_name: Name of CSV file.
    :type file_name: str
    :return: Cost matrix for cities
    :rtype: list of lists
    """
    city_data = []
    with open(file_name, 'r') as in_file:
        city_reader = csv.reader(in_file, delimiter=',')
        for line in city_reader:
            cols = [float(_) for _ in line]
            city_data.append(cols)

    return city_data


def get_tour_cost(tour, cost_matrix):
    """
    Cost of tour given a cost matrix.

    The tour cost is the sum of all edges. The tour returns to the start point

    :param tour: Nodes to visit
    :type tour: List of integers
    :param cost_matrix: Cost of path between nodes
    :type cost_matrix: list of lists
    :return: Total cost of tour
    """
    # Calculate cost of the new tour
    total_cost = 0
    # Get start point
    _from = tour[0]
    # Do not iterate over start and add return to destination
    _tour = tour[1:] + tour[:1]
    for _to in _tour:
        total_cost += cost_matrix[_from][_to]
        _from = _to

    return total_cost


if __name__ == '__main__':
    _city_file_name = 'tsp_costs.csv'
    # Parse the cities included in the tour
    _city_data = parse_city_data(_city_file_name)
    print(_city_data)
    tour_ = [0, 2, 1, 3]
    _total_cost = get_tour_cost(tour_, _city_data)
    print(tour_, _total_cost)
    tour = [1, 3, 2, 0]
    _total_cost = get_tour_cost(tour_, _city_data)
    print(tour_, _total_cost)
