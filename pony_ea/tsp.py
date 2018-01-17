import csv

__author__ = "Erik Hemberg"
__version__ = "2"
__date__ = "17/01/2018"


def parse_city_data(file_name):
    """Return a list of City instances

    """
    city_data = []
    with open(file_name, 'r') as in_file:
        city_reader = csv.reader(in_file, delimiter=',')
        for line in city_reader:
            cols = [float(_) for _ in line]
            city_data.append(cols)

    return city_data

def get_tour_cost(tour, cost_matrix):
    

    # Calculate cost of the new tour
    total_cost = 0
    # Get start point
    _from = tour[0]
    # Do not iterate over start
    _tour = tour[1:]
    # Add return to destination
    _tour = _tour + [_from]
    for to in _tour:
        total_cost += cost_matrix[_from][to]
        _from = to

    return total_cost

if __name__ == '__main__':
    _city_file_name = 'tsp_costs.csv'
    # Parse the cities included in the tour
    city_data = parse_city_data(_city_file_name)
    print(city_data)
    _tour = [0, 2, 1, 3]
    total_cost = get_tour_cost(_tour, city_data)
    print( _tour, total_cost)
    _tour = [1, 3, 2, 0]
    total_cost = get_tour_cost(_tour, city_data)
    print( _tour, total_cost)
