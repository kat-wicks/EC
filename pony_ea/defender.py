import csv
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

__author__ = "Erik Hemberg"


def parse_data(file_name):

    """
    INPUT: File name pointing to a CSV of a threeclass Tommy Morris Dataset. 
    OUTPUT: X,y where X is a scaled numpy array of size (data points, columns) and y is an array of size (data points)
    """

    df = pd.read_csv(file_name, header =0)
    scaler = StandardScaler()
    print(df.columns)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna( how="any", inplace=True)
    # EH: Always fixed split? (maybe use the random seed passed in when starting?)
    return train_test_split(scaler.fit_transform(df.drop('marker', axis =1)) , df.marker, test_size = 0.2, random_state = 2022) 



def evaluate_solution(solution, X, y):
    """
    Evaluate a given solution in the population according to cost_function

    INPUT: 
        - solution: sklearn-style model
        - X: test data
        - y: test labels
    """
    
    pca_scores = solution[0].transform(X)
    #knn = solution[1].fit(pca_scores,y)
    score = solution[1].score(pca_scores,y)
    return score




# if __name__ == '__main__':
#     _file_name = r'C:\Users\mitadm\Downloads\triple (1)\data1.csv'
#     # Parse the cities included in the tour
#     _city_data = parse_city_data(_city_file_name)
#     print(_city_data)
#     tour_ = [0, 2, 1, 3]
#     _total_cost = evaluate_solution(tour_, _city_data)
#     print(tour_, _total_cost)
#     tour = [1, 3, 2, 0]
#     _total_cost = evaluate_solution(tour_, _city_data)
#     print(tour_, _total_cost)
