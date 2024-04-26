"""This file is used to define tools functions that will be used in others files"""
import numpy as np
from itertools import product
from sksurv.metrics import concordance_index_censored
import random



'''
----------------------------Score functions----------------------------
'''

def concordance_censored(estimator,X,y):
    concordance = concordance_index_censored([elt[0] for elt in y],[elt[1] for elt in y],estimator.predict(X))
    return concordance[0]


'''
----------------------------Optimization functions----------------------------
'''


'''This function aims to return a dictionary of hyperparameters close to those obtained by optuna'''
def generate_param_grid(optimal_params, num_samples=3, percent_range=0.1):
    param_grid = {}

    for param_name, param_value in optimal_params.items():
        if isinstance(param_value, int):
            # If the parameter is an integer
            lower_bound = max(1, int(param_value * (1 - percent_range)))
            upper_bound = int(param_value * (1 + percent_range))
            # If the range is too small, extend the grid to include at least 3 values
            if param_value < 10:
                lower_bound = max(1, param_value - num_samples//2)
                upper_bound = param_value + num_samples//2 + 1
                if lower_bound == 1 : 
                    param_values = list(set(list(range(lower_bound, upper_bound))))
                else : 
                    param_values = list(range(lower_bound, upper_bound))
            else :
                step = (upper_bound - lower_bound) / (num_samples - 2)
                param_values = np.arange(lower_bound, upper_bound + 1, step, dtype=int).tolist()

        elif isinstance(param_value, float):
            # If the parameter is floating
            if param_value > 1:
                lower_bound = max(0.0, param_value * (1 - percent_range))
                upper_bound = param_value * (1 + percent_range)
                param_values = np.linspace(lower_bound, upper_bound, num_samples - 1).tolist()
            else : 
                lower_bound = max(0.0, param_value * (1 - percent_range))
                upper_bound = min(1, param_value * (1 + percent_range))
                param_values = np.linspace(lower_bound, upper_bound, num_samples - 1).tolist()
            param_values.append(param_value)

        elif isinstance(param_value, str):
            param_values = [param_value]

        param_grid[param_name] = param_values

    return param_grid


"""This function aims to create a list of all hyperparameters combinations possibles from a dictionary"""
def get_param_combinations(param_grid):
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    for values in product(*param_values):
        yield dict(zip(param_names, values))


"""This function aims to return the number of combinations possibles using all keys of the dictionary"""
def count_combinations(dic):
    nb_comb = 1
    for values in dic.values():
        if not(isinstance(values[0],bool)):
            nb_comb = nb_comb * len(values)
    return nb_comb


"""This function aims to genereate n random numbers between min_val and max_val"""
def generate_random_numbers(min_val, max_val, n):
    random_numbers = []
    for _ in range(n):
        random_numbers.append(random.uniform(min_val, max_val))
    return random_numbers


"""This function aims to generate a dictionary of random number knowing max_val et min_val giving in a dictionary at the beginning"""
def random_number_dict(dictionary):
    random_numbers = {}
    for key, bounds in dictionary.items():
        # If the values are int 
        if isinstance(bounds[0], int):
            [min_val, max_val] = bounds

            # Generate a random number within the bounds and store it in the result dictionary
            random_numbers[key] = random.randint(min_val, max_val)

        # If the values are float   
        elif isinstance(bounds[0],float):
            [min_val, max_val] = bounds

            # Generate a random number within the bounds and store it in the result dictionary
            random_numbers[key] = random.uniform(min_val, max_val)

        elif isinstance(bounds[0], str):
            random_numbers[key] = random.choice(bounds)

    return random_numbers