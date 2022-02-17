import numpy as np


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true+0.0001)) * 100


def get_min_score(my_list):
    temp_list = list()
    for k in my_list:
        temp_list.append(k[-1])
    min_val = min(temp_list)
    return min_val


def get_best_params(my_list):
    min_val = get_min_score(my_list)
    for j in my_list:
        if min_val == j[-1]:
            return j
