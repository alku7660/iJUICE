import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from evaluator_constructor import distance_calculation

def calculate_cf_train_distance_matrix(counterfactual):
    """
    Computes the distance matrix among the counterfactual training observations
    """
    data, distance_type = counterfactual.data, counterfactual.type
    train, train_label, desired_label = data.transformed_train_np, data.train_target, int(1 - data.undesired_class)
    idx_counterfactuals = np.where(train_label == desired_label)
    counterfactual_train = train[idx_counterfactuals,:]
    dist = np.zeros(len(counterfactual_train), len(counterfactual_train))
    for idx_i in range(len(counterfactual_train)-1):
        i = counterfactual_train[idx_i]
        for idx_j in range (idx_i+1, len(counterfactual_train)):
            j = counterfactual_train[idx_j]
            dist[idx_i, idx_j] = distance_calculation(i, j, data, distance_type)
    return dist, idx_counterfactuals

