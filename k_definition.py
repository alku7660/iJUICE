import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from evaluator_constructor import distance_calculation
from data_constructor import load_dataset
from main import train_fraction, seed_int, step
from address import results_k_definition

def calculate_cf_train_distance_matrix(data, type):
    """
    Computes the distance matrix among the counterfactual training observations
    """
    train, train_label, desired_label = data.transformed_train_np, data.train_target, int(1 - data.undesired_class)
    idx_counterfactuals = np.where(train_label == desired_label)
    counterfactual_train = train[idx_counterfactuals,:]
    dist = np.zeros(len(counterfactual_train), len(counterfactual_train))
    for idx_i in range(len(counterfactual_train)-1):
        i = counterfactual_train[idx_i]
        for idx_j in range (idx_i+1, len(counterfactual_train)):
            j = counterfactual_train[idx_j]
            dist[idx_i, idx_j] = distance_calculation(i, j, data, type)
    return dist, idx_counterfactuals

def estimate_outliers(data, type, neighbors):
    """
    Estimates how many outliers are there in the counterfactual class
    """
    dist, idx_counterfactuals = calculate_cf_train_distance_matrix(data, type)
    cl = LocalOutlierFactor(n_neighbors=neighbors, metric='precomputed')
    outliers_labels = cl.fit_predict(dist)
    original_data_idx_outliers = idx_counterfactuals[np.where(outliers_labels == -1)]
    outliers = data.transformed_train_np[original_data_idx_outliers]
    return outliers

range_neighbors = range(10,20)
datasets = ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law','heart','synthetic_athlete','synthetic_disease']
distance_type = ['L1_L0','L1_L0_L_inf','prob']

for data_str in datasets:
    data = load_dataset(data_str, train_fraction, seed_int, step)
    pandas_results_type = pd.DataFrame(index=distance_type, columns=range_neighbors)
    for type in distance_type:
        for neighbors in range_neighbors:
            outliers = estimate_outliers(data, type, neighbors)
            print(f'Dataset: {data_str}, Distance: {type}, Number of outliers: {len(outliers)}')
            suggested_k = len(outliers) + 1
            pandas_results_type.loc[type, neighbors] = suggested_k
    pandas_results_type.to_csv(results_k_definition+'k_definition.csv')