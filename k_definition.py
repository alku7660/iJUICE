import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from evaluator_constructor import distance_calculation
from data_constructor import load_dataset
from main import train_fraction, seed_int, step
from address import results_k_definition, save_obj, load_obj

def get_idx_cf(data):
    """
    Gets the indices of the counterfactual instances
    """
    train_label, desired_label = data.train_target, int(1 - data.undesired_class)
    idx_counterfactuals = np.where(train_label == desired_label)[0]
    return idx_counterfactuals

def calculate_cf_train_distance_matrix(data, idx_counterfactuals, type):
    """
    Computes the distance matrix among the counterfactual training observations
    """
    train = data.transformed_train_np
    counterfactual_train = train[idx_counterfactuals,:]
    dist = np.zeros((len(counterfactual_train), len(counterfactual_train)))
    for idx_i in range(len(counterfactual_train)-1):
        i = counterfactual_train[idx_i]
        for idx_j in range (idx_i+1, len(counterfactual_train)):
            j = counterfactual_train[idx_j]
            dist[idx_i, idx_j] = distance_calculation(i, j, data, type)
            dist[idx_j, idx_i] = dist[idx_i, idx_j]
    return dist

def estimate_outliers(data, type, neighbors):
    """
    Estimates how many outliers are there in the counterfactual class
    """
    dist, idx_counterfactuals = load_obj(f'dist_matrix_{data.name}_{type}', results_k_definition), load_obj(f'idx_counterfactuals_{data.name}', results_k_definition)
    cl = LocalOutlierFactor(n_neighbors=neighbors, metric='precomputed')
    outliers_labels = cl.fit_predict(dist)
    original_data_idx_outliers = idx_counterfactuals[np.where(outliers_labels == -1)]
    outliers = data.transformed_train_np[original_data_idx_outliers]
    return outliers

datasets = ['credit','compass','diabetes','student','oulad','law','heart','synthetic_disease'] # ,'adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law','heart','synthetic_disease'
distance_type = ['L1_L0','L1_L0_L_inf','prob']

for data_str in datasets:
    data = load_dataset(data_str, train_fraction, seed_int, step)
    idx_counterfactuals = get_idx_cf(data)
    save_obj(idx_counterfactuals, results_k_definition, f'idx_counterfactuals_{data_str}')
    for type in distance_type:
        dist = calculate_cf_train_distance_matrix(data, idx_counterfactuals, type)
        save_obj(dist, results_k_definition, f'dist_matrix_{data_str}_{type}')
        print(f'Dataset: {data_str}, Distance: {type}, Total instances: {data.train_target[np.where(data.train_target == 1 - data.undesired_class)[0]].shape[0]}')


# for data_str in datasets:
#     data = load_dataset(data_str, train_fraction, seed_int, step)
#     total_cf_instances = data.train_target[np.where(data.train_target == 1 - data.undesired_class)[0]].shape[0]
#     range_neighbors = range(int(total_cf_instances*0.05), int(total_cf_instances*0.2))
#     pandas_results_type = pd.DataFrame(index=distance_type, columns=range_neighbors)
#     for type in distance_type:
#         for neighbors in range_neighbors:
#             outliers = estimate_outliers(data, type, neighbors)
#             print(f'Dataset: {data_str}, Distance: {type}, Neihgbors: {neighbors}, Total instances: {total_cf_instances}, Number of outliers: {len(outliers)}')
#             suggested_k = len(outliers) + 1
#             pandas_results_type.loc[type, neighbors] = suggested_k
#     pandas_results_type.to_csv(results_k_definition+'k_definition.csv')