"""
Model-Agnostic Counterfactual Explanations (MACE)
Original authors implementation: Please see https://github.com/amirhk/mace
"""

import os
import sys
import copy
import pickle
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
from datetime import datetime
import loadData
import loadModel
import generateSATExplanations

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Datasets/'
results_obj_dir = str(path_here)+'/Results/obj/'

def load_obj(file_address, file_name):
    """
    Method to read an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(results_obj_dir+file_address+file_name, 'rb') as input:
        evaluator_obj = pickle.load(input)
    return evaluator_obj

def getEpsilonInString(approach_string):
    tmp_index = approach_string.find('eps')
    epsilon_string = approach_string[tmp_index + 4 : tmp_index + 8]
    return float(epsilon_string)

def generateExplanations(
    explanation_file_name,
    approach_string,
    model_trained,
    dataset_obj,
    factual_sample,
    norm_type_string):

    return generateSATExplanations.genExp(
        explanation_file_name,
        model_trained,
        dataset_obj,
        factual_sample,
        norm_type_string,
        'mace',
        getEpsilonInString(approach_string)
        )

def runIndices(dataset_values, model_class_values, norm_values, approaches_values, batch_number, gen_cf_for):
    for dataset_string in dataset_values:
        print(f'\n\nExperimenting with dataset_string = `{dataset_string}`')
        for model_class_string in model_class_values:
            print(f'\tExperimenting with model_class_string = `{model_class_string}`')
            for norm_type_string in norm_values:
                print(f'\t\tExperimenting with norm_type_string = `{norm_type_string}`')
                for approach_string in approaches_values:
                    print(f'\t\t\tExperimenting with approach_string = `{approach_string}`')
                    if model_class_string in {'tree', 'forest'}:
                        one_hot = False
                    elif model_class_string in {'lr', 'mlp'}:
                        one_hot = True
                    else:
                        raise Exception(f'{model_class_string} not recognized as a valid `model_class_string`.')
                    
                    # EXPERIMENT FOLDER
                    # experiment_name = f'{dataset_string}__{model_class_string}__{norm_type_string}__{approach_string}__batch{batch_number}__samples{sample_count}__pid{process_id}'
                    # experiment_folder_name = f"_experiments/{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}__{experiment_name}"
                    # explanation_folder_name = f'{experiment_folder_name}/__explanation_log'
                    # minimum_distance_folder_name = f'{experiment_folder_name}/__minimum_distances'
                    # os.mkdir(f'{experiment_folder_name}')
                    # os.mkdir(f'{explanation_folder_name}')
                    # os.mkdir(f'{minimum_distance_folder_name}')
                    # log_file = open(f'{experiment_folder_name}/log_experiment.txt','w')

                    # DATA FILES SAVE
                    dataset_obj = loadData.loadDataset(dataset_string, return_one_hot = one_hot, load_from_cache = False, debug_flag = False) # Load the same dataset_obj
                    # pickle.dump(dataset_obj, open(f'{experiment_folder_name}/_dataset_obj', 'wb'))
                    X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit() # Use the same data split as in the other baselines

                    # MODEL TRAINING
                    model_trained = loadModel.loadModelForDataset(model_class_string, dataset_string) # Cannot use same model as the one trained on other baselines
                    
                    # PREDICTIONS
                    X_test_pred_labels = model_trained.predict(X_test)
                    all_pred_data_df = X_test
                    all_pred_data_df['y'] = X_test_pred_labels
                    neg_pred_data_df = all_pred_data_df.where(all_pred_data_df['y'] == 0).dropna()
                    pos_pred_data_df = all_pred_data_df.where(all_pred_data_df['y'] == 1).dropna()

                    # batch_start_index = batch_number * sample_count
                    # batch_end_index = (batch_number + 1) * sample_count
                    # if gen_cf_for == 'neg_only':
                    #     iterate_over_data_df = neg_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
                    # elif gen_cf_for == 'pos_only':
                    #     iterate_over_data_df = pos_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
                    # elif gen_cf_for == 'neg_and_pos':
                    #     iterate_over_data_df = all_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
                    # else:
                    #     raise Exception(f'{gen_cf_for} not recognized as a valid `gen_cf_for`.')

                    if gen_cf_for == 'neg_only':
                        idx_test = [idx for idx in batch_number if idx in neg_pred_data_df.index]
                        iterate_over_data_df = neg_pred_data_df.loc[idx_test]# choose only a subset to compare
                    elif gen_cf_for == 'pos_only':
                        idx_test = [idx for idx in batch_number if idx in pos_pred_data_df.index]
                        iterate_over_data_df = pos_pred_data_df.loc[idx_test] # choose only a subset to compare
                    elif gen_cf_for == 'neg_and_pos':
                        idx_test = [idx for idx in batch_number if idx in all_pred_data_df.index]
                        iterate_over_data_df = all_pred_data_df.loc[idx_test] # choose only a subset to compare
                    else:
                        raise Exception(f'{gen_cf_for} not recognized as a valid `gen_cf_for`.')

                    pickle.dump(iterate_over_data_df, open(f'{results_obj_dir}/{dataset_string}/{dataset_string}_mace_df.pkl', 'wb'))

def runExperiments(dataset_values, model_class_values, norm_values, approaches_values, batch_number, gen_cf_for):
    for dataset_string in dataset_values:
        print(f'\n\nExperimenting with dataset_string = `{dataset_string}`')
        for model_class_string in model_class_values:
            print(f'\tExperimenting with model_class_string = `{model_class_string}`')
            for norm_type_string in norm_values:
                print(f'\t\tExperimenting with norm_type_string = `{norm_type_string}`')
                for approach_string in approaches_values:
                    print(f'\t\t\tExperimenting with approach_string = `{approach_string}`')
                    if model_class_string in {'tree', 'forest'}:
                        one_hot = False
                    elif model_class_string in {'lr', 'mlp'}:
                        one_hot = True
                    else:
                        raise Exception(f'{model_class_string} not recognized as a valid `model_class_string`.')
                    
                    # EXPERIMENT FOLDER
                    # experiment_name = f'{dataset_string}__{model_class_string}__{norm_type_string}__{approach_string}__batch{batch_number}__samples{sample_count}__pid{process_id}'
                    # experiment_folder_name = f"_experiments/{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}__{experiment_name}"
                    # explanation_folder_name = f'{experiment_folder_name}/__explanation_log'
                    # minimum_distance_folder_name = f'{experiment_folder_name}/__minimum_distances'
                    # os.mkdir(f'{experiment_folder_name}')
                    # os.mkdir(f'{explanation_folder_name}')
                    # os.mkdir(f'{minimum_distance_folder_name}')
                    # log_file = open(f'{experiment_folder_name}/log_experiment.txt','w')

                    # DATA FILES SAVE
                    dataset_obj = loadData.loadDataset(dataset_string, return_one_hot = one_hot, load_from_cache = False, debug_flag = False) # Load the same dataset_obj
                    # pickle.dump(dataset_obj, open(f'{experiment_folder_name}/_dataset_obj', 'wb'))
                    X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit() # Use the same data split as in the other baselines

                    # MODEL TRAINING
                    model_trained = loadModel.loadModelForDataset(model_class_string, dataset_string) # Cannot use same model as the one trained on other baselines
                    
                    # PREDICTIONS
                    X_test_pred_labels = model_trained.predict(X_test)
                    all_pred_data_df = X_test
                    all_pred_data_df['y'] = X_test_pred_labels
                    neg_pred_data_df = all_pred_data_df.where(all_pred_data_df['y'] == 0).dropna()
                    pos_pred_data_df = all_pred_data_df.where(all_pred_data_df['y'] == 1).dropna()

                    # batch_start_index = batch_number * sample_count
                    # batch_end_index = (batch_number + 1) * sample_count
                    # if gen_cf_for == 'neg_only':
                    #     iterate_over_data_df = neg_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
                    # elif gen_cf_for == 'pos_only':
                    #     iterate_over_data_df = pos_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
                    # elif gen_cf_for == 'neg_and_pos':
                    #     iterate_over_data_df = all_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
                    # else:
                    #     raise Exception(f'{gen_cf_for} not recognized as a valid `gen_cf_for`.')

                    if gen_cf_for == 'neg_only':
                        idx_test = [idx for idx in batch_number if idx in neg_pred_data_df.index]
                        iterate_over_data_df = neg_pred_data_df.loc[idx_test]# choose only a subset to compare
                    elif gen_cf_for == 'pos_only':
                        idx_test = [idx for idx in batch_number if idx in pos_pred_data_df.index]
                        iterate_over_data_df = pos_pred_data_df.loc[idx_test] # choose only a subset to compare
                    elif gen_cf_for == 'neg_and_pos':
                        idx_test = [idx for idx in batch_number if idx in all_pred_data_df.index]
                        iterate_over_data_df = all_pred_data_df.loc[idx_test] # choose only a subset to compare
                    else:
                        raise Exception(f'{gen_cf_for} not recognized as a valid `gen_cf_for`.')

                    iterate_over_data_dict = iterate_over_data_df.T.to_dict()
                    explanation_counter = 1
                    all_minimum_distances = {}
                    label_name = list(dataset_obj.attributes_long.keys())[0]
                    original_columns = [i for i in dataset_obj.attributes_long.keys() if i != label_name]
                    cf_df = pd.DataFrame(columns=original_columns)
                    sample_df = pd.DataFrame(columns=original_columns)
                    time_df = pd.DataFrame(columns=['time'])

                    for factual_sample_index, factual_sample in iterate_over_data_dict.items():

                        factual_sample['y'] = bool(factual_sample['y'])

                        print(
                        '\t\t\t\t'
                        f'Generating explanation for\t'
                        f'batch #{batch_number}\t'
                        f'sample #{explanation_counter}/{len(iterate_over_data_dict.keys())}\t'
                        f'(sample index {factual_sample_index}): ', end = '') # , file=log_file)
                        explanation_counter = explanation_counter + 1
                        explanation_file_name = f'sample_{factual_sample_index}.txt'

                        explanation_object = generateExplanations(
                        explanation_file_name,
                        approach_string,
                        model_trained,
                        dataset_obj,
                        factual_sample,
                        norm_type_string)

                        print(
                        f'\t- cfe_found: {explanation_object["cfe_found"]} -'
                        f'\t- cfe_plaus: {explanation_object["cfe_plausible"]} -'
                        f'\t- cfe_time: {explanation_object["cfe_time"]:.4f} -'
                        f'\t- int_cost: N/A -'
                        f'\t- cfe_dist: {explanation_object["cfe_distance"]:.4f} -'
                        ) # , file=log_file)

                        all_minimum_distances[f'sample_{factual_sample_index}'] = explanation_object
                        dataFrame_mace_cf = pd.DataFrame(all_minimum_distances[f'sample_{factual_sample_index}']['cfe_sample'],index=[factual_sample_index])
                        if np.isinf(explanation_object['cfe_distance']):
                            factual_col = list(factual_sample.keys())
                            dataFrame_mace_cf = pd.DataFrame([[np.nan]*len(factual_col)],index=[factual_sample_index],columns=factual_col)
                        del dataFrame_mace_cf['y']
                        dataFrame_mace_cf.columns = original_columns
                        cf_df = pd.concat((cf_df, dataFrame_mace_cf),axis=0)

                        dataFrame_mace_sample = pd.DataFrame(all_minimum_distances[f'sample_{factual_sample_index}']['fac_sample'],index=[factual_sample_index])
                        del dataFrame_mace_sample['y']
                        dataFrame_mace_sample.columns = original_columns
                        sample_df = pd.concat((sample_df, dataFrame_mace_sample),axis=0)

                        dataFrame_mace_time_cf = pd.DataFrame(all_minimum_distances[f'sample_{factual_sample_index}']['cfe_time'],index=[factual_sample_index],columns=['time'])
                        time_df = pd.concat((time_df, dataFrame_mace_time_cf),axis=0)
    
    return cf_df, sample_df, time_df

if __name__ == '__main__':
    dataset_model_dict = {'adult': 'mlp', 'kdd_census': 'forest', 'german':'forest', 'dutch':'forest',
                    'bank':'forest', 'credit':'mlp', 'compass':'mlp', 'diabetes':'mlp', 'ionosphere':'forest',
                    'student':'forest', 'oulad':'mlp', 'law':'mlp', 'synthetic_athlete':'mlp', 'synthetic_disease':'mlp', 'heart':'forest'}
    dataset_undesired_class = {'adult': 'neg_only', 'kdd_census': 'neg_only', 'german':'pos_only', 'dutch':'neg_only',
                    'bank':'neg_only', 'credit':'pos_only', 'compass':'pos_only', 'diabetes':'pos_only', 'ionosphere':'neg_only',
                    'student':'neg_only', 'oulad':'neg_only', 'law':'neg_only', 'synthetic_athlete':'neg_only', 'synthetic_disease':'pos_only', 'heart':'pos_only'}    
    dataset_try = ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','ionosphere','student','oulad','law','heart','synthetic_athlete','synthetic_disease']
    method_try = ['nn']
    norm_type_try = ['zero_norm']
    approach_try = ['MACE_eps_1e-3']
    process_id_try = '0'
    only_indices = False
    for i in range(len(dataset_try)):
        model_class_try = [dataset_model_dict[dataset_try[i]]] 
        batch_number_try = load_obj(f'{dataset_try[i]}/', f'{dataset_try[i]}_idx_list.pkl')            
        gen_cf_for_try = dataset_undesired_class[dataset_try[i]]
        if only_indices:
            runIndices([dataset_try[i]], model_class_try, norm_type_try, approach_try, batch_number_try, gen_cf_for_try)
        else:
            cf_df, sample_df, time_df = runExperiments([dataset_try[i]], model_class_try, norm_type_try, approach_try, batch_number_try, gen_cf_for_try)
            pickle.dump(cf_df, open(f'{results_obj_dir}/{dataset_try[i]}/{dataset_try[i]}_mace_cf_df.pkl', 'wb'))
            pickle.dump(time_df, open(f'{results_obj_dir}/{dataset_try[i]}/{dataset_try[i]}_mace_time_df.pkl', 'wb'))
