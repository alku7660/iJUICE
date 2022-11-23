import os
import pickle
path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Datasets/'
results_grid_search = str(path_here)+'/Results/grid_search/'
results_obj = str(path_here)+'/Results/obj/'

def save_obj(evaluator_obj, file_address, file_name):
    """
    Method to store an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(file_address+file_name, 'wb') as output:
        pickle.dump(evaluator_obj, output, pickle.HIGHEST_PROTOCOL)