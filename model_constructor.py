import numpy as np
import pandas as pd
import ast
from address import results_grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class Model:

    def __init__(self, data) -> None:
        self.model = self.train_model(data)
    
    def best_model_params(self, grid_search_pd, data_str):
        """
        Method that delivers the best model and its parameters according to the Grid Search done
        """
        if data_str in ['bank','ionosphere','german','dutch','kdd_census','student']:
            best = 'rf'
        elif data_str in ['adult','compass','credit','diabetes','german','law','oulad','synthetic_athlete','synthetic_disease']:
            best = 'mlp'
        params_best = ast.literal_eval(grid_search_pd.loc[(data_str, best), 'params'])[0]
        return best, params_best

    def classifier(self, model_str, best_params, train_data, train_target):
        """
        Method that outputs the best trained model according to Grid Search done
        """
        random_st = 54321 
        if model_str == 'mlp':
            best_activation = best_params['activation']
            best_hidden_layer_sizes = best_params['hidden_layer_sizes']
            best_solver = best_params['solver']
            best_model = MLPClassifier(activation=best_activation, hidden_layer_sizes=best_hidden_layer_sizes, solver=best_solver, random_state=random_st)
            best_model.fit(train_data,train_target)
        elif model_str == 'rf':
            best_max_depth = best_params['max_depth']
            best_min_samples_leaf = best_params['min_samples_leaf']
            best_min_samples_split = best_params['min_samples_split']
            best_n_estimators = best_params['n_estimators']
            best_model = RandomForestClassifier(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf, min_samples_split=best_min_samples_split, n_estimators=best_n_estimators)
            best_model.fit(train_data,train_target) 
        return best_model

    def train_model(self, data):
        """
        Constructs a model for the dataset using sklearn modules
        """
        grid_search_results = pd.read_csv(results_grid_search+'grid_search.csv', index_col = ['dataset','model'])
        sel_model_str, params_best = self.best_model_params(grid_search_results, data.name)
        global_model = self.classifier(sel_model_str, params_best, data.transformed_train_df, data.train_target)
        return global_model