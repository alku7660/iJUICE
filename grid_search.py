"""
Imports
"""
import warnings
warnings.filterwarnings("ignore")
import itertools
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from data_constructor import load_dataset
from model_constructor import Model
import pandas as pd
import numpy as np

path_here = os.path.abspath('')
datasets = ['diabetes'] # ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law','heart','synthetic_athlete','synthetic_disease'] ,'german','dutch','bank','credit','compass','student','oulad','law','heart','synthetic_athlete','synthetic_disease'
seed_int = 54321
step = 0.01
train_fraction = 0.7

np.random.seed(seed_int)

index_result = pd.MultiIndex.from_tuples(list(itertools.product(datasets, ['svm','dt','mlp','rf'])), names=['dataset','model']) 
results = pd.DataFrame(index=index_result,columns=['params','score'])

for data_str in datasets:
    data = load_dataset(data_str, train_fraction, seed_int, step)
    train, train_target = data.transformed_train_np, data.train_target
    param_grid_mlp = {'hidden_layer_sizes':[(10,1),(20,1),(50,1),(100,1),
                                            (10,2),(20,2),(50,2),(100,2),
                                            (10,5),(20,5),(50,5),(100,5),
                                            (10,10),(20,10),(50,10),(100,10),
                                            (10,20),(20,20),(50,20),(100,20)],
                                            'activation':['logistic','tanh','relu'],'solver':['lbfgs','sgd','adam']}
    clf_search_mlp = GridSearchCV(MLPClassifier(),param_grid=param_grid_mlp,scoring='f1',cv=5,verbose=2.5)
    clf_search_mlp.fit(train,train_target)
    results.loc[(data_str,'mlp'),'params'] = [clf_search_mlp.best_params_]
    results.loc[(data_str,'mlp'),'score'] = clf_search_mlp.best_score_

    param_grid_rf = {'n_estimators':[10,20,50,100,200],'max_depth':[2,5,10], 'min_samples_split':[2,5,10], 'min_samples_leaf':[1,3,5]}
    clf_search_rf = GridSearchCV(RandomForestClassifier(),param_grid=param_grid_rf,scoring='f1',cv=5,verbose=2.5)
    clf_search_rf.fit(train,train_target)
    results.loc[(data_str,'rf'),'params'] = [clf_search_rf.best_params_]
    results.loc[(data_str,'rf'),'score'] = clf_search_rf.best_score_
    
results.to_csv(str(path_here)+'/Results/grid_search/grid_search_fairness_added.csv')