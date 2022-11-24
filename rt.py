"""
RF Tweaking (RT)
"""

"""
Imports
"""
import time
import numpy as np
from collections import Counter
from nt import nn

class RT:

    def __init__(self, counterfactual) -> None:
        self.normal_x_cf, self.run_time = rf_tweak(counterfactual)

# As observed in https://github.com/tony-lind/Example-based-tweaking
"""
Created on Fri Feb 15 20:08:02 2019
@author: Tony Lindgren, tony@dsv.su.se
"""
#import pandas as pd
####
# Method that returns an three arrays, one, containing the number of times the an
# example co-occurs in leaf with ex, two the array of example and three the classes of the examples. 
# For which the following conditions holds true:
# examples share leaf node with example in count number of leafs 
# and is of class whish_class. 
# If classified_as_wish is False (default) is sufficent that the example 
# is of wish_class otherwise it must also be classified as such
def random_forest_tweaking(clf, ex, wish_class, X_train, y_train, classified_as_wish=False):
    # calculate leaf_id_matrix
    leaf_id_mat = clf.apply(X_train)
    # shape of mat
    (rows, cols) = leaf_id_mat.shape 
    # calulate leaf_example_array
    leaf_ex_arr = clf.apply(ex)   
    # for each tree for each training example check if they share leaf node
    cnt = Counter() 
    for col in range(cols - 1):      # trees
        for row in range(rows - 1):  # example id:s
            if leaf_id_mat[row, col] == leaf_ex_arr[0, col]:
                #print("found match at: ", row) 
                cnt[row] += 1
    sorted_cnt = cnt.most_common()
    # filter result on wish_class
    sub_X = np.empty((0, np.size(X_train, 1)))  # define X matrix 
    sub_y = np.empty((0, 1))                    # define y array
    sub_cnt = np.empty((0,1))                   # define cnt array
    for ex, freq in sorted_cnt:
        #print("(Ex, freq): ", ex, freq)
        correct_class = (y_train[ex] == wish_class)
        #print("correct_class: ", correct_class)
        #print("classified_as_wish: ", classified_as_wish)
        if correct_class and classified_as_wish:
            t_pred_c = clf.predict([X_train[ex]])
            #print("t_pred_c: ", t_pred_c)
            if t_pred_c[0] == wish_class:
                #print("Found example of actual wished class which is classified as such with freq: ", freq)
                sub_X = np.vstack([sub_X, X_train[ex]]) 
                sub_y = np.vstack([sub_y, y_train[ex]])  
                sub_cnt = np.vstack([sub_cnt, freq]) 
        elif correct_class:
            #print("Found example of actual wished class with freq:", freq)
            sub_X = np.vstack([sub_X, X_train[ex]]) 
            sub_y = np.vstack([sub_y, y_train[ex]])
            sub_cnt = np.vstack([sub_cnt, freq])        
    return sub_cnt, sub_X, sub_y

# Random Forest Tweaking method (based on Lindgren et al. 2019, found in: https://github.com/tony-lind/Example-based-tweaking)
# Added by Anonymous Author
def rf_tweak(counterfactual):
    """
    Function that returns the Random Forest tweaking counterfactual with respect to instance of interest x
    """
    data = counterfactual.data
    x = counterfactual.ioi.normal_x
    x_label = counterfactual.ioi.label
    rf_model = counterfactual.rf_model

    start_time = time.time()
    x_pred = rf_model.predict(x.reshape(1,-1))
    if x_pred == 0:
        aim = 1
    else:
        aim = 0
    rt_cf_freq, rt_cf_array, rt_cf_label_array = random_forest_tweaking(rf_model, x.reshape(1,-1), aim, data.transformed_train_np, data.train_target) 
    if len(rt_cf_array) > 0:
        rt_cf = rt_cf_array[0]
    else:
        print(f'No Random Forest Tweaking solution found: Calculating NN Counterfactual')
        rt_cf = nn(counterfactual)
    end_time = time.time()
    rt_time = end_time - start_time
    return rt_cf, rt_time