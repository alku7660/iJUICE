"""
JUstIfied CounterFactual Explanations (JUICE)
"""

"""
Imports
"""

import time
import numpy as np
from nt import nn, nn_for_juice
from itertools import permutations
from evaluator_constructor import distance_calculation

class Juice:

    def __init__(self, counterfactual):
        self.normal_ioi = counterfactual.ioi.normal_x
        self.ioi_label = counterfactual.ioi.label
        start_time = time.time()
        self.normal_x_cf, self.justifier = self.JUICE(counterfactual)
        end_time = time.time()
        self.total_time = end_time - start_time

    def JUICE(self, counterfactual):
        """
        Justified NN counterfactual generation method:
        """
        justifier, _ = nn_for_juice(counterfactual)
        if justifier is not None:
            if counterfactual.model.model.predict(justifier.reshape(1, -1)) != self.ioi.label:
                normal_x_cf, justified = justified_search(counterfactual.ioi, justifier, counterfactual.model, counterfactual.data), 1
            else:
                print(f'Justifier (NN CF instance) is not a prediction counterfactual. Returning ground truth NN counterfactual as CF')
                normal_x_cf = justifier
        else:
            print(f'No justifier available: Returning NN counterfactual')
            normal_x_cf, _ = nn(counterfactual)
            justifier = normal_x_cf
        return normal_x_cf, justifier

def verify_diff_label(label, model, v):
    """
    Function that verifies if the label given does not match that of the model predicted on instance v
    Input label: Label to be compared
    Input model: Model to be used on instance v
    Input v: Instance to be checked on same label
    Output different: Boolean indicating whether labels are the same or not
    """
    different = False
    v = v.reshape(1, -1)
    label_v = model.predict(v)
    if label_v != label:
        different = True
    return different


def justified_search(x,x_label,nn_cf,model,data,priority):
    """
    Search for instances justified by t (train instance), closer to instance x
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input nn_cf: Nearest neighbor training instance
    Input model: Trained model to verify connectedness between t and x, verifying justification
    Input data: Dataset object
    Input priority: String variable indicating whether to focus on proximity or sparsity
    Output closest_to_x: Instance that is justified by t
    """

    def get_indices_vector(x,closest_to_x,f_type_str=None):
        """
        Sub method that gets indices of the features to be changed, for any method
        Input closest_to_x: Closest instance to x that is justified by t
        Input f_type_str: Feature type string
        Output vector: Difference vector between x and closest_to_x so far
        Output closest_to_x_dist: Distance from closest_to_x to x
        Ooutput diff_index: Different indices between x and closest_to_x
        """
        vector = x - closest_to_x
        closest_to_x_cost = distance_calculation(x,closest_to_x)
        directionality_condition = ((vector < 0) & (data.feat_dir == 'pos') | (vector > 0) & (data.feat_dir == 'neg') | (vector != 0) & (data.feat_dir == 'any'))
        if f_type_str is not None:
            diff_index = np.where(directionality_condition & (data.feat_type == f_type_str) & (data.feat_mutable == 1))[0].tolist()
        else:
            diff_index = np.where(directionality_condition & (data.feat_mutable == 1))[0].tolist()
        return vector, closest_to_x_cost, diff_index 

    def order_by_cost(perm_list,len_diff_index):
        """
        Sub method that orders the numerical features by change cost with respect to x
        Input perm_list: List of indices permutation
        Input len_diff_index: Lenght of the different indices vector
        Output diff_index_sort: Numerical indices ordered according to cost
        """
        perm_feat_cost_list = []
        positions = np.array(range(len_diff_index))+1
        positions = np.sort(positions)[::-1]
        for i in range(len(perm_list)):
            bin_perm_feat_cost = 0
            for j in range(len(perm_list[i])):
                bin_perm_feat_cost += data.feat_cost.iloc[perm_list[i][j]]*positions[j]
            perm_feat_cost_list.append((perm_list[i],bin_perm_feat_cost))      
        perm_feat_cost_list.sort(key=lambda x: x[1])
        perm_list_sorted = [k[0] for k in perm_feat_cost_list]
        return perm_list_sorted

    def sparse_optimize(close_to_x_list):
        """
        Method that selects the best sparse counterfactual and brings it closer to x
        Input close_to_x_list: List of sparse counterfactuals
        Output closest_to_x: Best of the sparse counterfactuals to x
        """
        tuple_close_to_x_cost = [(i[0],i[1],distance_calculation(x, i[0])) for i in close_to_x_list]
        tuple_close_to_x_cost.sort(key=lambda x: x[2])
        closest_to_x = tuple_close_to_x_cost[0][0]
        closest_to_x_sparse_idx = tuple_close_to_x_cost[0][1]
        prio = 'proximity'
        closest_to_x = binary_search(x,closest_to_x,'bin',prio)
        closest_to_x = numerical_search(x,closest_to_x,'num-ord',prio)
        closest_to_x = numerical_search(x,closest_to_x,'num-con',prio)
        return (closest_to_x,closest_to_x_sparse_idx)

    def perturbation_cf(instance,j,direction,step_size):
        """
        Method that calculates the vector change in binary or numerical search and applies correlations changes if requested, verifying its values
        Input instance: Current feature vector to be changed
        Input j: Index of the feature of interest to be changed
        Input direction: Direction of change of the feature of interest
        Input step_size: Magnitude of change of the feature of interest
        Output instance: Changed instance
        """
        instance[j] += direction*step_size
        if instance[j] < 0 or instance[j] > 1:
            instance[j] -= direction*step_size
        return instance

    def categorical_perturbation(instance,j,permutation,vector):
        """
        Method that creates a valid perturbation using only the categorical features that are different between x and a cf, with the 1 sum restriction
        Input instance: Instance to be changed in categorical features
        Input j: Index of the feature of interest to be changed
        Input permutation: The current oder of different binary indices to be tried
        Input vector: Vector of changes between features among x and a cf
        Input data.feat_cat: Categorical groups to which each feature belongs to
        Output instance: Changed instance (only categorical variables changed) 
        """
        cat_group = data.feat_cat.iloc[j]
        idx_in_group = [idx for idx in range(len(data.feat_cat)) if data.feat_cat.iloc[idx] == cat_group]
        vector_group = [vector[idx] if idx in idx_in_group else 0 for idx in range(len(vector))]
        instance += vector_group
        if verify_diff_label(x_label,model,instance):
            result_instance = instance
        else:
            instance -= vector_group
            result_instance = instance
        permutation_remaining = [idx for idx in permutation if idx in idx_in_group and idx != j]
        return result_instance, permutation_remaining

    def binary_search(x,closest_to_x,f_type_str,priority):
        """
        Method that searches for instances closer to x by modifying only binary features
        Input closest_to_x: Closest instance to x that is justified by t
        Input f_type_str: Feature type string
        Input priority: Whether it is proximity or sparsity (added to allow search in sparsity mode)
        Output closest_to_x: Updated closest instance to x that is justified by t
        """

        vector, closest_to_x_cost, bin_diff_index = get_indices_vector(x,closest_to_x,f_type_str)
        len_bin_diff_index = len(bin_diff_index)
        perm_list = list(permutations(bin_diff_index,len_bin_diff_index))

        if len(perm_list) > 0:
            perm_list = order_by_cost(perm_list,len_bin_diff_index)
        found = 0
        original_closest_to_x = np.copy(closest_to_x)
        for i in perm_list:
            v = np.copy(original_closest_to_x)
            counter = 0
            idx_same_cat = None
            for j in i:
                v_old = np.copy(v)
                if 'cat' in data.feat_cat[j] and idx_same_cat is None:
                    v, idx_same_cat = categorical_perturbation(v,j,i,vector)
                elif idx_same_cat is not None:
                    if j in idx_same_cat:
                        continue
                else:
                    v = perturbation_cf(v,j,np.sign(vector[j]),np.abs(vector[j]))
                if np.array_equal(v,v_old):
                    break
                if priority == 'proximity':
                    if verify_diff_label(x_label,model,v) and distance_calculation(x, v) < closest_to_x_cost:
                        closest_to_x = np.copy(v)
                        closest_to_x_cost = distance_calculation(x, closest_to_x)
                    else:
                        break
                elif priority == 'sparsity':
                    if verify_diff_label(x_label,model,v):
                        counter+=1
                        if counter == len(perm_list):
                            closest_to_x = np.copy(v)
                            found = 1
                    else:
                        break
            if found:
                break
        return closest_to_x

    def numerical_search(x,closest_to_x,f_type_str,priority):
        """
        Search for instances closer to x by modifying only binary features
        Input closest_to_x: Closest instance to x that is justified by t
        Input f_type_str: Feature type string
        Input priority: Whether it is proximity or sparsity (added to allow search in sparsity mode)
        Output closest_to_x: Updated closest instance to x that is justified by t
        """

        def direction_step():
            """
            Method that outputs direction and step size for each feature depending on the type
            Output direc_j: Direction of movement in feature j
            Output step_j: Step size of movement in feature j
            """
            if f_type_str == 'num-ord':
                    direc_j = np.sign(vector[j])
            elif f_type_str == 'num-con':
                    direc_j = vector[j]
            step_j = data.feat_step.iloc[j]
            return direc_j, step_j

        vector, closest_to_x_cost, num_diff_index = get_indices_vector(x,closest_to_x,f_type_str)
        len_num_diff_index = len(num_diff_index)
        perm_list = list(permutations(num_diff_index,len_num_diff_index))
        
        if len(perm_list) > 0:
            perm_list = order_by_cost(perm_list,len_num_diff_index)
        original_closest_to_x = np.copy(closest_to_x)
        for i in perm_list:
            counter_viable = 0
            counter_feat_tried = 0
            v = np.copy(original_closest_to_x)
            for j in i:
                unviable = False
                if priority == 'sparsity' and counter_viable < counter_feat_tried or unviable:
                    break
                direc_j, step_j = direction_step()
                not_close = True
                while not_close:
                    v_old = np.copy(v)
                    v = perturbation_cf(v,j,direc_j,step_j)
                    if np.array_equal(v,v_old):
                        break
                    if priority == 'proximity':
                        if verify_diff_label(x_label,model,v):
                            if distance_calculation(x, v) < closest_to_x_cost:
                                closest_to_x = np.copy(v)
                                closest_to_x_cost = distance_calculation(x,closest_to_x,data.feat_cost)
                        else:
                            unviable = True
                            break
                        if np.isclose(np.abs(x[j] - v[j]),0,rtol=0.000001):
                            v[j] = x[j]
                            not_close = False
                    elif priority == 'sparsity':
                        if verify_diff_label(x_label,model,v):
                            if np.isclose(np.abs(x[j] - v[j]),0,rtol=0.000001):
                                v[j] = x[j]
                                counter_viable+=1
                        else:
                            break
                        if counter_viable == len(i):
                            closest_to_x = np.copy(v)
                            not_close = False
                        if np.isclose(np.abs(x[j] - v[j]),0,rtol=0.000001):
                            not_close = False
                counter_feat_tried += 1
            if priority == 'sparsity' and counter_viable == len(i):
                break
        return closest_to_x

    # Must be run in this order: (1) Binary search, (2) Ordinal search (3) Continuous search
    closest_to_x = binary_search(x,nn_cf,'bin')
    closest_to_x = numerical_search(x,closest_to_x,'num-ord')
    closest_to_x = numerical_search(x,closest_to_x,'num-con')
    return closest_to_x