import numpy as np
import pandas as pd
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist
from scipy.stats import norm

class Evaluator():

    def __init__(self, data, method_str, type, lagrange): # data, method_str, type, split
        self.data_name = data.name
        self.method_name = method_str
        self.distance_type = type
        self.lagrange = lagrange
        self.feat_type = data.feat_type
        self.feat_mutable = data.feat_mutable
        self.feat_directionality = data.feat_directionality
        self.feat_step = data.feat_step
        self.data_cols = data.processed_features
        self.x_dict, self.normal_x_dict = {}, {}
        self.normal_x_cf_dict, self.x_cf_dict = {}, {}
        self.proximity_dict, self.feasibility_dict, self.justifiers_dict, self.justifier_ratio, self.time_dict = {}, {}, {}, {}, {}

    def add_specific_x_data(self, counterfactual):
        """
        Method to add specific data from an instance x
        """
        if self.method_name == 'mace':
            x_cf_df = counterfactual.cf_method.normal_x_cf_df
            try:
                counterfactual.cf_method.normal_x_cf = counterfactual.data.transform_data(x_cf_df).to_numpy()[0]
            except:
                x_cf_df = counterfactual.data.inverse(x_cf_df, mace=True)
                counterfactual.cf_method.normal_x_cf = counterfactual.data.transform_data(x_cf_df).to_numpy()[0]
            x_cf = x_cf_df.to_numpy()
        else:
            x_cf = counterfactual.data.inverse(counterfactual.cf_method.normal_x_cf) 
        self.x_dict[counterfactual.ioi.idx] = counterfactual.ioi.x
        self.normal_x_dict[counterfactual.ioi.idx] = counterfactual.ioi.normal_x
        self.x_cf_dict[counterfactual.ioi.idx] = x_cf
        self.feasibility_dict[counterfactual.ioi.idx] = verify_feasibility(counterfactual.ioi.normal_x, counterfactual.cf_method.normal_x_cf, counterfactual.data)
        L2 = distance_calculation(counterfactual.ioi.normal_x, counterfactual.cf_method.normal_x_cf, counterfactual.data, 'euclidean')
        L1 = distance_calculation(counterfactual.ioi.normal_x, counterfactual.cf_method.normal_x_cf, counterfactual.data, 'L1')
        Linf = distance_calculation(counterfactual.ioi.normal_x, counterfactual.cf_method.normal_x_cf, counterfactual.data, 'L_inf')
        L1_L0 = distance_calculation(counterfactual.ioi.normal_x, counterfactual.cf_method.normal_x_cf, counterfactual.data, 'L1_L0')
        L1_L0_L_inf = distance_calculation(counterfactual.ioi.normal_x, counterfactual.cf_method.normal_x_cf, counterfactual.data, 'L1_L0_L_inf')
        prob = distance_calculation(counterfactual.ioi.normal_x, counterfactual.cf_method.normal_x_cf, counterfactual.data, 'prob')
        self.proximity_dict[counterfactual.ioi.idx] = {'euclidean':L2, 'L1':L1, 'L_inf':Linf, 'L1_L0':L1_L0, 'L1_L0_L_inf':L1_L0_L_inf, 'prob':prob}
        if self.method_name == 'ijuice':
            self.justifiers_dict[counterfactual.ioi.idx], self.justifier_ratio[counterfactual.ioi.idx] = counterfactual.cf_method.justifiers, counterfactual.cf_method.justifier_ratio
        else:
            self.justifiers_dict[counterfactual.ioi.idx], self.justifier_ratio[counterfactual.ioi.idx] = verify_justification(counterfactual.cf_method.normal_x_cf, counterfactual)
        self.time_dict[counterfactual.ioi.idx] = counterfactual.cf_method.run_time

def distance_calculation(x, y, data, type='euclidean'):
    """
    Method that calculates the distance between two points. Default is 'euclidean'. Other types are 'L1', 'mixed_L1' and 'mixed_L1_Linf'
    """
    def euclid(x, y):
        """
        Calculates the euclidean distance between the instances (inputs must be Numpy arrays)
        """
        return np.sqrt(np.sum((x - y)**2))

    def L1(x, y):
        """
        Calculates the L1-Norm distance between the instances (inputs must be Numpy arrays)
        """
        return np.sum(np.abs(x - y))

    def L0(x, y):
        """
        Calculates a simple matching distance between the features of the instances (pass only categortical features, inputs must be Numpy arrays)
        """
        return len(list(np.where(x != y)))

    def Linf(x, y):
        """
        Calculates the Linf distance
        """
        return np.max(np.abs(x - y))

    def L1_L0(x, y, x_original, y_original, data):
        """
        Calculates the distance components according to Sharma et al.: Please see: https://arxiv.org/pdf/1905.07857.pdf
        """
        x_df, y_df = pd.DataFrame(data=x.reshape(1, -1), index=[0], columns=data.processed_features), pd.DataFrame(data=y.reshape(1, -1), index=[0], columns=data.processed_features)
        x_original_df, y_original_df = pd.DataFrame(data=x_original, index=[0], columns=data.features), pd.DataFrame(data=y_original, index=[0], columns=data.features)
        x_continuous_df, y_continuous_df = x_df[data.ordinal + data.continuous], y_df[data.ordinal + data.continuous]
        x_continuous_np, y_continuous_np = x_continuous_df.to_numpy()[0], y_continuous_df.to_numpy()[0]
        x_categorical_df, y_categorical_df = x_original_df[data.binary + data.categorical], y_original_df[data.binary + data.categorical]
        x_categorical_np, y_categorical_np = x_categorical_df.to_numpy()[0], y_categorical_df.to_numpy()[0]
        L1_distance, L0_distance = L1(x_continuous_np, y_continuous_np), L0(x_categorical_np, y_categorical_np)
        return L1_distance, L0_distance
    
    def L1_L0_L_inf(x, y, x_original, y_original, data, alpha=1/4, beta=1/4):
        """
        Calculates the distance used by Karimi et al.: Please see: http://proceedings.mlr.press/v108/karimi20a/karimi20a.pdf
        """
        J = len(data.continuous) + len(data.bin_cat_enc_cols)
        gamma = 1/((alpha + beta)*J)
        L1_distance, L0_distance = L1_L0(x, y, x_original, y_original, data)
        Linf_distance = Linf(x, y)
        return alpha*L0_distance + beta*L1_distance + gamma*Linf_distance

    def max_percentile_shift(x_original, y_original, data):
        """
        Calculates the maximum percentile shift as a cost function between two instances
        """
        perc_shift_list = []
        x_original_df, y_original_df = pd.DataFrame(data=x_original, index=[0], columns=data.features), pd.DataFrame(data=y_original, index=[0], columns=data.features)
        for col in data.features:
            x_col_value = float(x_original_df[col].values)
            try:
                y_col_value = float(y_original_df[col].values)
            except:
                perc_shift_list.append(1)
                continue
            distribution = data.feat_dist[col]
            if x_col_value == y_col_value:
                continue
            else:
                if col in data.binary or col in data.categorical:
                    perc_shift = np.abs(distribution[x_col_value] - distribution[y_col_value])
                elif col in data.ordinal:
                    min_val, max_val = min(x_col_value, y_col_value), max(x_col_value, y_col_value)
                    values_range = [i for i in distribution.keys() if i >= min_val and i <= max_val]
                    values_range.sort()
                    prob_values = np.cumsum([distribution[val] for val in values_range])
                    try:
                        perc_shift = np.abs(prob_values[-1] - prob_values[0])
                    except:
                        perc_shift = 0
                elif col in data.continuous:
                    mean_val, std_val = distribution['mean'], distribution['std']
                    normalized_x = (x_col_value - mean_val)/std_val
                    normalized_y = (y_col_value - mean_val)/std_val
                    perc_shift = np.abs(norm.cdf(normalized_x) - norm.cdf(normalized_y))
            perc_shift_list.append(perc_shift)
        if len(perc_shift_list) == 0:
            value_to_return = 0
        else:
            value_to_return = max(perc_shift_list)
        return value_to_return

    x_original, y_original = data.inverse(x), data.inverse(y)
    if type == 'euclidean':
        distance = euclid(x, y)
    elif type == 'L1':
        distance = L1(x, y)
    elif type == 'L_inf':
        distance = Linf(x, y)
    elif type == 'L1_L0':
        n_con, n_cat = len(data.continuous), len(data.bin_cat_enc_cols)
        n = n_con + n_cat
        L1_distance, L0_distance = L1_L0(x, y, x_original, y_original, data)
        """
        Equation from Sharma et al.: Please see: https://arxiv.org/pdf/1905.07857.pdf
        """
        distance = (n_con/n)*L1_distance + (n_cat/n)*L0_distance
    elif type == 'L1_L0_L_inf':
        distance = L1_L0_L_inf(x, y, x_original, y_original, data)
    elif type == 'prob':
        distance = max_percentile_shift(x_original, y_original, data)
    return distance

def verify_feasibility(x, cf, data):
    """
    Method that indicates whether the cf is a feasible counterfactual with respect to x, feature mutability and directionality
    """
    # x = x[0]
    toler = 0.000001
    feasibility = True
    for i in range(len(data.feat_type)):
        if data.feat_type[i] == 'bin' or data.feat_type[i] == 'cat':
            if not np.isclose(cf[i], [0,1], atol=toler).any():
                feasibility = False
                break
        elif data.feat_type[i] == 'ord':
            possible_val = np.linspace(0, 1, int(1/data.feat_step[i]+1), endpoint=True)
            if not np.isclose(cf[i], possible_val, atol=toler).any():
                feasibility = False
                break  
        else:
            if cf[i] < 0-toler or cf[i] > 1+toler:
                feasibility = False
                break
        vector = cf - x
        if data.feat_directionality[i] == 0 and vector[i] != 0:
            feasibility = False
            break
        elif data.feat_directionality[i] == 'pos' and vector[i] < 0:
            feasibility = False
            break
        elif data.feat_directionality[i] == 'neg' and vector[i] > 0:
            feasibility = False
            break
    if not np.array_equal(x[np.where(data.feat_mutable == 0)], cf[np.where(data.feat_mutable == 0)]):
        feasibility = False
    return feasibility

def sparsity(x, cf, data):
    """
    Function that calculates sparsity for a given counterfactual according to x
    Sparsity: 1 - the fraction of features changed in the cf. Takes the value of 1 if the number of changed features is 1.
    Input data: The data object with the feature information regarding plausibility, mutability, directionality
    Input x: The (could be normalized) instance of interest
    Input cf: The (could be normalized) counterfactual instance        
    """
    unchanged_features = np.sum(np.equal(x,cf))
    categories_feat_changed = data.feat_cat[np.where(np.equal(x,cf) == False)[0]]
    len_categories_feat_changed_unique = len([i for i in np.unique(categories_feat_changed) if 'cat' in i])
    unchanged_features += len_categories_feat_changed_unique
    n_changed = len(x) - unchanged_features
    if n_changed == 1:
        cf_sparsity = 1.000
    else:
        cf_sparsity = np.round_(1 - n_changed/len(x),3)
    return cf_sparsity

def verify_justification(cf, counterfactual):
    """
    Method that verifies justification for any given cf, and a dataset.
    """
    data = counterfactual.data
    ioi = counterfactual.ioi
    model = counterfactual.model
    type = counterfactual.type
    lagrange = counterfactual.lagrange

    def potential_justifiers_list(cf):
        """
        Obtains all the training instances sorted by distance to the CF
        """
        train_np = counterfactual.data.transformed_train_np
        train_target = counterfactual.data.train_target
        train_pred = counterfactual.model.model.predict(train_np)
        potential_justifiers = train_np[train_target != ioi.label] #[(train_target != ioi.label) & (train_pred != ioi.label)]
        sort_potential_justifiers = []
        for i in range(potential_justifiers.shape[0]):
            dist = distance_calculation(potential_justifiers[i], cf, counterfactual.data, type=counterfactual.type)
            sort_potential_justifiers.append((potential_justifiers[i], dist))
        sort_potential_justifiers.sort(key=lambda x: x[1])
        sort_potential_justifiers = [i[0] for i in sort_potential_justifiers]
        return sort_potential_justifiers

    def nn_list(potential_justifiers, cf):
        """
        Method that gets the list of training observations labeled as cf-label with respect to the cf, ordered based on graph nodes size
        """
        if len(potential_justifiers) > 100:
            potential_justifiers = potential_justifiers[:100]
        permutations_potential_justifiers = []
        for i in range(len(potential_justifiers)):
            possible_feat_values_justifier_i = get_feat_possible_values(data, cf, [potential_justifiers[i]])[0]
            len_permutations = len(list(product(*possible_feat_values_justifier_i)))
            permutations_potential_justifiers.append((potential_justifiers[i], len_permutations))
            # print(f'Justifier {i+1}: Length permutations: {len_permutations}')
        permutations_potential_justifiers.sort(key=lambda x: x[1])
        permutations_potential_justifiers = [i[0] for i in permutations_potential_justifiers]
        if len(permutations_potential_justifiers) > 10:
            permutations_potential_justifiers = permutations_potential_justifiers[:10]
        return permutations_potential_justifiers

    def continuous_feat_values(i, min_val, max_val, data):
        """
        Method that defines how to discretize the continuous features
        """
        # if split in ['2','5','10','20','50','100']:
        #     value = list(np.linspace(min_val, max_val, num = int(split) + 1, endpoint = True))
        # elif split == 'train': # Most likely only using this, because the others require several divisions for each of the continuous features ranges
        
        sorted_feat_i = list(np.sort(data.transformed_train_np[:,i][(data.transformed_train_np[:,i] >= min_val) & (data.transformed_train_np[:,i] <= max_val)]))
        value = list(np.unique(sorted_feat_i))
        if len(value) <= 10:
            if min_val not in value:
                value = [min_val] + value
            if max_val not in value:
                value = value + [max_val]
            return value
        else:
            mean_val, std_val = np.mean(data.transformed_train_np[:,i]), np.std(data.transformed_train_np[:,i])
            percentiles_range = list(np.linspace(0, 1, 11))
            value = []
            for perc in percentiles_range:
                value.append(norm.ppf(perc, loc=mean_val, scale=std_val))
            value = [val for val in value if val >= min_val and val <= max_val]
            if min_val not in value:
                value = [min_val] + value
            if max_val not in value:
                value = value + [max_val]
        return value

    def get_feat_possible_values(data, obj, points):
        """
        Method that obtains the features possible values
        """
        try:
            normal_x = obj.normal_x
        except:
            normal_x = obj
        pot_justifier_feat_possible_values = {}
        for k in range(len(points)):
            potential_justifier_k = points[k]
            v = normal_x - potential_justifier_k
            nonzero_index = list(np.nonzero(v)[0])
            feat_checked = []
            feat_possible_values = []
            for i in range(len(normal_x)):
                if i not in feat_checked:
                    feat_i = data.processed_features[i]
                    if feat_i in data.bin_enc_cols:
                        if i in nonzero_index:
                            value = [potential_justifier_k[i], normal_x[i]]
                        else:
                            value = [potential_justifier_k[i]]
                        feat_checked.extend([i])
                    elif feat_i in data.cat_enc_cols:
                        idx_cat_i = data.idx_cat_cols_dict[feat_i[:-4]]
                        nn_cat_idx = list(potential_justifier_k[idx_cat_i])
                        if any(item in idx_cat_i for item in nonzero_index):
                            ioi_cat_idx = list(normal_x[idx_cat_i])
                            value = [nn_cat_idx, ioi_cat_idx]
                        else:
                            value = [nn_cat_idx]
                        feat_checked.extend(idx_cat_i)
                    elif feat_i in data.ordinal:
                        if i in nonzero_index:
                            values_i = list(data.processed_feat_dist[feat_i].keys())
                            max_val_i, min_val_i = max(normal_x[i], potential_justifier_k[i]), min(normal_x[i], potential_justifier_k[i])
                            value = [j for j in values_i if j <= max_val_i and j >= min_val_i]
                        else:
                            value = [potential_justifier_k[i]]
                        feat_checked.extend([i])
                    elif feat_i in data.continuous:
                        if i in nonzero_index:
                            max_val_i, min_val_i = max(normal_x[i], potential_justifier_k[i]), min(normal_x[i], potential_justifier_k[i])
                            value = continuous_feat_values(i, min_val_i, max_val_i, data)
                        else:
                            value = [potential_justifier_k[i]]
                        feat_checked.extend([i])
                    feat_possible_values.append(value)
            pot_justifier_feat_possible_values[k] = feat_possible_values
        return pot_justifier_feat_possible_values

    def make_array(i):
        """
        Method that transforms a generator instance into array  
        """
        list_i = list(i)
        new_list = []
        for j in list_i:
            if isinstance(j,list):
                new_list.extend([k for k in j])
            else:
                new_list.extend([j])
        return np.array(new_list)

    def get_graph_nodes(model, filter_pot_justifiers, train_instances, feat_possible_values):
        """
        Generator that contains all the nodes located in the space between the potential justifiers and the normal_ioi (all possible, CF-labeled nodes)
        """
        graph_nodes = []
        for k in range(len(filter_pot_justifiers)):
            feat_possible_values_k = feat_possible_values[k]
            permutations = product(*feat_possible_values_k)
            for i in permutations:
                perm_i = make_array(i)
                if model.model.predict(perm_i.reshape(1, -1)) != ioi.label and \
                    not any(np.array_equal(perm_i, x) for x in graph_nodes) and \
                    not any(np.array_equal(perm_i, x) for x in filter_pot_justifiers):
                    graph_nodes.append(perm_i)
        return graph_nodes

    def get_adjacency(data, all_nodes):
        """
        Method that outputs the adjacency matrix required for optimization
        """
        toler = 0.00001
        all_nodes = np.array(all_nodes)
        A = tuplelist()
        for i in range(1, len(all_nodes) + 1):
            node_i = all_nodes[i - 1]
            for j in range(i + 1, len(all_nodes) + 1):
                node_j = all_nodes[j - 1]
                vector_ij = node_j - node_i
                nonzero_index = list(np.nonzero(vector_ij)[0])
                feat_nonzero = [data.processed_features[l] for l in nonzero_index]
                if len(nonzero_index) > 2:
                    continue
                elif len(nonzero_index) == 2:
                    if any(item in data.cat_enc_cols for item in feat_nonzero):
                        A.append((i,j))
                elif len(nonzero_index) == 1:
                    if any(item in data.ordinal for item in feat_nonzero):
                        if np.isclose(np.abs(vector_ij[nonzero_index]), data.feat_step[feat_nonzero], atol=toler).any():
                            A.append((i,j))
                    elif any(item in data.continuous for item in feat_nonzero):
                        max_val_i, min_val_i = float(max(cf[nonzero_index], max(all_nodes[:,nonzero_index]))), float(min(cf[nonzero_index], min(all_nodes[:,nonzero_index])))
                        values = continuous_feat_values(nonzero_index, min_val_i, max_val_i, data)
                        try:
                            values_idx = int(np.where(np.isclose(values, node_i[nonzero_index]))[0])
                            if values_idx > 0:
                                values_idx_inf = values_idx - 1
                            else:
                                values_idx_inf = 0
                            if values_idx < len(values) - 1:
                                values_idx_sup = values_idx + 1
                            else:
                                values_idx_sup = values_idx
                        except:
                            if node_i[nonzero_index] < values[0]:
                                values_idx_inf, values_idx_sup = 0, 0
                            elif node_i[nonzero_index] > values[-1]:
                                values_idx_inf, values_idx_sup = len(values) - 1, len(values) - 1
                            else:
                                for k in range(len(values) - 1):
                                    if node_i[nonzero_index] <= values[k+1] and node_i[nonzero_index] >= values[k]:
                                        values_idx_inf, values_idx_sup = k, k+1        
                        close_node_j_values = [values[values_idx_inf], values[values_idx_sup]]
                        if node_j[nonzero_index] <= close_node_j_values[1] and node_j[nonzero_index] >= close_node_j_values[0]:
                            A.append((i,j))
                    elif any(item in data.binary for item in feat_nonzero):
                        if np.isclose(np.abs(vector_ij[nonzero_index]),[0,1],atol=toler).any():
                            A.append((i,j))
        return A

    print(f'Sorting training instances w.r.t. CF')
    pot_justifiers = potential_justifiers_list(cf)
    print(f'Selecting closest to CF based on distance / permutations')
    filter_pot_justifiers = nn_list(pot_justifiers, cf)
    print(f'Number of training instances considered: {len(filter_pot_justifiers)}')
    train_nn_feat_possible_values = get_feat_possible_values(counterfactual.data, cf, filter_pot_justifiers)
    print(f'Obtained all possible feature values from potential justifiers')
    graph_nodes = get_graph_nodes(model, filter_pot_justifiers, pot_justifiers, train_nn_feat_possible_values)
    all_nodes = filter_pot_justifiers + graph_nodes
    justifiers = []
    try:
        cf_index = [i for i in range(1, len(all_nodes)+1) if np.array_equal(all_nodes[i-1], cf)][0]
    except:
        cf_index = -1
    if cf_index != -1:
        if cf_index <= len(filter_pot_justifiers) + 1:
            justifiers.append(cf)
        range_justifier_nodes = range(1, len(filter_pot_justifiers) + 1)
        print(f'Obtained closest graphs nodes to CF: {len(all_nodes)}')
        adjacency = get_adjacency(data, all_nodes)
        G = nx.DiGraph()
        G.add_edges_from(adjacency)
        if G.has_node(cf_index):
            for i in range_justifier_nodes:
                if G.has_node(i):
                    if nx.has_path(G, i, cf_index):
                        if not any(np.array_equal(all_nodes[i - 1], x) for x in justifiers):
                            justifiers.append(all_nodes[i - 1])
        else:
            print(f'The CF is not found in the graph of nodes for justification verification. May be justified by itself.')
    else:
        print(f'The CF is not found in the graph of nodes for justification verification. The instance is not justifiable.')
    if counterfactual.method == 'juice':
        justifier_juice = counterfactual.cf_method.justifier
    else:
        justifier_juice = None
    if justifier_juice is not None:
        justifiers.append(justifier_juice)
    justifier_ratio = len(justifiers)/len(filter_pot_justifiers)
    if justifier_ratio > 1:
        justifier_ratio = 1
    print(f'Evaluated Justifier Ratio: {np.round(justifier_ratio*100, 3)}%')
    return justifiers, justifier_ratio