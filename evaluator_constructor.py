import numpy as np
import pandas as pd
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist

class Evaluator():

    def __init__(self, data, method_str, type, split, justification_train_perc): # data, method_str, type, split
        self.data_name = data.name
        self.method_name = method_str
        self.distance_type = type
        self.continuous_split = split
        self.feat_type = data.feat_type
        self.feat_mutable = data.feat_mutable
        self.feat_directionality = data.feat_directionality
        self.feat_step = data.feat_step
        self.data_cols = data.processed_features
        self.perc = justification_train_perc
        self.x_dict, self.normal_x_dict = {}, {}
        self.normal_x_cf_dict, self.x_cf_dict = {}, {}
        self.proximity_dict, self.feasibility_dict, self.sparsity_dict, self.justification_dict, self.time_dict = {}, {}, {}, {}, {}

    def add_specific_x_data(self, counterfactual):
        """
        Method to add specific data from an instance x
        """
        x_cf = counterfactual.data.inverse(counterfactual.cf_method.normal_x_cf)
        self.x_dict[counterfactual.ioi.idx] = counterfactual.ioi.x
        self.normal_x_dict[counterfactual.ioi.idx] = counterfactual.ioi.normal_x
        self.x_cf_dict[counterfactual.ioi.idx] = x_cf
        feasibility = verify_feasibility(counterfactual.ioi.normal_x, counterfactual.cf_method.normal_x_cf, counterfactual.data)
        self.feasibility_dict[counterfactual.ioi.idx] = feasibility
        self.proximity_dict[counterfactual.ioi.idx] = distance_calculation(counterfactual.ioi.normal_x, counterfactual.cf_method.normal_x_cf, counterfactual.data, self.distance_type)
        self.sparsity_dict[counterfactual.ioi.idx] = sparsity(counterfactual.ioi.normal_x, counterfactual.cf_method.normal_x_cf, counterfactual.data)
        self.justification_dict[counterfactual.ioi.idx] = verify_justification(counterfactual.cf_method.normal_x_cf, counterfactual, self.perc, feasibility)
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
        x_df, y_df = pd.DataFrame(data=x, index=[0], columns=data.processed_features), pd.DataFrame(data=y, index=[0], columns=data.processed_features)
        x_original_df, y_original_df = pd.DataFrame(data=x_original, index=[0], columns=data.features), pd.DataFrame(data=y_original, index=[0], columns=data.features)
        x_continuous_df, y_continuous_df = x_df[data.ordinal + data.continuous], y_df[data.ordinal + data.continuous]
        x_continuous_np, y_continuous_np = x_continuous_df.to_numpy()[0], y_continuous_df.to_numpy()[0]
        x_categorical_df, y_categorical_df = x_original_df[data.bin_cat_enc_cols], y_original_df[data.bin_cat_enc_cols]
        x_categorical_np, y_categorical_np = x_categorical_df.to_numpy()[0], y_categorical_df.to_numpy()[0]
        L1_distance, L0_distance = L1(x_continuous_np, y_continuous_np), L0(x_categorical_np, y_categorical_np)
        return L1_distance, L0_distance
    
    def L1_L0_Linf(x, y, x_original, y_original, data, alpha=1/4, beta=1/4):
        """
        Calculates the distance used by Karimi et al.: Please see: http://proceedings.mlr.press/v108/karimi20a/karimi20a.pdf
        """
        J = len(data.continuous) + len(data.bin_cat_enc_cols)
        gamma = 1/((alpha + beta)*J)
        L1_distance, L0_distance = L1_L0(x, y, x_original, y_original, data)
        Linf_distance = Linf(x, y)
        return alpha*L0_distance + beta*L1_distance + gamma*Linf_distance

    x_original, y_original = data.inverse(x), data.inverse(y)
    if type == 'euclidean':
        distance = euclid(x, y)
    elif type == 'L1':
        distance = L1(x, y)
    elif type == 'L1_L0':
        n_con, n_cat = len(data.continuous), len(data.bin_cat_enc_cols)
        n = n_con + n_cat
        L1_distance, L0_distance = L1_L0(x, y, x_original, y_original, data)
        """
        Equation from Sharma et al.: Please see: https://arxiv.org/pdf/1905.07857.pdf
        """
        distance = (n_con/n)*L1_distance + (n_cat/n)*L0_distance
    elif type == 'L1_L0_Linf':
        distance = L1_L0_Linf(x, y, x_original, y_original, data)
    return distance

def verify_feasibility(x, cf, data):
    """
    Method that indicates whether the cf is a feasible counterfactual with respect to x, feature mutability and directionality
    """
    x = x[0]
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

def verify_justification(cf, counterfactual, perc, feasibility):
    """
    Method that verifies justification for any given cf, and a dataset.
    """
    data = counterfactual.data
    ioi = counterfactual.ioi
    model = counterfactual.model
    type = counterfactual.type
    split = counterfactual.split

    def nn_list(perc=0.2):
        """
        Method that gets the list of training observations labeled as cf-label with respect to the cf, ordered based on graph nodes size
        """
        train_true_label_data = data.transformed_train_np[data.train_target != ioi.label]
        train_prediction = model.model.predict(train_true_label_data)
        train_cf_label_prediction_data = train_true_label_data[train_prediction != ioi.label]
        sort_data_distance = []
        for i in range(train_cf_label_prediction_data.shape[0]):
            dist = distance_calculation(train_cf_label_prediction_data[i], cf, data)
            sort_data_distance.append((train_cf_label_prediction_data[i], dist, 1 - ioi.label))    
        sort_data_distance.sort(key=lambda x: x[1])
        sort_data_distance_possible_values = []
        for i in range(int(len(sort_data_distance)*perc)):
            print(f'Instance {i+1} of {int(len(sort_data_distance)*perc)} ({np.round((i+1)*100/int(len(sort_data_distance)*perc),2)}%)')
            possible_values = get_feat_possible_values(data, sort_data_distance[i][0], split)
            num_nodes = len(list(get_nodes(model, possible_values)))
            sort_data_distance_possible_values.append((sort_data_distance[i][0], possible_values, num_nodes, 1 - ioi.label))    
        sort_data_distance_possible_values.sort(key=lambda x: x[2])
        return sort_data_distance_possible_values

    def continuous_feat_values(i, min_val, max_val, data, split):
        """
        Method that defines how to discretize the continuous features
        """
        if split in ['2','5','10','20','50','100']:
            value = list(np.linspace(min_val, max_val, num = int(split) + 1, endpoint = True))
        elif split == 'train':
            sorted_feat_i = list(np.sort(data.transformed_train_np[:,i][(data.transformed_train_np[:,i] > min_val) & (data.transformed_train_np[:,i] < max_val)]))
            sorted_feat_i = min_val + sorted_feat_i + max_val 
            value = list(np.unique(sorted_feat_i))
        return value

    def get_feat_possible_values(data, point, split):
        """
        Method that obtains the features possible values
        """
        v = point - cf
        nonzero_index = list(np.nonzero(v)[0])
        feat_checked = []
        feat_possible_values = []
        for i in range(len(cf)):
            if i not in feat_checked:
                feat_i = data.processed_features[i]
                if feat_i in data.bin_enc_cols:
                    if i in nonzero_index:
                        value = [cf[i],point[i]]
                    else:
                        value = [cf[i]]
                    feat_checked.extend([i])
                elif feat_i in data.cat_enc_cols:
                    idx_cat_i = data.idx_cat_cols_dict[data.processed_features[i][:-2]]
                    nn_cat_idx = list(cf[idx_cat_i])
                    if any(item in idx_cat_i for item in nonzero_index):
                        ioi_cat_idx = list(point[idx_cat_i])
                        value = [nn_cat_idx, ioi_cat_idx]
                    else:
                        value = [nn_cat_idx]
                    feat_checked.extend(idx_cat_i)
                elif feat_i in data.ordinal:
                    if i in nonzero_index:
                        values_i = list(data.processed_feat_dist[data.processed_features[i]].keys())
                        max_val_i, min_val_i = max(cf[i],point[i]), min(cf[i],point[i])
                        value = [j for j in values_i if j <= max_val_i and j >= min_val_i]
                    else:
                        value = [cf[i]]
                    feat_checked.extend([i])
                elif feat_i in data.continuous:
                    if i in nonzero_index:
                        max_val_i, min_val_i = max(cf[i],point[i]), min(cf[i],point[i])
                        value = continuous_feat_values(i, min_val_i, max_val_i, data, split)
                    else:
                        value = [cf[i]]
                    feat_checked.extend([i])
                feat_possible_values.append(value)
        return feat_possible_values

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

    def get_nodes(model, possible_values):
        """
        Generator that contains all the nodes located in the space between the nn_cf and the normal_ioi (all possible, CF-labeled nodes)
        """
        permutations = product(*possible_values)
        for i in permutations:
            perm_i = make_array(i)
            if model.model.predict(perm_i.reshape(1, -1)) != ioi.label and not np.array_equal(perm_i, cf):
                yield perm_i

    def get_cost(data, model, possible_values, point, type):
        """
        Method that outputs the cost parameters required for optimization
        """
        C = {}
        C[1] = distance_calculation(cf, point, data, type)
        nodes = get_nodes(model, possible_values)
        ind = 2
        for i in nodes:
            C[ind] = distance_calculation(point, i, data, type)
            ind += 1
        return C

    def get_adjacency(data, possible_values, point, model):
        """
        Method that outputs the adjacency matrix required for optimization
        """
        toler = 0.00001
        nodes = [cf]
        nodes.extend(list(get_nodes(model, possible_values)))
        print(f'Getting adjacency: Nodes length: {len(nodes)}')
        A = tuplelist()
        for i in range(1, len(nodes) + 1):
            node_i = nodes[i - 1]
            # print(f'Getting adjacency: Started node edge verify: {i}. Progress ({np.round(100*(i)/(len(nodes)),2)}%)')
            for j in range(i + 1, len(nodes) + 1):
                node_j = nodes[j - 1]
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
                        max_val_i, min_val_i = max(cf[nonzero_index], point[nonzero_index]), min(cf[nonzero_index], point[nonzero_index])
                        values = continuous_feat_values(i, min_val_i, max_val_i, data, split)
                        values_idx = int(np.where(np.isclose(values, node_i[nonzero_index]))[0])
                        if values_idx > 0:
                            values_idx_inf = values_idx - 1
                        else:
                            values_idx_inf = 0
                        if values_idx < len(values) - 1:
                            values_idx_sup = values_idx + 1
                        else:
                            values_idx_sup = values_idx
                        close_node_j_values = [values[values_idx_inf], values[values_idx_sup]]
                        if np.isclose(node_j[nonzero_index], close_node_j_values, atol=toler).any():
                            A.append((i,j))
                    elif any(item in data.binary for item in feat_nonzero):
                        if np.isclose(np.abs(vector_ij[nonzero_index]),[0,1],atol=toler).any():
                            A.append((i,j))
        return A

    if feasibility:
        print(f'Preprocessing and ordering of training instances')
        train_nn_list = nn_list(perc)
        print(f'Number of training instances considered: {len(train_nn_list)}')
        for i in range(len(train_nn_list)):
            print(f'Verifying justification from train instance {i+1}')
            train_nn_i = train_nn_list[i][0]
            possible_values = train_nn_list[i][1]
            if np.array_equal(cf, train_nn_i):
                justifier = train_nn_i
                print(f'Justified by itself!')
                break
            # print(f'got possible values')
            cost = get_cost(data, model, possible_values, train_nn_i, type)
            # print(f'got costs')
            adjacency = get_adjacency(data, possible_values, train_nn_i, model)
            # print(f'got adjacency')
            opt_model_i = gp.Model(name='verify_justification_train_i')
            G = nx.DiGraph()
            G.add_edges_from(adjacency)
            set_I = list(cost.keys())
            x = opt_model_i.addVars(set_I, vtype=GRB.BINARY, obj=np.array(list(cost.values())), name='verification_cf')   # Function to optimize and x variables
            y = gp.tupledict()
            for (j,k) in G.edges:
                y[j,k] = opt_model_i.addVar(vtype=GRB.BINARY, name='Path')
            for v in G.nodes:
                if v > 1:
                    opt_model_i.addConstr(gp.quicksum(y[j,v] for j in G.predecessors(v)) - gp.quicksum(y[v,k] for k in G.successors(v)) == x[v])
                else:
                    opt_model_i.addConstr(gp.quicksum(y[j,v] for j in G.predecessors(v)) - gp.quicksum(y[v,k] for k in G.successors(v)) == -1)      
            # print(f'set all variables')
            opt_model_i.Params.LogToConsole = 0
            opt_model_i.optimize()
            nodes = [cf]
            nodes.extend(list(get_nodes(model, possible_values)))
            for i in cost.keys():
                if x[i].x > 0:
                    sol_x = nodes[i - 1]
            # sol_y = {}
            # for i,j in adjacency:
            #     if y[i,j].x > 0:
            #         sol_y[i,j] = y[i,j].x
            if np.array_equal(sol_x, train_nn_i):
                justifier = train_nn_i
                print(f'Justified through a path!')
                break
    else:
        justifier = None
    return justifier