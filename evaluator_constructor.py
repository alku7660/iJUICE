import numpy as np
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist

class Evaluator():

    def __init__(self, counterfactual):
        self.data_name = counterfactual.data.name
        self.method_name = counterfactual.method_name
        self.distance_type = counterfactual.type
        self.continuous_split = counterfactual.split
        self.feat_type = counterfactual.data.feat_type
        self.feat_mutable = counterfactual.data.feat_mutable
        self.feat_directionality = counterfactual.data.feat_directionality
        self.feat_cost = counterfactual.data.feat_cost
        self.feat_step = counterfactual.data.feat_step
        self.data_cols = counterfactual.data.processed_features
        self.x_dict, self.normal_x_dict = {}, {}
        self.normal_x_cf_dict, self.x_cf_dict = {}, {}
        self.proximity_dict, self.feasibility_dict, self.sparsity_dict, self.justification_dict, self.time_dict = {}, {}, {}, {}, {}

    def add_specific_x_data(self, counterfactual):
        """
        Method to add specific data from an instance x
        """
        x_cf = counterfactual.data.inverse(self.cf_method.normal_x_cf)
        self.x_dict[counterfactual.ioi.idx] = counterfactual.ioi.x
        self.normal_x_dict[counterfactual.ioi.idx] = counterfactual.ioi.normal_x
        self.x_cf_dict[counterfactual.ioi.idx] = x_cf
        self.proximity_dict[counterfactual.ioi.idx] = distance_calculation(counterfactual.ioi.x, self.cf_method.normal_x_cf, counterfactual.data, self.distance_type)
        self.feasibility_dict[counterfactual.ioi.idx] = verify_feasibility(counterfactual.ioi.normal_x, self.cf_method.normal_x_cf, counterfactual.data)
        self.sparsity_dict[counterfactual.ioi.idx] = sparsity(counterfactual.ioi.normal_x, self.cf_method.normal_x_cf, counterfactual.data)
        self.justification_dict[counterfactual.ioi.idx] = verify_justification(self.cf_method.normal_x_cf, counterfactual)
        self.time_dict[counterfactual.ioi.idx] = counterfactual.cf_method.total_time

def distance_calculation(x, y, data, type='euclidean'):
    """
    Method that calculates the distance between two points. Default is 'euclidean'. Other types are 'L1', 'mixed_L1' and 'mixed_L1_Linf'
    """
    x_original, y_original = data.inverse(x), data.inverse(y)
    if type == 'euclidean':
        distance = np.sqrt(np.sum((x - y)**2))
    elif type == 'L1':
        distance = np.sum(np.abs(x - y))
    elif type == 'mixed_L1':
        distance = 1
    elif type == 'mixed_L1_Linf':
        distance = 1
    return distance

def verify_feasibility(x, cf, data):
    """
    Method that indicates whether the cf is a feasible counterfactual with respect to x, feature mutability and directionality
    """
    toler = 0.000001
    feasibility = True
    for i in range(len(data.feat_type)):
        if data.feat_type[i] == 'bin' or data.feat_type[i] == 'cat':
            if not np.isclose(cf[i], [0,1],atol=toler).any():
                feasibility = False
                break
        elif data.feat_type[i] == 'ord':
            possible_val = np.linspace(0,1,int(1/data.feat_step[i]+1),endpoint=True)
            if not np.isclose(cf[i],possible_val,atol=toler).any():
                feasibility = False
                break  
        else:
            if cf[i] < 0-toler or cf[i] > 1+toler:
                feasibility = False
                break
        vector = cf - x
        if data.feat_dir[i] == 0 and vector[i] != 0:
            feasibility = False
            break
        elif data.feat_dir[i] == 'pos' and vector[i] < 0:
            feasibility = False
            break
        elif data.feat_dir[i] == 'neg' and vector[i] > 0:
            feasibility = False
            break
    if not np.array_equal(x[np.where(data.feat_mutable == 0)],cf[np.where(data.feat_mutable == 0)]):
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
    split = counterfactual.split

    def nn_list():
        """
        Method that gets the list of nearest training observations labeled as cf-label with respect to the cf
        """
        train_true_label_data = data.transformed_train_np[data.train_target != ioi.label]
        train_prediction = model.model.predict(train_true_label_data)
        train_cf_label_prediction_data = train_true_label_data[train_prediction != ioi.label]
        sort_data_distance = []
        for i in range(train_cf_label_prediction_data.shape[0]):
            dist = distance_calculation(train_cf_label_prediction_data[i], cf)
            sort_data_distance.append((train_cf_label_prediction_data[i], dist, 1 - ioi.label))      
        sort_data_distance.sort(key=lambda x: x[1])
        return sort_data_distance

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
                        value = [nn_cat_idx,ioi_cat_idx]
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
        permutations = product(possible_values)
        for i in permutations:
            perm_i = make_array(i)
            if model.model.predict(perm_i.reshape(1, -1)) != ioi.label and not np.array_equal(perm_i, cf):
                yield perm_i

    def get_cost(model, possible_values, point, type):
        """
        Method that outputs the cost parameters required for optimization
        """
        C = {}
        C[1] = distance_calculation(cf, point, type)
        nodes = get_nodes(model, possible_values)
        ind = 2
        for i in nodes:
            C[ind] = distance_calculation(point, i, type)
            ind += 1
        return C

    def get_adjacency(data, possible_values, point, model):
        """
        Method that outputs the adjacency matrix required for optimization
        """
        toler = 0.00001
        nodes = [cf]
        nodes.extend(list(get_nodes(model, possible_values)))
        A = tuplelist()
        for i in range(1, len(nodes) + 1):
            node_i = nodes[i - 1]
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
                        if np.isclose(np.abs(vector_ij[nonzero_index]),data.feat_step[feat_nonzero], atol=toler).any():
                            A.append((i,j))
                    elif any(item in data.continuous for item in feat_nonzero):
                        max_val_i, min_val_i = max(cf[nonzero_index], point[nonzero_index]), min(cf[nonzero_index], point[nonzero_index])
                        values = continuous_feat_values(i, min_val_i, max_val_i, data, split)
                        close_node_j_values = [values[max(np.where(node_i[nonzero_index] > values))], values[min(np.where(node_i[nonzero_index] <= values))]]
                        if np.isclose(node_j[nonzero_index], close_node_j_values, atol=toler).any():
                            A.append((i,j))
                    elif any(item in data.binary for item in feat_nonzero):
                        if np.isclose(np.abs(vector_ij[nonzero_index]),[0,1],atol=toler).any():
                            A.append((i,j))
        return A

    justifier = None
    train_nn_list = nn_list()
    for i in train_nn_list:
        train_nn_i = train_nn_list[i]
        possible_values = get_feat_possible_values(data, train_nn_i, split)
        cost = get_cost(model, possible_values, train_nn_i, type)
        adjacency = get_adjacency(data, possible_values, train_nn_i, model)
        opt_model_i = gp.Model(name='verify_justification_train_i')
        G = nx.DiGraph()
        G.add_edges_from(adjacency)
        set_I = list(cost.keys())
        x = opt_model_i.addVars(set_I, vtype=GRB.BINARY, obj=np.array(list(cost.values())), name='verification_cf')   # Function to optimize and x variables
        y = gp.tupledict()
        for (i,j) in G.edges:
            y[i,j] = opt_model_i.addVar(vtype=GRB.BINARY, name='Path')
        for v in G.nodes:
            if v > 1:
                opt_model_i.addConstr(gp.quicksum(y[i,v] for i in G.predecessors(v)) - gp.quicksum(y[v,j] for j in G.successors(v)) == x[v])
            else:
                opt_model_i.addConstr(gp.quicksum(y[i,v] for i in G.predecessors(v)) - gp.quicksum(y[v,j] for j in G.successors(v)) == -1)      
        opt_model_i.optimize()
        nodes = [cf]
        nodes.extend(list(get_nodes(model)))
        for i in cost.keys():
            if x[i].x > 0:
                sol_x = nodes[i - 1]
        if np.equal(sol_x, train_nn_i):
            justifier = train_nn_i
            break
    return justifier