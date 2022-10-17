import numpy as np
from itertools import product
from itertools import filterfalse

class Ijuice:

    def __init__(self, data, ioi, model):
        self.name = data.name
        self.ioi_idx = ioi.idx
        self.ioi = ioi.x
        self.normal_ioi = ioi.normal_ioi
        self.ioi_label = ioi.label
        self.nn_cf = self.nn(self.normal_ioi, self.ioi_label, data, model)
        self.C = self.get_cost(data, model) 
        self.A = self.get_adjacency_matrix(data, model)
    
    def verify_feasibility(self, x, cf, mutable_feat, feat_type, feat_step):
        """
        Method that indicates whether the cf is a feasible counterfactual with respect to x and the feature mutability
        Input x: Instance of interest
        Input cf: Counterfactual to be evaluated
        Input mutable_feat: Vector indicating mutability of the features of x
        Input feat_type: Type of the features used
        Input feat_step: Feature plausible change step size    
        Output: Boolean value indicating whether cf is a feasible counterfactual with regards to x and the feature mutability vector
        """
        toler = 0.000001
        feasibility = True
        for i in range(len(feat_type)):
            if feat_type[i] == 'bin':
                if not np.isclose(cf[i], [0,1],atol=toler).any():
                    feasibility = False
                    break
            elif feat_type[i] == 'num-ord':
                possible_val = np.linspace(0,1,int(1/feat_step[i]+1),endpoint=True)
                if not np.isclose(cf[i],possible_val,atol=toler).any():
                    feasibility = False
                    break  
            else:
                if cf[i] < 0-toler or cf[i] > 1+toler:
                    feasibility = False
                    break
        if not np.array_equal(x[np.where(mutable_feat == 0)],cf[np.where(mutable_feat == 0)]):
            feasibility = False
        return feasibility

    def nn(self, x, x_label, data, model):
        """
        Function that returns the nearest counterfactual with respect to instance of interest x
        Input x: Instance of interest
        Input x_label: Label of instance of interest x
        Input data: Dataset object
        Input model: Classifier trained
        Output nn_cf: Minimum observable counterfactual to the instance of interest x
        """
        nn_cf = None
        for i in data.train_sorted:
            if i[2] != x_label and model.predict(i[0].reshape(1,-1)) != x_label and self.verify_feasibility(x,i[0],data.feat_mutable,data.feat_type,data.feat_step) and not np.array_equal(x,i[0]):
                nn_cf = i[0]
                break
        if nn_cf is None:
            print(f'NT could not find a feasible and counterfactual predicted CF (None output)')
            return nn_cf
        return nn_cf

    def get_nodes(self, data, model):
        """
        Generator that contains all the nodes located in the space between the nn_cf and the normal_ioi (all possible, CF-labeled nodes)
        """
        v = self.normal_ioi - self.nn_cf
        nonzero_index = np.nonzero(v)
        feat_checked = []
        feat_possible_values = []
        for i in nonzero_index:
            if i not in feat_checked:
                if i in data.bin_enc_cols:
                    value = [0,1]
                    feat_checked.extend(i)
                elif i in data.cat_enc_cols:
                    idx_cat_i = data.idx_cat_cols_dict[data.processed_features[i][:-2]]
                    ioi_cat_idx = self.normal_ioi[idx_cat_i]
                    nn_cat_idx = self.nn_cf[idx_cat_i]
                    value = [ioi_cat_idx,nn_cat_idx]
                    feat_checked.extend(idx_cat_i)
                elif i in data.ordinal:
                    values_i = list(data.processed_feat_dist[data.processed_features[i]].keys())
                    max_val_i, min_val_i = max(self.normal_ioi[i],self.nn_cf[i]), min(self.normal_ioi[i],self.nn_cf[i])
                    value = [j for j in values_i if j <= max_val_i and j >= min_val_i]
                    feat_checked.extend(i)
                elif i in data.continuous:
                    max_val_i, min_val_i = max(self.normal_ioi[i],self.nn_cf[i]), min(self.normal_ioi[i],self.nn_cf[i])
                    value = list(np.linspace(min_val_i, max_val_i, num = 100, endpoint = True))
                    feat_checked.extend(i)
                feat_possible_values.append(value)
        # permutations = product(*feat_possible_values)
        # for i in permutations:
        #     if model.predict(np.array(i).reshape(1, -1)) != self.ioi_label:
        #         yield i
        permutations = filterfalse(lambda x: model.predict(x) == self.ioi_label, product(*feat_possible_values))
        for i in permutations:
            yield i

    def get_cost(self, data, model):
        """
        Method that outputs the cost parameters required for optimization
        """
        C = []
        nodes = self.get_nodes(data, model)
        for node in nodes:
            C.extend(distance_calculation(self.normal_ioi, node))
        return C

    def get_adjacency_matrix(self, data, model):
        """
        Method that outputs the adjacency matrix required for optimization
        """
        toler = 0.000001
        nodes = list(self.get_nodes(data, model))
        A = np.zeros(shape=(len(nodes),len(nodes)))
        for i in range(len(nodes)):
            node_i = nodes[i]
            for j in range(len(nodes)):
                node_j = nodes[j]
                vector_ij = node_j - node_i
                nonzero_index = np.nonzero(vector_ij)
                if len(nonzero_index) > 2:
                    continue
                elif len(nonzero_index) == 2:
                    if any(map(lambda x: x in data.cat_enc_cols, nonzero_index)):
                        A[i,j] = 1
                elif len(nonzero_index) == 1:
                    if nonzero_index in data.ordinal:
                        if np.isclose(np.abs(vector_ij[nonzero_index]),data.feat_step[nonzero_index],atol=toler).any():
                            A[i,j] = 1
                    elif nonzero_index in data.continuous:
                        list_values = [k[nonzero_index] for k in nodes]
                        min_value, max_value = np.min(list_values), np.max(list_values)
                        if np.isclose(np.abs(vector_ij[nonzero_index]),(max_value - min_value)/100,atol=toler):
                            A[i,j] = 1
        return A


def distance_calculation(x, y, type='Euclidean'):
    """
    Method that calculates the distance between two points. Default is Euclidean.
    """
    if type == 'Euclidean':
        return np.sqrt(np.sum((x - y)**2))