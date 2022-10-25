import numpy as np
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist
from evaluator_constructor import distance_calculation, verify_feasibility

class Ijuice:

    def __init__(self, data, model, ioi, type='euclidean', split='100'):
        self.name = data.name
        self.ioi = ioi.x
        self.normal_ioi = ioi.normal_x
        self.ioi_label = ioi.label
        self.feat_possible_values = self.get_feat_possible_values(data, split)
        self.nn_cf = self.nn(ioi, data, model)
        self.C = self.get_cost(model, type) 
        self.A = self.get_adjacency(data, model, split)
        self.optimizer, self.normal_x_cf, self.sol_y = self.do_optimize(model)
        self.x_cf = data.inverse(self.normal_x_cf)

    def nn(self, ioi, data, model):
        """
        Function that returns the nearest counterfactual with respect to instance of interest x
        """
        nn_cf = None
        for i in ioi.train_sorted:
            if i[2] != ioi.label and model.model.predict(i[0].reshape(1,-1)) != ioi.label and verify_feasibility(ioi.normal_x, i[0], data) and not np.array_equal(ioi.normal_x, i[0]):
                nn_cf = i[0]
                break
        if nn_cf is None:
            print(f'NT could not find a feasible and counterfactual predicted CF (None output)')
            return nn_cf
        return nn_cf

    def continuous_feat_values(self, i, min_val, max_val, data, split):
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

    def get_feat_possible_values(self, data, split):
        """
        Method that obtains the features possible values
        """
        v = self.normal_ioi - self.nn_cf
        nonzero_index = list(np.nonzero(v)[0])
        feat_checked = []
        feat_possible_values = []
        for i in range(len(self.normal_ioi)):
            if i not in feat_checked:
                feat_i = data.processed_features[i]
                if feat_i in data.bin_enc_cols:
                    if i in nonzero_index:
                        value = [self.nn_cf[i],self.normal_ioi[i]]
                    else:
                        value = [self.nn_cf[i]]
                    feat_checked.extend([i])
                elif feat_i in data.cat_enc_cols:
                    idx_cat_i = data.idx_cat_cols_dict[data.processed_features[i][:-2]]
                    nn_cat_idx = list(self.nn_cf[idx_cat_i])
                    if any(item in idx_cat_i for item in nonzero_index):
                        ioi_cat_idx = list(self.normal_ioi[idx_cat_i])
                        value = [nn_cat_idx,ioi_cat_idx]
                    else:
                        value = [nn_cat_idx]
                    feat_checked.extend(idx_cat_i)
                elif feat_i in data.ordinal:
                    if i in nonzero_index:
                        values_i = list(data.processed_feat_dist[data.processed_features[i]].keys())
                        max_val_i, min_val_i = max(self.normal_ioi[i],self.nn_cf[i]), min(self.normal_ioi[i],self.nn_cf[i])
                        value = [j for j in values_i if j <= max_val_i and j >= min_val_i]
                    else:
                        value = [self.nn_cf[i]]
                    feat_checked.extend([i])
                elif feat_i in data.continuous:
                    if i in nonzero_index:
                        max_val_i, min_val_i = max(self.normal_ioi[i],self.nn_cf[i]), min(self.normal_ioi[i],self.nn_cf[i])
                        value = self.continuous_feat_values(i, min_val_i, max_val_i, data, split)
                    else:
                        value = [self.nn_cf[i]]
                    feat_checked.extend([i])
                feat_possible_values.append(value)
        return feat_possible_values

    def make_array(self, i):
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

    def get_nodes(self, model):
        """
        Generator that contains all the nodes located in the space between the nn_cf and the normal_ioi (all possible, CF-labeled nodes)
        """
        permutations = product(*self.feat_possible_values)
        for i in permutations:
            perm_i = self.make_array(i)
            if model.model.predict(perm_i.reshape(1, -1)) != self.ioi_label and not np.array_equal(perm_i,self.nn_cf):
                yield perm_i

    def get_cost(self, model, type):
        """
        Method that outputs the cost parameters required for optimization
        """
        C = {}
        C[1] = distance_calculation(self.normal_ioi, self.nn_cf, type)
        nodes = self.get_nodes(model)
        ind = 2
        for i in nodes:
            C[ind] = distance_calculation(self.normal_ioi, i, type)
            ind += 1
        return C

    def get_adjacency(self, data, model, split):
        """
        Method that outputs the adjacency matrix required for optimization
        """
        toler = 0.00001
        nodes = [self.nn_cf]
        nodes.extend(list(self.get_nodes(model)))
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
                        if np.isclose(np.abs(vector_ij[nonzero_index]),data.feat_step[feat_nonzero],atol=toler).any():
                            A.append((i,j))
                    elif any(item in data.continuous for item in feat_nonzero):
                        max_val_i, min_val_i = max(self.normal_ioi[nonzero_index],self.nn_cf[nonzero_index]), min(self.normal_ioi[nonzero_index],self.nn_cf[nonzero_index])
                        values = self.continuous_feat_values(i, min_val_i, max_val_i, data, split)
                        close_node_j_values = [values[max(np.where(node_i[nonzero_index] > values))], values[min(np.where(node_i[nonzero_index] <= values))]]
                        if any(np.isclose(node_j[nonzero_index], close_node_j_values)):
                            A.append((i,j))
                    elif any(item in data.binary for item in feat_nonzero):
                        if np.isclose(np.abs(vector_ij[nonzero_index]),[0,1],atol=toler).any():
                            A.append((i,j))
        return A

    def do_optimize(self, model):
        """
        Method that finds iJUICE CF using an optimization package
        """
        opt_model = gp.Model(name='iJUICE')
        G = nx.DiGraph()
        G.add_edges_from(self.A)
        set_I = list(self.C.keys())   
        x = opt_model.addVars(set_I, vtype=GRB.BINARY, obj=np.array(list(self.C.values())), name='iJUICE_cf')   # Function to optimize and x variables
        y = gp.tupledict()
        for (i,j) in G.edges:
            y[i,j] = opt_model.addVar(vtype=GRB.BINARY, name='Path')
        for v in G.nodes:
            if v > 1:
                opt_model.addConstr(gp.quicksum(y[i,v] for i in G.predecessors(v)) - gp.quicksum(y[v,j] for j in G.successors(v)) == x[v])
            else:
                opt_model.addConstr(gp.quicksum(y[i,v] for i in G.predecessors(v)) - gp.quicksum(y[v,j] for j in G.successors(v)) == -1)      
        opt_model.optimize()

        nodes = [self.nn_cf]
        nodes.extend(list(self.get_nodes(model)))
        sol_y = {}
        for i in self.C.keys():
            if x[i].x > 0:
                sol_x = nodes[i - 1]
        for i,j in self.A:
            if y[i,j].x > 0:
                sol_y[i,j] = y[i,j].x
        return opt_model, sol_x, sol_y

    # def do_optimize(self):
    #     """
    #     Method that finds iJUICE CF using an optimization package
    #     """
    #     opt_model = gp.Model(name='iJUICE')
    #     set_I = list(self.C.keys())
    #     x = opt_model.addVars(set_I, vtype=GRB.BINARY, obj=self.C, name='iJUICE_cf')   # Function to optimize and x variables
    #     y = opt_model.addVars(self.A, vtype=GRB.BINARY, name='Path') # y variables
    #     for i in set_I: 
    #         opt_model.addConstr(sum(y[i,j] for i,j in self.A.select(i,'*')) - sum(y[j,i] for j,i in self.A.select('*',i)) == (1 if i == 1 else -x[i]), f'Network {i}') # Network constraints
    #         opt_model.addConstr(sum(y[j,i] for j,i in self.A.select('*',i)) <= 1, f'Entry {i}') # Single entry per node constraint
    #         opt_model.addConstr(sum(y[i,j] for i,j in self.A.select(i,'*')) <= 1, f'Exit {i}') # Single exit per node constraint
    #     opt_model.addConstr(sum(x) == 1, 'Single CF')  # Single CF constraint
    #     opt_model.optimize()
    #     return opt_model, x, y