import numpy as np
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist
from ioi_constructor import distance_calculation

class Ijuice:

    def __init__(self, data, model, ioi, type='euclidean'):
        self.name = data.name
        self.normal_ioi = ioi.normal_x
        self.ioi_label = ioi.label
        self.nn_cf = self.nn(ioi, data, model)
        self.feat_possible_values = self.get_feat_possible_values(data)
        self.C = self.get_cost(model, type) 
        self.A = self.get_adjacency(data, model)
        self.optimizer, self.x, self.y = self.do_optimize()
    
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
            if feat_type[i] == 'bin' or feat_type[i] == 'cat':
                if not np.isclose(cf[i], [0,1],atol=toler).any():
                    feasibility = False
                    break
            elif feat_type[i] == 'ord':
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

    def nn(self, ioi, data, model):
        """
        Function that returns the nearest counterfactual with respect to instance of interest x
        """
        nn_cf = None
        for i in ioi.train_sorted:
            if i[2] != ioi.label and model.model.predict(i[0].reshape(1,-1)) != ioi.label and self.verify_feasibility(ioi.normal_x, i[0], data.feat_mutable, data.feat_type, data.feat_step) and not np.array_equal(ioi.normal_x, i[0]):
                nn_cf = i[0]
                break
        if nn_cf is None:
            print(f'NT could not find a feasible and counterfactual predicted CF (None output)')
            return nn_cf
        return nn_cf

    def get_feat_possible_values(self, data):
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
                        value = list(np.linspace(min_val_i, max_val_i, num = 101, endpoint = True))
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

    def get_adjacency(self, data, model):
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
                        list_values = [k[nonzero_index] for k in nodes]
                        min_value, max_value = np.min(list_values), np.max(list_values)
                        if np.less(np.abs(vector_ij[nonzero_index]),(max_value - min_value)/100 + toler):
                            A.append((i,j))
                    elif any(item in data.binary for item in feat_nonzero):
                        if np.isclose(np.abs(vector_ij[nonzero_index]),[0,1],atol=toler).any():
                            A.append((i,j))
        return A

    def do_optimize(self):
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
        return opt_model, x, y

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