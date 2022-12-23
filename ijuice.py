import numpy as np
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist
from evaluator_constructor import distance_calculation, verify_feasibility
from nnt import nn_for_juice
import time
from scipy.stats import norm

class IJUICE:

    def __init__(self, counterfactual):
        self.normal_ioi = counterfactual.ioi.normal_x[0]
        self.ioi_label = counterfactual.ioi.label
        self.lagrange = counterfactual.lagrange
        self.potential_justifiers = self.find_potential_justifiers(counterfactual)
        start_time = time.time()
        self.normal_x_cf, self.justifiers, self.justifier_ratio = self.Ijuice(counterfactual)
        end_time = time.time()
        self.run_time = end_time - start_time

    def find_potential_justifiers(self, counterfactual):
        """
        Finds the set of training observations belonging to, and predicted as, the counterfactual class
        """
        train_np = counterfactual.data.transformed_train_np
        train_target = counterfactual.data.train_target
        train_pred = counterfactual.model.model.predict(train_np)
        potential_justifiers = train_np[(train_target != self.ioi_label) & (train_pred != self.ioi_label)]
        sort_potential_justifiers = []
        for i in range(potential_justifiers.shape[0]):
            dist = distance_calculation(potential_justifiers[i], self.normal_ioi, counterfactual.data, type=counterfactual.type)
            sort_potential_justifiers.append((potential_justifiers[i], dist))    
        sort_potential_justifiers.sort(key=lambda x: x[1])
        sort_potential_justifiers = [i[0] for i in sort_potential_justifiers]
        sort_potential_justifiers = sort_potential_justifiers[:30]
        return sort_potential_justifiers
    
    def Ijuice(self, counterfactual):
        """
        Improved JUICE generation method
        """
        print(f'Obtained all potential justifiers: {len(self.potential_justifiers)}')
        self.pot_justifier_feat_possible_values = self.get_feat_possible_values(counterfactual.data)
        print(f'Obtained all possible feature values from potential justifiers')
        self.graph_nodes = self.get_graph_nodes(counterfactual.model)
        self.all_nodes = self.potential_justifiers + self.graph_nodes
        print(f'Obtained all possible nodes in the graph: {len(self.all_nodes)}')
        self.C = self.get_all_costs(counterfactual.data, counterfactual.type)
        print(f'Obtained all costs in the graph')
        self.F = self.get_all_feasibility(counterfactual.data)
        print(f'Obtained all feasibility in the graph')
        self.A = self.get_all_adjacency(counterfactual.data, counterfactual.model)
        print(f'Obtained adjacency matrix')
        if len(self.potential_justifiers) > 0:
            normal_x_cf, justifiers, justifier_ratio = self.do_optimize_all()
        else:
            print(f'CF cannot be justified. Returning NN counterfactual')
            normal_x_cf, _ = nn_for_juice(counterfactual)
            justifiers = normal_x_cf
            justifier_ratio = 1/len(self.potential_justifiers)
        return normal_x_cf, justifiers, justifier_ratio 

    def continuous_feat_values(self, i, min_val, max_val, data):
        """
        Method that defines how to discretize the continuous features
        """
        # if split in ['2','5','10','20','50','100']:
        #     value = list(np.linspace(min_val, max_val, num = int(split) + 1, endpoint = True))
        # elif split == 'train': # Most likely only using this, because the others require several divisions for each of the continuous features ranges
        
        # sorted_feat_i = list(np.sort(data.transformed_train_np[:,i][(data.transformed_train_np[:,i] >= min_val) & (data.transformed_train_np[:,i] <= max_val)]))
        # value = list(np.unique(sorted_feat_i))
        
        sorted_feat_i = list(np.sort(data.transformed_train_np[:,i][(data.transformed_train_np[:,i] >= min_val) & (data.transformed_train_np[:,i] <= max_val)]))
        value = list(np.unique(sorted_feat_i))
        if len(value) <= 100:
            if min_val not in value:
                value = [min_val] + value
            if max_val not in value:
                value = value + [max_val]
            return value
        else:
            mean_val, std_val = np.mean(data.transformed_train_np[:,i]), np.std(data.transformed_train_np[:,i])
            percentiles_range = list(np.linspace(0, 1, 101))
            value = []
            for perc in percentiles_range:
                value.append(norm.ppf(perc, loc=mean_val, scale=std_val))
            value = [val for val in value if val >= min_val and val <= max_val]
            if min_val not in value:
                value = [min_val] + value
            if max_val not in value:
                value = value + [max_val]
        return value

    def get_feat_possible_values(self, data):
        """
        Method that obtains the features possible values
        """
        pot_justifier_feat_possible_values = {}
        for k in range(len(self.potential_justifiers)):
            potential_justifier_k = self.potential_justifiers[k]
            v = self.normal_ioi - potential_justifier_k
            nonzero_index = list(np.nonzero(v)[0])
            feat_checked = []
            feat_possible_values = []
            for i in range(len(self.normal_ioi)):
                if i not in feat_checked:
                    feat_i = data.processed_features[i]
                    if feat_i in data.bin_enc_cols:
                        if i in nonzero_index:
                            value = [potential_justifier_k[i], self.normal_ioi[i]]
                        else:
                            value = [potential_justifier_k[i]]
                        feat_checked.extend([i])
                    elif feat_i in data.cat_enc_cols:
                        idx_cat_i = data.idx_cat_cols_dict[data.processed_features[i][:-2]]
                        nn_cat_idx = list(potential_justifier_k[idx_cat_i])
                        if any(item in idx_cat_i for item in nonzero_index):
                            ioi_cat_idx = list(self.normal_ioi[idx_cat_i])
                            value = [nn_cat_idx, ioi_cat_idx]
                        else:
                            value = [nn_cat_idx]
                        feat_checked.extend(idx_cat_i)
                    elif feat_i in data.ordinal:
                        if i in nonzero_index:
                            values_i = list(data.processed_feat_dist[data.processed_features[i]].keys())
                            max_val_i, min_val_i = max(self.normal_ioi[i], potential_justifier_k[i]), min(self.normal_ioi[i], potential_justifier_k[i])
                            value = [j for j in values_i if j <= max_val_i and j >= min_val_i]
                        else:
                            value = [potential_justifier_k[i]]
                        feat_checked.extend([i])
                    elif feat_i in data.continuous:
                        if i in nonzero_index:
                            max_val_i, min_val_i = max(self.normal_ioi[i], potential_justifier_k[i]), min(self.normal_ioi[i], potential_justifier_k[i])
                            value = self.continuous_feat_values(i, min_val_i, max_val_i, data)
                        else:
                            value = [potential_justifier_k[i]]
                        feat_checked.extend([i])
                    feat_possible_values.append(value)
            pot_justifier_feat_possible_values[k] = feat_possible_values
        return pot_justifier_feat_possible_values

    def make_array(self, i):
        """
        Method that transforms a generator instance into array  
        """
        list_i = list(i)
        new_list = []
        for j in list_i:
            if isinstance(j, list):
                new_list.extend([k for k in j])
            else:
                new_list.extend([j])
        return np.array(new_list)
    
    def get_graph_nodes(self, model):
        """
        Generator that contains all the nodes located in the space between the potential justifiers and the normal_ioi (all possible, CF-labeled nodes)
        """
        graph_nodes = []
        for k in range(len(self.potential_justifiers)):
            feat_possible_values_k = self.pot_justifier_feat_possible_values[k]
            permutations = product(*feat_possible_values_k)
            for i in permutations:
                perm_i = self.make_array(i)                     # 
                if model.model.predict(perm_i.reshape(1, -1)) != self.ioi_label and \
                    not any(np.array_equal(perm_i, x) for x in graph_nodes) and \
                    not any(np.array_equal(perm_i, x) for x in self.potential_justifiers):
                    graph_nodes.append(perm_i)
        return graph_nodes

    def get_all_costs(self, data, type):
        """
        Method that outputs the cost parameters required for optimization
        """
        C = {}
        for k in range(1, len(self.all_nodes)+1):
            node_k = self.all_nodes[k-1]
            C[k] = distance_calculation(self.normal_ioi, node_k, data, type)
        return C

    def get_all_feasibility(self, data):
        """
        Outputs the counterfactual feasibility parameter for all graph nodes (including the potential justifiers) 
        """
        F = {}
        for k in range(1, len(self.all_nodes)+1):
            node_k = self.all_nodes[k-1]
            F[k] = verify_feasibility(self.normal_ioi, node_k, data)
        return F

    def get_all_adjacency(self, data, model):
        """
        Method that outputs the adjacency matrix required for optimization
        """
        toler = 0.00001
        nodes = self.all_nodes
        justifiers_array = np.array(self.potential_justifiers)
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
                        if np.isclose(np.abs(vector_ij[nonzero_index]), data.feat_step[feat_nonzero], atol=toler).any():
                            A.append((i,j))
                    elif any(item in data.continuous for item in feat_nonzero):
                        max_val, min_val = max(self.normal_ioi[nonzero_index], max(justifiers_array[:,nonzero_index])), min(self.normal_ioi[nonzero_index], min(justifiers_array[:,nonzero_index]))
                        values = self.continuous_feat_values(nonzero_index, min_val, max_val, data)
                        value_node_i_idx = int(np.where(np.isclose(values, node_i[nonzero_index]))[0])
                        if value_node_i_idx > 0:
                            value_node_i_idx_inf = value_node_i_idx - 1
                        else:
                            value_node_i_idx_inf = 0
                        if value_node_i_idx < len(values) - 1:
                            value_node_i_idx_sup = value_node_i_idx + 1
                        else:
                            value_node_i_idx_sup = value_node_i_idx
                        close_node_j_values = [values[value_node_i_idx_inf], values[value_node_i_idx_sup]]
                        if any(np.isclose(node_j[nonzero_index], close_node_j_values)):
                            A.append((i,j))
                    elif any(item in data.binary for item in feat_nonzero):
                        if np.isclose(np.abs(vector_ij[nonzero_index]), [0,1], atol=toler).any():
                            A.append((i,j))
        return A

    def do_optimize_all(self):
        """
        Method that finds iJUICE CF using an optimization package
        """
        def output_path(node, cf_node, path=[]):
            """
            Function that prints the connection paths from a justifier towards the found CF
            """
            path.extend([node])
            if cf_node == node:
                return path
            new_node = [j for j in G.successors(node) if edge[node,j].x >= 1][0]
            return output_path(new_node, cf_node, path)

        """
        MODEL
        """
        opt_model = gp.Model(name='iJUICE')
        G = nx.DiGraph()
        G.add_edges_from(self.A)
        
        """
        SETS AND VARIABLES
        """
        set_I = list(self.C.keys())   
        cf = opt_model.addVars(set_I, vtype=GRB.BINARY, name='Counterfactual')   # Node chosen as destination
        source = opt_model.addVars(set_I, vtype=GRB.BINARY, name='Justifiers')       # Nodes chosen as sources (justifier points)
        edge = gp.tupledict()
        
        """
        CONSTRAINTS AND OBJECTIVE
        """
        len_justifiers = len(self.potential_justifiers)
        for (i,j) in G.edges:
            edge[i,j] = opt_model.addVar(vtype=GRB.INTEGER, name='Path')
        for v in G.nodes:
            if v <= len_justifiers:
                opt_model.addConstr(gp.quicksum(edge[i,v] for i in G.predecessors(v)) - gp.quicksum(edge[v,j] for j in G.successors(v)) == -source[v]) # Source contraints
            else:
                opt_model.addConstr(gp.quicksum(edge[i,v] for i in G.predecessors(v)) - gp.quicksum(edge[v,j] for j in G.successors(v)) == cf[v]*source.sum()) # Sink constraints
                opt_model.addConstr(cf[v] <= self.F[v])
                opt_model.addConstr(source[v] == 0)
        opt_model.addConstr(source.sum() >= 1)
        opt_model.setObjective(cf.prod(self.C)*self.lagrange - source.sum()/len_justifiers*(1-self.lagrange), GRB.MINIMIZE)  # cf.prod(self.C) - source.sum()/len_justifiers
        
        """
        OPTIMIZATION AND RESULTS
        """
        opt_model.optimize()
        time.sleep(0.5)
        for i in self.C.keys():
            if cf[i].x > 0:
                sol_x = self.all_nodes[i - 1]
        print(f'Optimizer solution status: {opt_model.status}') # 1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'
        print(f'Solution:')
        justifiers = []
        for i in self.C.keys():
            if source[i].x > 0:
                justifiers.append(i)
        print(f'Number of justifiers: {len(justifiers)}')
        time.sleep(0.5)
        for i in self.C.keys():
            if cf[i].x > 0:
                print(f'cf({i}): {cf[i].x}')
                print(f'Node {i}: {self.all_nodes[i - 1]}')
                print(f'Original IOI: {self.normal_ioi}')
                print(f'Euclidean Distance: {np.round(np.sqrt(np.sum((self.all_nodes[i - 1] - self.normal_ioi)**2)),3)}')
                cf_node_idx = i
        time.sleep(0.5)
        for i in justifiers:
            path = []
            print(f'Source {i} Path to CF: {output_path(i, cf_node_idx, path=path)}')
            time.sleep(0.25)
        justifier_ratio = len(justifiers)/len(self.potential_justifiers)
        print(f'Justifier Ratio: {np.round(justifier_ratio, 2)}')
        return sol_x, justifiers, justifier_ratio