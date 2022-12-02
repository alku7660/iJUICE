import numpy as np
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist
from evaluator_constructor import distance_calculation, verify_feasibility
from nnt import near_neigh, nn_for_juice
import time

class IJUICE:

    def __init__(self, counterfactual):
        self.normal_ioi = counterfactual.ioi.normal_x[0]
        self.ioi_label = counterfactual.ioi.label
        self.potential_justifiers = self.find_potential_justifiers(counterfactual)
        start_time = time.time()
        self.normal_x_cf, self.justifier = self.Ijuice(counterfactual)
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
            # if verify_feasibility(self.normal_ioi, potential_justifiers[i], counterfactual.data):
            dist = distance_calculation(potential_justifiers[i], self.normal_ioi, counterfactual.data, type=counterfactual.type)
            sort_potential_justifiers.append((potential_justifiers[i], dist))    
        sort_potential_justifiers.sort(key=lambda x: x[1])
        sort_potential_justifiers =  [i[0] for i in sort_potential_justifiers]
        sort_potential_justifiers = sort_potential_justifiers
        return sort_potential_justifiers

    # def Ijuice(self, counterfactual):
    #     """
    #     Improved JUICE generation method
    #     """
    #     self.feat_possible_values = self.get_feat_possible_values(counterfactual.data, counterfactual.split)
    #     justifier, _ = nn_for_juice(counterfactual)
    #     if justifier is not None:
    #         if counterfactual.model.model.predict(justifier.reshape(1, -1)) != self.ioi.label:
    #             self.C = self.get_cost(counterfactual.model, counterfactual.type) 
    #             self.A = self.get_adjacency(counterfactual.data, counterfactual.model, counterfactual.split)
    #             self.optimizer, normal_x_cf = self.do_optimize(counterfactual.model)
    #         else:
    #             print(f'Justifier (NN CF instance) is not a prediction counterfactual. Returning ground truth NN counterfactual as CF')
    #             normal_x_cf = justifier
    #     else:
    #         print(f'No justifier available: Returning NN counterfactual')
    #         normal_x_cf, _ = near_neigh(counterfactual)
    #         justifier = normal_x_cf
    #     return normal_x_cf, justifier
    
    def Ijuice(self, counterfactual):
        """
        Improved JUICE generation method
        """
        print(f'Obtained all potential justifiers: {len(self.potential_justifiers)}')
        self.pot_justifier_feat_possible_values = self.get_feat_possible_values(counterfactual.data, counterfactual.split)
        print(f'Obtained all posible feature values from potential justifiers')
        self.graph_nodes = self.get_graph_nodes(counterfactual.data, counterfactual.model)
        self.all_nodes = self.potential_justifiers + self.graph_nodes
        print(f'Obtained all posible nodes in the graph: {len(self.all_nodes)}')
        self.C = self.get_all_costs(counterfactual.data, counterfactual.type)
        print(f'Obtained all costs in the graph')
        self.F = self.get_all_feasibility(counterfactual.data)
        print(f'Obtained all feasibility in the graph')
        self.A = self.get_all_adjacency(counterfactual.data, counterfactual.model, counterfactual.split)
        print(f'Obtained adjacency matrix')
        if len(self.potential_justifiers) > 0:
            normal_x_cf, justifier = self.do_optimize_all(counterfactual.model)
        else:
            print(f'CF cannot be justified. Returning NN counterfactual')
            normal_x_cf, _ = nn_for_juice(counterfactual)
            justifier = normal_x_cf
        return normal_x_cf, justifier 

    def continuous_feat_values(self, i, min_val, max_val, data, split):
        """
        Method that defines how to discretize the continuous features
        """
        if split in ['2','5','10','20','50','100']:
            value = list(np.linspace(min_val, max_val, num = int(split) + 1, endpoint = True))
        elif split == 'train': # Most likely only using this, because the others require several divisions for each of the continuous features ranges
            sorted_feat_i = list(np.sort(data.transformed_train_np[:,i][(data.transformed_train_np[:,i] >= min_val) & (data.transformed_train_np[:,i] <= max_val)]))
            value = list(np.unique(sorted_feat_i))
        return value

    # def get_feat_possible_values(self, data, split):
    #     """
    #     Method that obtains the features possible values
    #     """
    #     v = self.normal_ioi - self.nn_cf
    #     nonzero_index = list(np.nonzero(v)[0])
    #     feat_checked = []
    #     feat_possible_values = []
    #     for i in range(len(self.normal_ioi)):
    #         if i not in feat_checked:
    #             feat_i = data.processed_features[i]
    #             if feat_i in data.bin_enc_cols:
    #                 if i in nonzero_index:
    #                     value = [self.nn_cf[i],self.normal_ioi[i]]
    #                 else:
    #                     value = [self.nn_cf[i]]
    #                 feat_checked.extend([i])
    #             elif feat_i in data.cat_enc_cols:
    #                 idx_cat_i = data.idx_cat_cols_dict[data.processed_features[i][:-2]]
    #                 nn_cat_idx = list(self.nn_cf[idx_cat_i])
    #                 if any(item in idx_cat_i for item in nonzero_index):
    #                     ioi_cat_idx = list(self.normal_ioi[idx_cat_i])
    #                     value = [nn_cat_idx,ioi_cat_idx]
    #                 else:
    #                     value = [nn_cat_idx]
    #                 feat_checked.extend(idx_cat_i)
    #             elif feat_i in data.ordinal:
    #                 if i in nonzero_index:
    #                     values_i = list(data.processed_feat_dist[data.processed_features[i]].keys())
    #                     max_val_i, min_val_i = max(self.normal_ioi[i],self.nn_cf[i]), min(self.normal_ioi[i],self.nn_cf[i])
    #                     value = [j for j in values_i if j <= max_val_i and j >= min_val_i]
    #                 else:
    #                     value = [self.nn_cf[i]]
    #                 feat_checked.extend([i])
    #             elif feat_i in data.continuous:
    #                 if i in nonzero_index:
    #                     max_val_i, min_val_i = max(self.normal_ioi[i],self.nn_cf[i]), min(self.normal_ioi[i],self.nn_cf[i])
    #                     value = self.continuous_feat_values(i, min_val_i, max_val_i, data, split)
    #                 else:
    #                     value = [self.nn_cf[i]]
    #                 feat_checked.extend([i])
    #             feat_possible_values.append(value)
    #     return feat_possible_values

    def get_feat_possible_values(self, data, split):
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
                            value = self.continuous_feat_values(i, min_val_i, max_val_i, data, split)
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

    # def get_nodes(self, model):
    #     """
    #     Generator that contains all the nodes located in the space between the nn_cf and the normal_ioi (all possible, CF-labeled nodes)
    #     """
    #     permutations = product(*self.feat_possible_values)
    #     for i in permutations:
    #         perm_i = self.make_array(i)
    #         if model.model.predict(perm_i.reshape(1, -1)) != self.ioi_label and not np.array_equal(perm_i, self.nn_cf):
    #             yield perm_i
    
    def get_graph_nodes(self, data, model):
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

    # def get_cost(self, model, type):
    #     """
    #     Method that outputs the cost parameters required for optimization
    #     """
    #     C = {}
    #     C[1] = distance_calculation(self.normal_ioi, self.nn_cf, type)
    #     nodes = self.get_nodes(model)
    #     ind = 2
    #     for i in nodes:
    #         C[ind] = distance_calculation(self.normal_ioi, i, type)
    #         ind += 1
    #     return C

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

    # def get_adjacency(self, data, model, split):
    #     """
    #     Method that outputs the adjacency matrix required for optimization
    #     """
    #     toler = 0.00001
    #     nodes = [self.nn_cf]
    #     nodes.extend(list(self.get_nodes(model)))
    #     A = tuplelist()
    #     for i in range(1, len(nodes) + 1):
    #         node_i = nodes[i - 1]
    #         for j in range(i + 1, len(nodes) + 1):
    #             node_j = nodes[j - 1]
    #             vector_ij = node_j - node_i
    #             nonzero_index = list(np.nonzero(vector_ij)[0])
    #             feat_nonzero = [data.processed_features[l] for l in nonzero_index]
    #             if len(nonzero_index) > 2:
    #                 continue
    #             elif len(nonzero_index) == 2:
    #                 if any(item in data.cat_enc_cols for item in feat_nonzero):
    #                     A.append((i,j))
    #             elif len(nonzero_index) == 1:
    #                 if any(item in data.ordinal for item in feat_nonzero):
    #                     if np.isclose(np.abs(vector_ij[nonzero_index]),data.feat_step[feat_nonzero],atol=toler).any():
    #                         A.append((i,j))
    #                 elif any(item in data.continuous for item in feat_nonzero):
    #                     max_val_i, min_val_i = max(self.normal_ioi[nonzero_index],self.nn_cf[nonzero_index]), min(self.normal_ioi[nonzero_index],self.nn_cf[nonzero_index])
    #                     values = self.continuous_feat_values(i, min_val_i, max_val_i, data, split)
    #                     close_node_j_values = [values[max(np.where(node_i[nonzero_index] > values))], values[min(np.where(node_i[nonzero_index] <= values))]]
    #                     if any(np.isclose(node_j[nonzero_index], close_node_j_values)):
    #                         A.append((i,j))
    #                 elif any(item in data.binary for item in feat_nonzero):
    #                     if np.isclose(np.abs(vector_ij[nonzero_index]),[0,1],atol=toler).any():
    #                         A.append((i,j))
    #     return A

    def get_all_adjacency(self, data, model, split):
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
                        values = self.continuous_feat_values(nonzero_index, min_val, max_val, data, split)
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

    # def do_optimize(self, model):
    #     """
    #     Method that finds iJUICE CF using an optimization package
    #     """
    #     opt_model = gp.Model(name='iJUICE')
    #     G = nx.DiGraph()
    #     G.add_edges_from(self.A)
    #     set_I = list(self.C.keys())   
    #     x = opt_model.addVars(set_I, vtype=GRB.BINARY, obj=np.array(list(self.C.values())), name='iJUICE_cf')   # Function to optimize and x variables
    #     y = gp.tupledict()
    #     for (i,j) in G.edges:
    #         y[i,j] = opt_model.addVar(vtype=GRB.BINARY, name='Path')
    #     for v in G.nodes:
    #         if v > 1:
    #             opt_model.addConstr(gp.quicksum(y[i,v] for i in G.predecessors(v)) - gp.quicksum(y[v,j] for j in G.successors(v)) == x[v])
    #         else:
    #             opt_model.addConstr(gp.quicksum(y[i,v] for i in G.predecessors(v)) - gp.quicksum(y[v,j] for j in G.successors(v)) == -1)      
    #     opt_model.optimize()
    #     nodes = [self.nn_cf]
    #     nodes.extend(list(self.get_nodes(model)))
    #     sol_y = {}
    #     for i in self.C.keys():
    #         if x[i].x > 0:
    #             sol_x = nodes[i - 1]
    #     # for i,j in self.A:
    #     #     if y[i,j].x > 0:
    #     #         sol_y[i,j] = y[i,j].x
    #     return opt_model, sol_x

    def do_optimize_all(self, model):
        """
        Method that finds iJUICE CF using an optimization package
        """
        def output_path(node, cf_node, path=[]):
            path.extend([node])
            if cf_node == node:
                return path
            new_node = [j for j in G.successors(node) if np.isclose(edge[node,j].x, 1)][0]
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
        CONSTRAINTS
        """
        len_justifiers = len(self.potential_justifiers)
        for (i,j) in G.edges:
            edge[i,j] = opt_model.addVar(vtype=GRB.BINARY, name='Path')
        for v in G.nodes:
            if v <= len_justifiers:
                opt_model.addConstr(gp.quicksum(edge[i,v] for i in G.predecessors(v)) - gp.quicksum(edge[v,j] for j in G.successors(v)) == -source[v]) # Source contraints
            else:
                opt_model.addConstr(gp.quicksum(edge[i,v] for i in G.predecessors(v)) - gp.quicksum(edge[v,j] for j in G.successors(v)) == cf[v]*source.sum()) # Sink constraints
                opt_model.addConstr(cf[v] <= self.F[v])
                opt_model.addConstr(source[v] == 0)
        opt_model.addConstr(source.sum() >= 1)
        opt_model.setObjective(cf.prod(self.C) - source.sum(), GRB.MINIMIZE)  # cf.prod(self.C) - source.sum()/len_justifiers
        
        """
        OPTIMIZATION AND RESULTS
        """
        opt_model.optimize()
        time.sleep(1)
        for i in self.C.keys():
            if cf[i].x > 0:
                sol_x = self.all_nodes[i - 1]
        print(f'Optimizer solution status: {opt_model.status}') # 1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'
        print(f'Solution:')
        justifiers = []
        for i in self.C.keys():
            if source[i].x > 0:
                justifiers.append(i)
        time.sleep(1)
        for i in self.C.keys():
            if cf[i].x > 0:
                print(f'cf({i}): {cf[i].x}')
                # print(f'Node {i}: {self.all_nodes[i - 1]}')
                print(f'Original IOI: {self.normal_ioi}')
                print(f'Euclidean Distance: {np.round(np.sqrt(np.sum((self.all_nodes[i - 1] - self.normal_ioi)**2)),3)}')
                cf_node_idx = i
        for i in justifiers:
            path = []
            print(f'Source {i} Path to CF: {output_path(i, cf_node_idx, path=path)}')
        time.sleep(1)
        return sol_x, justifiers

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