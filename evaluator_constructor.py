import numpy as np

class Evaluator():

    def __init__(self, data, cf_method_name):
        self.data_name = data.name
        self.cf_method = cf_method_name
        self.feat_type = data.feat_type
        self.feat_mutable = data.feat_mutable
        self.feat_directionality = data.feat_directionality
        self.feat_cost = data.feat_cost
        self.feat_step = data.feat_step
        self.data_cols = data.processed_features
        self.ioi_idx_dict, self.x_dict, self.normal_x_dict = {}, {}, {}
        self.normal_x_cf_dict, self.x_cf_dict = {}, {}
        self.proximity_dict, self.feasibility_dict, self.sparsity_dict, self.time_dict = {}, {}, {}, {}

    def add_specific_x_data(self, data, ioi, cf_method):
        """
        Method to add specific data from an instance x
        """
        x_cf = data.inverse(cf_method.normal_x_cf)
        self.ioi_idx_dict[ioi.idx] = ioi.idx
        self.x_dict[ioi.idx] = ioi.x
        self.normal_x_dict[ioi.idx] = ioi.normal_x
        self.x_cf_dict[ioi.idx] = x_cf
        self.proximity_dict[ioi.idx] = distance_calculation(ioi.x, cf_method.normal_x_cf, ioi.normal_x, x_cf)
        self.feasibility_dict[ioi.idx] = 

def distance_calculation(x, y, normal_x, normal_y, type='euclidean'):
    """
    Method that calculates the distance between two points. Default is 'euclidean'. Other types are 'L1', 'mixed_L1' and 'mixed_L1_Linf'
    """
    if type == 'euclidean':
        return np.sqrt(np.sum((x - y)**2))
    elif type == 'L1':
        return np.sum(np.abs(x - y))
    elif type == 'mixed_L1':
        return 1
    elif type == 'mixed_L1_Linf':
        return 1
    else:
        return 1