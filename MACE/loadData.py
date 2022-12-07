"""
Model-Agnostic Counterfactual Explanations (MACE)
Original authors implementation: Please see https://github.com/amirhk/mace
"""

import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from sklearn.model_selection import train_test_split
from address import dataset_dir

VALID_ATTRIBUTE_DATA_TYPES = { \
    'numeric-int', \
    'numeric-real', \
    'binary', \
    'categorical', \
    'sub-categorical', \
    'ordinal', \
    'sub-ordinal'}
VALID_ATTRIBUTE_NODE_TYPES = { \
    'meta', \
    'input', \
    'output'}
VALID_ACTIONABILITY_TYPES = { \
    'none', \
    'any', \
    'same-or-increase', \
    'same-or-decrease'}
VALID_MUTABILITY_TYPES = { \
    True, \
    False}

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

class Dataset(object):

    def __init__(self, data_frame, attributes, is_one_hot, dataset_name):
        self.dataset_name = dataset_name
        self.is_one_hot = is_one_hot
        attributes_long = attributes
        data_frame_long = data_frame
        self.data_frame_long = data_frame_long # i.e., data_frame is indexed by attr_name_long
        self.attributes_long = attributes_long # i.e., attributes is indexed by attr_name_long
        attributes_kurz = dict((attributes[key].attr_name_kurz, value) for (key, value) in attributes_long.items())
        data_frame_kurz = copy.deepcopy(data_frame_long)
        data_frame_kurz.columns = self.getAllAttributeNames('kurz')
        self.data_frame_kurz = data_frame_kurz # i.e., data_frame is indexed by attr_name_kurz
        self.attributes_kurz = attributes_kurz # i.e., attributes is indexed by attr_name_kurz

        # assert that data_frame and attributes match on variable names (long)
        assert len(np.setdiff1d(data_frame.columns.values, np.array(self.getAllAttributeNames('long')))) == 0

        for attr_name in np.setdiff1d(self.getInputAttributeNames('long'), self.getRealBasedAttributeNames('long'), ):
            unique_values = np.unique(data_frame_long[attr_name].to_numpy())
            # all non-numerical-real values should be integer or {0,1}
        for value in unique_values:
            assert value == np.floor(value)
        if is_one_hot and attributes_long[attr_name].attr_type != 'numeric-int': # binary, sub-categorical, sub-ordinal
            try:
                assert \
                    np.array_equal(unique_values, [0,1]) or \
                    np.array_equal(unique_values, [1,2]) or \
                    np.array_equal(unique_values, [1]) # the first sub-ordinal attribute is always 1
                    # race (binary) in compass is encoded as {1,2}
            except:
                print(f'Assertion error: values not valid in {attr_name}')

        self.assertSiblingsShareAttributes('long')
        self.assertSiblingsShareAttributes('kurz')
    
    def getAttributeNames(self, allowed_node_types, long_or_kurz = 'kurz'):
        names = []
        for attr_name in self.attributes_long.keys():
            attr_obj = self.attributes_long[attr_name]
            if attr_obj.node_type not in allowed_node_types:
                continue
            if long_or_kurz == 'long':
                names.append(attr_obj.attr_name_long)
            elif long_or_kurz == 'kurz':
                names.append(attr_obj.attr_name_kurz)
            else:
                raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
        return np.array(names)
    
    def getAllAttributeNames(self, long_or_kurz = 'kurz'):
        return self.getAttributeNames({'meta', 'input', 'output'}, long_or_kurz)

    def getInputOutputAttributeNames(self, long_or_kurz = 'kurz'):
        return self.getAttributeNames({'input', 'output'}, long_or_kurz)

    def getMetaInputAttributeNames(self, long_or_kurz = 'kurz'):
        return self.getAttributeNames({'meta', 'input'}, long_or_kurz)

    def getMetaAttributeNames(self, long_or_kurz = 'kurz'):
        return self.getAttributeNames({'meta'}, long_or_kurz)

    def getInputAttributeNames(self, long_or_kurz = 'kurz'):
        return self.getAttributeNames({'input'}, long_or_kurz)

    def getOutputAttributeNames(self, long_or_kurz = 'kurz'):
        return self.getAttributeNames({'output'}, long_or_kurz)

    def getBinaryAttributeNames(self, long_or_kurz = 'kurz'):
        names = []
        for attr_name_long in self.getInputAttributeNames('long'):
            attr_obj = self.attributes_long[attr_name_long]
            if attr_obj.node_type == 'input' and attr_obj.attr_type == 'binary':
                if long_or_kurz == 'long':
                    names.append(attr_obj.attr_name_long)
                elif long_or_kurz == 'kurz':
                    names.append(attr_obj.attr_name_kurz)
                else:
                    raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
        return np.array(names)

    def getActionableAttributeNames(self, long_or_kurz = 'kurz'):
        names = []
        for attr_name_long in self.getInputAttributeNames('long'):
            attr_obj = self.attributes_long[attr_name_long]
            if attr_obj.node_type == 'input' and attr_obj.actionability != 'none':
                if long_or_kurz == 'long':
                    names.append(attr_obj.attr_name_long)
                elif long_or_kurz == 'kurz':
                    names.append(attr_obj.attr_name_kurz)
                else:
                    raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
        return np.array(names)

    def getNonActionableAttributeNames(self, long_or_kurz = 'kurz'):
        a = self.getInputAttributeNames(long_or_kurz)
        b = self.getActionableAttributeNames(long_or_kurz)
        return np.setdiff1d(a,b)

    def getMutableAttributeNames(self, long_or_kurz = 'kurz'):
        names = []
        for attr_name_long in self.getInputAttributeNames('long'):
            attr_obj = self.attributes_long[attr_name_long]
            if attr_obj.node_type == 'input' and attr_obj.mutability != False:
                if long_or_kurz == 'long':
                    names.append(attr_obj.attr_name_long)
                elif long_or_kurz == 'kurz':
                    names.append(attr_obj.attr_name_kurz)
                else:
                    raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
        return np.array(names)

    def getNonMutableAttributeNames(self, long_or_kurz = 'kurz'):
        a = self.getInputAttributeNames(long_or_kurz)
        b = self.getMutableAttributeNames(long_or_kurz)
        return np.setdiff1d(a,b)

    def getIntegerBasedAttributeNames(self, long_or_kurz = 'kurz'):
        names = []
        for attr_name_long in self.getInputAttributeNames('long'):
            attr_obj = self.attributes_long[attr_name_long]
            if attr_obj.attr_type == 'numeric-int':
                if long_or_kurz == 'long':
                    names.append(attr_obj.attr_name_long)
                elif long_or_kurz == 'kurz':
                    names.append(attr_obj.attr_name_kurz)
                else:
                    raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
        return np.array(names)

    def getRealBasedAttributeNames(self, long_or_kurz = 'kurz'):
        names = []
        for attr_name_long in self.getInputAttributeNames('long'):
            attr_obj = self.attributes_long[attr_name_long]
            if attr_obj.attr_type == 'numeric-real':
                if long_or_kurz == 'long':
                    names.append(attr_obj.attr_name_long)
                elif long_or_kurz == 'kurz':
                    names.append(attr_obj.attr_name_kurz)
                else:
                    raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
        return np.array(names)
    
    def assertSiblingsShareAttributes(self, long_or_kurz = 'kurz'):
        dict_of_siblings = self.getDictOfSiblings(long_or_kurz)
        for parent_name in dict_of_siblings['cat'].keys():
            siblings = dict_of_siblings['cat'][parent_name]
            assert len(siblings) > 1
            for sibling in siblings:
                if long_or_kurz == 'long':
                    self.attributes_long[sibling].attr_type = self.attributes_long[siblings[0]].attr_type
                    self.attributes_long[sibling].node_type = self.attributes_long[siblings[0]].node_type
                    self.attributes_long[sibling].actionability = self.attributes_long[siblings[0]].actionability
                    self.attributes_long[sibling].mutability = self.attributes_long[siblings[0]].mutability
                    self.attributes_long[sibling].parent_name_long = self.attributes_long[siblings[0]].parent_name_long
                    self.attributes_long[sibling].parent_name_kurz = self.attributes_long[siblings[0]].parent_name_kurz
                elif long_or_kurz == 'kurz':
                    self.attributes_kurz[sibling].attr_type = self.attributes_kurz[siblings[0]].attr_type
                    self.attributes_kurz[sibling].node_type = self.attributes_kurz[siblings[0]].node_type
                    self.attributes_kurz[sibling].actionability = self.attributes_kurz[siblings[0]].actionability
                    self.attributes_kurz[sibling].mutability = self.attributes_kurz[siblings[0]].mutability
                    self.attributes_kurz[sibling].parent_name_long = self.attributes_kurz[siblings[0]].parent_name_long
                    self.attributes_kurz[sibling].parent_name_kurz = self.attributes_kurz[siblings[0]].parent_name_kurz
                else:
                    raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')

    def getSiblingsFor(self, attr_name_long_or_kurz):
        assert \
        'cat' in attr_name_long_or_kurz or 'ord' in attr_name_long_or_kurz, \
        'attr_name must include either `cat` or `ord`.'
        if attr_name_long_or_kurz in self.getInputOutputAttributeNames('long'):
            attr_name_long = attr_name_long_or_kurz
            dict_of_siblings_long = self.getDictOfSiblings('long')
            for parent_name_long in dict_of_siblings_long['cat']:
                siblings_long = dict_of_siblings_long['cat'][parent_name_long]
                if attr_name_long_or_kurz in siblings_long:
                    return siblings_long
            for parent_name_long in dict_of_siblings_long['ord']:
                siblings_long = dict_of_siblings_long['ord'][parent_name_long]
                if attr_name_long_or_kurz in siblings_long:
                    return siblings_long
        elif attr_name_long_or_kurz in self.getInputOutputAttributeNames('kurz'):
            attr_name_kurz = attr_name_long_or_kurz
            dict_of_siblings_kurz = self.getDictOfSiblings('kurz')
            for parent_name_kurz in dict_of_siblings_kurz['cat']:
                siblings_kurz = dict_of_siblings_kurz['cat'][parent_name_kurz]
                if attr_name_long_or_kurz in siblings_kurz:
                    return siblings_kurz
            for parent_name_kurz in dict_of_siblings_kurz['ord']:
                siblings_kurz = dict_of_siblings_kurz['ord'][parent_name_kurz]
                if attr_name_long_or_kurz in siblings_kurz:
                    return siblings_kurz
        else:
            raise Exception(f'{attr_name_long_or_kurz} not recognized as a valid `attr_name_long_or_kurz`.')
    
    def getDictOfSiblings(self, long_or_kurz = 'kurz'):
        if long_or_kurz == 'long':
            dict_of_siblings_long = {}
            dict_of_siblings_long['cat'] = {}
            dict_of_siblings_long['ord'] = {}
            for attr_name_long in self.getInputAttributeNames('long'):
                attr_obj = self.attributes_long[attr_name_long]
                if attr_obj.attr_type == 'sub-categorical':
                    if attr_obj.parent_name_long not in dict_of_siblings_long['cat'].keys():
                        dict_of_siblings_long['cat'][attr_obj.parent_name_long] = [] # initiate key-value pair
                    dict_of_siblings_long['cat'][attr_obj.parent_name_long].append(attr_obj.attr_name_long)
                elif attr_obj.attr_type == 'sub-ordinal':
                    if attr_obj.parent_name_long not in dict_of_siblings_long['ord'].keys():
                        dict_of_siblings_long['ord'][attr_obj.parent_name_long] = [] # initiate key-value pair
                    dict_of_siblings_long['ord'][attr_obj.parent_name_long].append(attr_obj.attr_name_long)
            for key in dict_of_siblings_long['cat'].keys():
                dict_of_siblings_long['cat'][key] = sorted(dict_of_siblings_long['cat'][key], key = lambda x : int(x.split('_')[-1]))
            for key in dict_of_siblings_long['ord'].keys():
                dict_of_siblings_long['ord'][key] = sorted(dict_of_siblings_long['ord'][key], key = lambda x : int(x.split('_')[-1]))
            return dict_of_siblings_long
        elif long_or_kurz == 'kurz':
            dict_of_siblings_kurz = {}
            dict_of_siblings_kurz['cat'] = {}
            dict_of_siblings_kurz['ord'] = {}
            for attr_name_kurz in self.getInputAttributeNames('kurz'):
                attr_obj = self.attributes_kurz[attr_name_kurz]
                if attr_obj.attr_type == 'sub-categorical':
                    if attr_obj.parent_name_kurz not in dict_of_siblings_kurz['cat'].keys():
                        dict_of_siblings_kurz['cat'][attr_obj.parent_name_kurz] = [] # initiate key-value pair
                    dict_of_siblings_kurz['cat'][attr_obj.parent_name_kurz].append(attr_obj.attr_name_kurz)
                elif attr_obj.attr_type == 'sub-ordinal':
                    if attr_obj.parent_name_kurz not in dict_of_siblings_kurz['ord'].keys():
                        dict_of_siblings_kurz['ord'][attr_obj.parent_name_kurz] = [] # initiate key-value pair
                    dict_of_siblings_kurz['ord'][attr_obj.parent_name_kurz].append(attr_obj.attr_name_kurz)
            for key in dict_of_siblings_kurz['cat'].keys():
                dict_of_siblings_kurz['cat'][key] = sorted(dict_of_siblings_kurz['cat'][key], key = lambda x : int(x.split('_')[-1]))
            for key in dict_of_siblings_kurz['ord'].keys():
                dict_of_siblings_kurz['ord'][key] = sorted(dict_of_siblings_kurz['ord'][key], key = lambda x : int(x.split('_')[-1]))
            return dict_of_siblings_kurz
        else:
            raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')

    def getOneHotAttributesNames(self, long_or_kurz = 'kurz'):
        tmp = self.getDictOfSiblings(long_or_kurz)
        names = []
        for key1 in tmp.keys():
            for key2 in tmp[key1].keys():
                names.extend(tmp[key1][key2])
        return np.array(names)

    def getNonHotAttributesNames(self, long_or_kurz = 'kurz'):
        a = self.getInputAttributeNames(long_or_kurz)
        b = self.getOneHotAttributesNames(long_or_kurz)
        return np.setdiff1d(a,b)

    def getVariableRanges(self):
        return dict(zip(self.getInputAttributeNames('kurz'), [self.attributes_kurz[attr_name_kurz].upper_bound - self.attributes_kurz[attr_name_kurz].lower_bound
        for attr_name_kurz in self.getInputAttributeNames('kurz')],))
    
    def printDataset(self, long_or_kurz = 'kurz'):
        if long_or_kurz == 'long':
            for attr_name_long in self.attributes_long:
                print(self.attributes_long[attr_name_long].__dict__)
        elif long_or_kurz == 'kurz':
            for attr_name_kurz in self.attributes_kurz:
                print(self.attributes_kurz[attr_name_kurz].__dict__)
        else:
            raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    
    def getBalancedDataFrame(self):
        balanced_data_frame = copy.deepcopy(self.data_frame_kurz)
        meta_cols = self.getMetaAttributeNames()
        input_cols = self.getInputAttributeNames()
        output_col = self.getOutputAttributeNames()[0]
        assert np.array_equal(np.unique(balanced_data_frame[output_col]), np.array([0, 1]))
        unique_values_and_count = balanced_data_frame[output_col].value_counts()
        if self.dataset_name == 'heart':
            number_of_subsamples_in_each_class = unique_values_and_count.min() // 50 * 50
        else:
            number_of_subsamples_in_each_class = unique_values_and_count.min() // 250 * 250
        balanced_data_frame = pd.concat([
            balanced_data_frame[balanced_data_frame.loc[:,output_col] == 0].sample(number_of_subsamples_in_each_class, random_state = RANDOM_SEED),
            balanced_data_frame[balanced_data_frame.loc[:,output_col] == 1].sample(number_of_subsamples_in_each_class, random_state = RANDOM_SEED),
        ]).sample(frac = 1, random_state = RANDOM_SEED)
        return balanced_data_frame, meta_cols, input_cols, output_col
    
    def getTrainTestSplit(self, preprocessing = None, with_meta = False):

        def setBoundsToZeroOne():
            for attr_name_kurz in self.getNonHotAttributesNames('kurz'):
                attr_obj = self.attributes_kurz[attr_name_kurz]
                attr_obj.lower_bound = 0.0
                attr_obj.upper_bound = 1.0
                attr_obj = self.attributes_long[attr_obj.attr_name_long]
                attr_obj.lower_bound = 0.0
                attr_obj.upper_bound = 1.0

        # Normalize data: bring everything to [0, 1] - implemented for when feeding the model to DiCE
        def normalizeData(X_train, X_test):
            for attr_name_kurz in self.getNonHotAttributesNames('kurz'):
                attr_obj = self.attributes_kurz[attr_name_kurz]
                lower_bound = attr_obj.lower_bound
                upper_bound =attr_obj.upper_bound
                X_train[attr_name_kurz] = (X_train[attr_name_kurz] - lower_bound) / (upper_bound - lower_bound)
                X_test[attr_name_kurz] = (X_test[attr_name_kurz] - lower_bound) / (upper_bound - lower_bound)
            setBoundsToZeroOne()
            return X_train, X_test

        def standardizeData(X_train, X_test):
            x_mean = X_train.mean()
            x_std = X_train.std()
            for index in x_std.index:
                if '_ord_' in index or '_cat_' in index:
                    x_mean[index] = 0
                    x_std[index] = 1
            X_train = (X_train - x_mean) / x_std
            X_test = (X_test - x_mean) / x_std
            return X_train, X_test

        balanced_data_frame, meta_cols, input_cols, output_col = self.getBalancedDataFrame()

        if with_meta:
            all_data = balanced_data_frame.loc[:,np.array((input_cols, meta_cols)).flatten()]
            all_true_labels = balanced_data_frame.loc[:,output_col]
            if preprocessing is not None:
                assert with_meta == False, 'This feature is not built yet...'
            X_train, X_test, y_train, y_test = train_test_split(all_data, all_true_labels, train_size=.7, random_state = RANDOM_SEED)
            U_train = X_train[self.getMetaAttributeNames()]
            U_test = X_test[self.getMetaAttributeNames()]
            X_train = X_train[self.getInputAttributeNames()]
            X_test = X_test[self.getInputAttributeNames()]
            y_train = y_train
            y_test = y_test
            return X_train, X_test, U_train, U_test, y_train, y_test
        else:
            all_data = balanced_data_frame.loc[:,input_cols]
            all_true_labels = balanced_data_frame.loc[:,output_col]
            X_train, X_test, y_train, y_test = train_test_split(all_data, all_true_labels, train_size=.7, random_state = RANDOM_SEED)
            if preprocessing == 'standardize':
                X_train, X_test = standardizeData(X_train, X_test)
            elif preprocessing == 'normalize':
                X_train, X_test = normalizeData(X_train, X_test)
            return X_train, X_test, y_train, y_test

class DatasetAttribute(object):

    def __init__(
        self,
        attr_name_long,
        attr_name_kurz,
        attr_type,
        node_type,
        actionability,
        mutability,
        parent_name_long,
        parent_name_kurz,
        lower_bound,
        upper_bound):

        if attr_type not in VALID_ATTRIBUTE_DATA_TYPES:
            raise Exception("`attr_type` must be one of %r." % VALID_ATTRIBUTE_DATA_TYPES)

        if node_type not in VALID_ATTRIBUTE_NODE_TYPES:
            raise Exception("`node_type` must be one of %r." % VALID_ATTRIBUTE_NODE_TYPES)

        if actionability not in VALID_ACTIONABILITY_TYPES:
            raise Exception("`actionability` must be one of %r." % VALID_ACTIONABILITY_TYPES)

        if mutability not in VALID_MUTABILITY_TYPES:
            raise Exception("`mutability` must be one of %r." % VALID_MUTABILITY_TYPES)

        if lower_bound > upper_bound:
            raise Exception("`lower_bound` must be <= `upper_bound`")
    
        if attr_type in {'sub-categorical', 'sub-ordinal'}:
            assert parent_name_long != -1, 'Parent ID set for non-hot attribute.'
            assert parent_name_kurz != -1, 'Parent ID set for non-hot attribute.'
            if attr_type == 'sub-categorical':
                assert lower_bound == 0
                assert upper_bound == 1
        if attr_type == 'sub-ordinal':
            assert lower_bound == 0 or lower_bound == 1
            assert upper_bound == 1
        else:
            assert parent_name_long == -1, 'Parent ID set for non-hot attribute.'
            assert parent_name_kurz == -1, 'Parent ID set for non-hot attribute.'
        if attr_type in {'categorical', 'ordinal'}:
            assert lower_bound == 1 # setOneHotValue & setThermoValue assume this in their logic

        if attr_type in {'binary', 'categorical', 'sub-categorical'}: # not 'ordinal' or 'sub-ordinal'
            assert actionability in {'none', 'any'}, f"{attr_type}'s actionability can only be in {'none', 'any'}, not `{actionability}`."

        if node_type != 'input':
            assert actionability == 'none', f'{node_type} attribute is not actionable.'
            assert mutability == False, f'{node_type} attribute is not mutable.'

        if actionability != 'none':
            assert mutability == True
        if mutability == False:
            assert actionability == 'none'

        if parent_name_long == -1 or parent_name_kurz == -1:
            assert parent_name_long == parent_name_kurz == -1
    
        self.attr_name_long = attr_name_long
        self.attr_name_kurz = attr_name_kurz
        self.attr_type = attr_type
        self.node_type = node_type
        self.actionability = actionability
        self.mutability = mutability
        self.parent_name_long = parent_name_long
        self.parent_name_kurz = parent_name_kurz
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

def loadDataset(dataset_name, return_one_hot, load_from_cache = False, debug_flag = True, meta_param = None):

    def getInputOutputColumns(data_frame):
        all_data_frame_cols = data_frame.columns.values
        input_cols = [x for x in all_data_frame_cols if 'label' not in x.lower()]
        output_cols = [x for x in all_data_frame_cols if 'label' in x.lower()]
        assert len(output_cols) == 1
        return input_cols, output_cols[0]

    one_hot_string = 'one_hot' if return_one_hot else 'non_hot'
    save_file_path = os.path.join(os.path.dirname(__file__), f'_data_main/_cached/{dataset_name}_{one_hot_string}')

    if dataset_name == 'adult':
        binary = ['Sex','NativeCountry','Race']
        categorical = ['WorkClass','MaritalStatus','Occupation','Relationship']
        ordinal = ['EducationLevel','AgeGroup']
        continuous = ['EducationNumber','CapitalGain','CapitalLoss','HoursPerWeek']
        input_cols = binary + categorical + ordinal + continuous
        label = ['label']
        data_frame_non_hot = pd.read_csv(dataset_dir+'adult/preprocessed_adult.csv', index_col=0)
        data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
        attributes_non_hot = {}
        input_cols, output_col = getInputOutputColumns(data_frame_non_hot)
        col_name = output_col
        attributes_non_hot[col_name] = DatasetAttribute(
            attr_name_long = col_name,
            attr_name_kurz = 'y',
            attr_type = 'binary',
            node_type = 'output',
            actionability = 'none',
            mutability = False,
            parent_name_long = -1,
            parent_name_kurz = -1,
            lower_bound = data_frame_non_hot[col_name].min(),
            upper_bound = data_frame_non_hot[col_name].max())
        
        col_name = label[0]
        attributes_non_hot[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
                                                   mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = data_frame_non_hot[col_name].min(), upper_bound = data_frame_non_hot[col_name].max())
        for col_idx, col_name in enumerate(input_cols):

            if col_name == 'Sex':
                attr_type = 'binary'
                actionability = 'none' # 'none'
                mutability = False
            elif col_name == 'AgeGroup':
                attr_type = 'ordinal'
                actionability = 'none' # 'none'
                mutability = False
            elif col_name == 'NativeCountry':
                attr_type = 'binary'
                actionability = 'none' # 'none'
                mutability = False
            elif col_name == 'Race':
                attr_type = 'binary'
                actionability = 'none'
                mutability = False
            elif col_name == 'WorkClass':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'EducationNumber':
                attr_type = 'numeric-int'
                actionability = 'same-or-increase'
                mutability = True
            elif col_name == 'EducationLevel':
                attr_type = 'ordinal'
                actionability = 'same-or-increase'
                mutability = True
            elif col_name == 'MaritalStatus':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Occupation':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Relationship':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'CapitalGain':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'CapitalLoss':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'HoursPerWeek':
                attr_type = 'numeric-int'
                actionability = 'any'
                mutability = True

            attributes_non_hot[col_name] = DatasetAttribute(
                attr_name_long = col_name,
                attr_name_kurz = f'x{col_idx}',
                attr_type = attr_type,
                node_type = 'input',
                actionability = actionability,
                mutability = mutability,
                parent_name_long = -1,
                parent_name_kurz = -1,
                lower_bound = attributes_non_hot[col_name].min(),
                upper_bound = attributes_non_hot[col_name].max())
        
    elif dataset_name == 'german':
        binary = ['Sex','Single','Unemployed']
        categorical = ['PurposeOfLoan','InstallmentRate','Housing']
        ordinal = []
        continuous = ['Age','Credit','LoanDuration']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Label']
        data_frame_non_hot = pd.read_csv(dataset_dir+'german/preprocessed_german.csv', index_col=0)
        data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
        attributes_non_hot = {}
    
        """
        MACE variables / attributes
        """
        col_name = label[0]
        attributes_non_hot[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
                                                   mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = data_frame_non_hot[col_name].min(), upper_bound = data_frame_non_hot[col_name].max())
        for col_idx, col_name in enumerate(input_cols):

            if col_name == 'Sex':
                attr_type = 'binary'
                actionability = 'none'
                mutability = False
            elif col_name == 'Single':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'Unemployed':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'PurposeOfLoan':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'InstallmentRate':
                attr_type = 'categorical'
                actionability = 'Any'
                mutability = True
            elif col_name == 'Housing':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Age':
                attr_type = 'numeric-real'
                actionability = 'same-or-increase'
                mutability = True
            elif col_name == 'Credit':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'LoanDuration':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            
            attributes_non_hot[col_name] = DatasetAttribute(
                attr_name_long = col_name,
                attr_name_kurz = f'x{col_idx}',
                attr_type = attr_type,
                node_type = 'input',
                actionability = actionability,
                mutability = mutability,
                parent_name_long = -1,
                parent_name_kurz = -1,
                lower_bound = data_frame_non_hot[col_name].min(),
                upper_bound = data_frame_non_hot[col_name].max())
    
    elif dataset_name == 'synthetic_athlete':
        binary = ['Sex']
        categorical = ['Diet','Sport','TrainingTime']
        ordinal = []
        continuous = ['Age','SleepHours']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Label']
        data_frame_non_hot = pd.read_csv(dataset_dir+'synthetic_athlete/processed_synthetic_athlete.csv',index_col=0)
        data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
        attributes_non_hot = {}

        """
        MACE variables / attributes
        """
        col_name = label[0]
        attributes_non_hot[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
                                                   mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = df[col_name].min(), upper_bound = df[col_name].max())
        for col_idx, col_name in enumerate(input_cols):

            if col_name == 'Sex':
                attr_type = 'binary'
                actionability = 'none'
                mutability = False
            elif col_name == 'Diet':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Sport':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'TrainingTime':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Age':
                attr_type = 'numeric-real'
                actionability = 'none'
                mutability = False
            elif col_name == 'SleepHours':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True

            attributes_non_hot[col_name] = DatasetAttribute(
                attr_name_long = col_name,
                attr_name_kurz = col_name,
                attr_type = attr_type,
                node_type = 'input',
                actionability = actionability,
                mutability = mutability,
                parent_name_long = -1,
                parent_name_kurz = -1,
                lower_bound = data_frame_non_hot[col_name].min(),
                upper_bound = data_frame_non_hot[col_name].max())
    
    else:
        raise Exception(f'{dataset_name} not recognized as a valid dataset.')
    
    if return_one_hot:
        data_frame, attributes = getOneHotEquivalent(data_frame_non_hot, attributes_non_hot)
    else:
        data_frame, attributes = data_frame_non_hot, attributes_non_hot

    dataset_obj = Dataset(data_frame, attributes, return_one_hot, dataset_name)
    pickle.dump(dataset_obj, open(save_file_path, 'wb'))
    return dataset_obj