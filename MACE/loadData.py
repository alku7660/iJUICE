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
                ipsh()

