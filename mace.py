"""
Model-Agnostic Counterfactual Explanations (MACE)
Based on original authors implementation: Please see https://github.com/amirhk/mace
"""

"""
Imports
"""
import numpy as np
import pandas as pd
import time
from pysmt.shortcuts import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from address import load_obj

class MACE:

    def __init__(self, counterfactual) -> None:
        self.normal_x_cf_df, self.run_time = mace_method(counterfactual)
        self.columns = list(self.normal_x_cf_df.columns)
        self.normal_x_cf = self.normal_x_cf_df.to_numpy()

def mace_method(counterfactual):
    """
    Function that returns MACE with respect to instance of interest x
    """
    data_name = counterfactual.data.name
    ioi_idx = counterfactual.ioi.idx
    mace_cf_df = load_obj(f'{data_name}/{data_name}_mace_cf_df_extra.pkl')
    mace_time_df = load_obj(f'{data_name}/{data_name}_mace_time_df_extra.pkl')
    cf, run_time = mace_cf_df.loc[ioi_idx].to_frame().T, mace_time_df.loc[ioi_idx].values
    return cf, run_time

    
    