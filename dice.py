"""
Diverse Counterfactual Explanations (DiCE)
Based on original authors implementation @ Microsoft: Please see https://github.com/interpretml/DiCE
"""

"""
Imports
"""
import numpy as np
import pandas as pd
import time
import dice_ml

class DICE:

    def __init__(self, counterfactual) -> None:
        self.normal_x_cf, self.run_time = dice_method(counterfactual)

def dice_method(counterfactual):
    """
    Function that returns a single DiCE with respect to instance of interest x
    """
    x = counterfactual.ioi.normal_x
    train_df = counterfactual.data.transformed_train_df
    numerical_feat = counterfactual.data.ordinal + counterfactual.data.continuous
    label = counterfactual.data.label_name
    start_time = time.time()
    d = dice_ml.Data(dataframe=train_df, continuous_features=numerical_feat, outcome_name=label)
    m = dice_ml.Model(model=counterfactual.model.model, backend='sklearn')
    exp = dice_ml.Dice(d, m, method='random')
    dice_exp = exp.generate_counterfactuals(query_instances=x, total_CFs=1, desired_class="opposite")
    cf_df = dice_exp.cf_examples_list[0].final_cfs_df
    cf = cf_df.to_numpy()[0]
    end_time = time.time()
    run_time = end_time - start_time
    return cf, run_time
