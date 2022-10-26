import numpy as np
import time
from evaluator_constructor import verify_feasibility

def nn(ioi, data, model):
    """
    Original nn counterfactual method
    """
    start_time = time.time()
    nn_cf = None
    for i in ioi.train_sorted:
        if i[2] != ioi.label and not np.array_equal(ioi.normal_x, i[0]):
            nn_cf = i[0]
            break
    end_time = time.time()
    total_time = end_time - start_time + ioi.train_sorting_time
    return nn_cf, total_time

def nn_for_juice(ioi, data, model):
    """
    Function that returns the nearest counterfactual with respect to instance of interest x
    """
    start_time = time.time()
    nn_cf = None
    for i in ioi.train_sorted:
        if i[2] != ioi.label and model.model.predict(i[0].reshape(1,-1)) != ioi.label and verify_feasibility(ioi.normal_x, i[0], data) and not np.array_equal(ioi.normal_x, i[0]):
            nn_cf = i[0]
            break
    if nn_cf is None:
        for i in ioi.train_sorted:
            if i[2] != ioi.label and verify_feasibility(ioi.normal_x, i[0], data) and not np.array_equal(ioi.normal_x, i[0]):
                nn_cf = i[0]
                break
    if nn_cf is None:
        print(f'NT could not find a feasible and counterfactual predicted CF (None output)')
        return nn_cf
    end_time = time.time()
    total_time = end_time - start_time + ioi.train_sorting_time
    return nn_cf, total_time