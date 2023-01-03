import numpy as np
import time
from evaluator_constructor import verify_feasibility

class NN:

    def __init__(self, counterfactual) -> None:
        self.normal_x_cf, self.run_time = near_neigh(counterfactual)

def near_neigh(counterfactual):
    """
    Original nn counterfactual method
    """
    start_time = time.time()
    nn_cf = None
    for i in counterfactual.ioi.train_sorted:
        if i[2] != counterfactual.ioi.label and not np.array_equal(counterfactual.ioi.normal_x, i[0]):
            nn_cf = i[0]
            break
    end_time = time.time()
    total_time = end_time - start_time + counterfactual.ioi.train_sorting_time
    return nn_cf, total_time

def nn_for_juice(counterfactual):
    """
    Function that returns the nearest counterfactual with respect to instance of interest x
    """
    start_time = time.time()
    nn_cf = None
    for i in counterfactual.ioi.train_sorted:
        if i[2] != counterfactual.ioi.label and counterfactual.model.model.predict(i[0].reshape(1,-1)) != counterfactual.ioi.label and verify_feasibility(counterfactual.ioi.normal_x, i[0], counterfactual.data) and not np.array_equal(counterfactual.ioi.normal_x, i[0]):
            nn_cf = i[0]
            break
    if nn_cf is None:
        for i in counterfactual.ioi.train_sorted:
            if i[2] != counterfactual.ioi.label and verify_feasibility(counterfactual.ioi.normal_x, i[0], counterfactual.data) and not np.array_equal(counterfactual.ioi.normal_x, i[0]):
                nn_cf = i[0]
                break
    end_time = time.time()
    total_time = end_time - start_time + counterfactual.ioi.train_sorting_time
    return nn_cf, total_time