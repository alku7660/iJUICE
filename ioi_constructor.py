import numpy as np
import pandas as pd
import time
from evaluator_constructor import distance_calculation

class IOI:

    def __init__(self, idx, data, model, type='euclidean') -> None:
        self.idx = idx
        self.x_df = data.test_df.loc[idx].to_frame().T
        self.x = self.x_df.to_numpy()
        self.normal_x_df = data.transformed_test_df.loc[idx].to_frame().T
        self.normal_x = self.normal_x_df.to_numpy()
        self.label = model.model.predict(self.normal_x.reshape(1, -1))
        self.train_sorted, self.train_sorting_time = self.sorted(data, type)
    
    def sorted(self, data, type):
        """
        Function to organize dataset with respect to distance to instance x
        """
        start_time = time.time()
        sort_data_distance = []
        for i in range(data.transformed_train_np.shape[0]):
            dist = distance_calculation(data.transformed_train_np[i], self.normal_x, type)
            sort_data_distance.append((data.transformed_train_np[i], dist, data.train_target[i]))      
        sort_data_distance.sort(key=lambda x: x[1])
        end_time = time.time()
        total_time = end_time - start_time 
        return sort_data_distance, total_time