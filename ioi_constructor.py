import numpy as np
import pandas as pd
from evaluator_constructor import distance_calculation

class IOI:

    def __init__(self, idx, data, model) -> None:
        self.idx = idx
        self.x = data.test_df.loc[idx].to_numpy()
        self.normal_x = data.transformed_test_df.loc[idx].to_numpy()
        self.label = model.model.predict(self.normal_x.reshape(1, -1))
        self.train_sorted = self.sorted(data)
    
    def sorted(self, data):
        """
        Function to organize dataset with respect to distance to instance x
        """
        sort_data_distance = []
        for i in range(data.transformed_train_np.shape[0]):
            dist = distance_calculation(data.transformed_train_np[i], self.normal_x)
            sort_data_distance.append((data.transformed_train_np[i], dist, data.train_target[i]))      
        sort_data_distance.sort(key=lambda x: x[1])
        return sort_data_distance