import numpy as np
import pandas as pd

class IOI:

    def __init__(self, idx, data, model) -> None:
        self.idx = idx
        self.x = data.test_df.loc[idx].to_numpy()
        self.normal_x = data.transformed_test_df.loc[idx].to_numpy()
        self.label = model.model.predict(self.normal_x.reshape(1, -1))