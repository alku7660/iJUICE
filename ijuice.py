import numpy as np


class Ijuice:

    def __init__(self, data, ioi, model):
        self.name = data.name
        self.ioi_idx = ioi.idx
        self.ioi = ioi.x
        self.normal_ioi = ioi.normal_ioi
        self.ioi_label = ioi.label
        self.nn_cf = 