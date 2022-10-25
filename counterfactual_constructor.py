from ijuice import Ijuice

class Counterfactual:

    def __init__(self, data, model, method, ioi, type='euclidean', split='100'):
        self.name = data.name
        self.method = method
        self.ioi = ioi.x
        self.normal_ioi = ioi.normal_x
        self.ioi_label = ioi.label

    def