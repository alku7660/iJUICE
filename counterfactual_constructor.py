from ijuice import Ijuice
from juice import Juice

class Counterfactual:

    def __init__(self, data, model, method, ioi, type='euclidean', split='100'):
        self.data = data
        self.model = model
        self.method = method
        self.ioi = ioi
        self.type = type
        self.split = split
        self.cf_method = self.select_train()

    def select_train(self):
        """
        Method that selects the method to find the counterfactual and stores it in "normal_x_cf"
        """
        if self.method == 'ijuice':
            cf_method = Ijuice(self)
        elif self.method == 'juice':
            cf_method = Juice(self)