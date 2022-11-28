from ijuice import IJUICE
from nt import NN
from mo import MO
from ft import FT
from rt import RT
from gs import GS
from face import FACE
from juice import JUICE


class Counterfactual:

    def __init__(self, data, model, method, ioi, type='euclidean', split='100'):
        self.data = data
        self.model = model
        self.method = method
        self.ioi = ioi
        self.type = type
        self.split = split
        self.cf_method = self.select_cf_method()

    def select_cf_method(self):
        """
        Method that selects the method to find the counterfactual and stores it in "normal_x_cf"
        ['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice','ijuice']
        """
        if self.method == 'ijuice':
            cf_method = IJUICE(self)
        elif self.method == 'nn':
            cf_method = NN(self)
        elif self.method == 'mo':
            cf_method = MO(self)
        elif self.method == 'ft':
            cf_method = FT(self)
        elif self.method == 'rt':
            cf_method = RT(self)
        elif self.method == 'gs':
            cf_method = GS(self)
        elif self.method == 'face':
            cf_method = FACE(self)
        elif self.method == 'juice':
            cf_method = JUICE(self)
        return cf_method