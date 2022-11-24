"""
C-CHVAE
Based on original authors implementation: Please see https://github.com/MartinPawel/c-chvae
"""

"""
Imports
"""
from address import results_obj
from address import dataset_dir

class CCHVAE:

    def __init__(self, counterfactual) -> None:
        self.normal_x_cf, self.run_time = cchvae_method(counterfactual)

class Args:
    def __init__(self, data_obj):
        self.batch_size = 100
        self.epochs = 80
        self.train = 1
        self.display = 1
        self.save = 1000
        self.restore = 0
        self.dim_latent_s = 3
        self.dim_latent_z = 2
        self.dim_latent_y = 5
        self.dim_latent_y_partition = '+'
        self.save_file = f'{results_obj}/{data_obj.name}/{data_obj.name}_cchvae.csv'
        self.data_file = f'{dataset_dir}/{data_obj.name}/preprocessed_{data_obj.name}.csv'
        self.data_file_c = data_obj.name
        self.types_file = 'preprocessed_adult_types.csv'
        self.types_file_c = 'preprocessed_adult_types_c.csv'
        self.classifier = ''


def cchvae_method(counterfactual):
    """
    Function that returns C-CHVAE with respect to instance of interest x
    """

    def sampling(settings, types_dict, types_dict_c, out, ncounterfactuals, clf, n_batches_train, n_samples_train, k, n_input, degree_active):

        argvals = settings.split()
        args = Helpers.getArgs(argvals)

    

