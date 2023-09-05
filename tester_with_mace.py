import warnings
warnings.filterwarnings("ignore")
from data_constructor import load_dataset
from model_constructor import Model
from ioi_constructor import IOI
from evaluator_constructor import Evaluator
from counterfactual_constructor import Counterfactual
from address import save_obj, load_obj, results_obj

"""
Instructions to run tests with the MACE method:
    If you want to test iJUICE with MACE as competitor, do the following: 
    1. Set the variable "prepare_for_mace" (line 25) equal to true [prepare_for_mace = True].
    2. Set the variable "num_instances" (lines 38 and 48) equal to 50 or a number of instances desired to study [num_instances = 50].
    3. Set the variable "datasets" equal to the list of datasets desired for running the tests. The variable "methods" can be a list containing any of the methods.
    4. Run tester.py: this stores the num_instances undesired class test instances indices.
    5. In maceTest.py (located in Competitors/MACE/) set the variable "datasets" equal to the "datasets" variable in tester.py [Simply define datasets = same list as in here].
    6. In maceTest.py set the variable "only_indices" equal to True [only_indices = True].
    7. Run maceTest.py: this stores the matching undesired class indices of both MACE and iJUICE (the total amount of indices may be lower than num_instances).
    8. Run maceTest.py with variable "only_indices" equal to False [only_indices = False]: this runs the algorithm and prints the times and CF for all matching undesired instances.
    9. Run main.py with mace among the "methods" list.
"""

datasets = ['adult','kdd_census','german'] # ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law','heart','synthetic_athlete','synthetic_disease']
methods = ['ijuice'] # ['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice','ijuice']
seed_int = 54321
step = 0.01
train_fraction = 0.7
distance_type = ['L1_L0'] # ['euclidean','L1','L_inf','L1_L0','L1_L0_L_inf','prob']
lagranges = [0.5]    # np.linspace(start=0, stop=1, num=11)
prepare_for_mace = False

if __name__ == '__main__':

    if prepare_for_mace:
        for data_str in datasets:
            num_instances = 50 # data.test_df.shape[0]
            data = load_dataset(data_str, train_fraction, seed_int, step)
            model = Model(data)
            data.undesired_test(model)
            idx_list = [data.undesired_transformed_test_df.index[ins] for ins in range(num_instances)]
            save_obj(idx_list, results_obj+data_str+'/', f'{data_str}_idx_list.pkl')
            print(f'Saved list of indices for {data_str} Dataset')

    else:
        for data_str in datasets:
            num_instances = 50 # data.test_df.shape[0]
            data = load_dataset(data_str, train_fraction, seed_int, step)
            model = Model(data)
            data.undesired_test(model)
            idx_list = [data.undesired_transformed_test_df.index[ins] for ins in range(num_instances)]
            mace_df = load_obj(f'{data_str}/{data_str}_mace_df.pkl')
            mace_cf_df = load_obj(f'{data_str}/{data_str}_mace_cf_df.pkl')
            mace_df_idx = list(mace_cf_df.index)
            num_instances = num_instances if num_instances <= len(mace_df_idx) else len(mace_df_idx)
            print(f'Dataset {data_str} num instances: {num_instances}')
            for method_str in methods:
                for typ in distance_type:
                    for lagrange in lagranges:
                        eval = Evaluator(data, method_str, typ, lagrange)
                        for ins in range(num_instances):
                            idx = mace_df_idx[ins]
                            ioi = IOI(idx, data, model, typ)
                            cf_gen = Counterfactual(data, model, method_str, ioi, typ, lagrange)
                            eval.add_specific_x_data(cf_gen)
                            print(f'Data {data_str.capitalize()} | Method {method_str.capitalize()} | Type {typ.capitalize()} | lagrange {str(lagrange)} | Instance {ins+1}')
                        save_obj(eval, results_obj, f'{data_str}_{method_str}_{typ}_{str(lagrange)}.pkl')