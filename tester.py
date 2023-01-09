import warnings
warnings.filterwarnings("ignore")
from data_constructor import load_dataset
from model_constructor import Model
from ioi_constructor import IOI
from evaluator_constructor import Evaluator
from counterfactual_constructor import Counterfactual
from address import save_obj, load_obj, results_obj

datasets = ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law','heart','synthetic_athlete','synthetic_disease'] # ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law','heart','synthetic_athlete','synthetic_disease']
methods = ['juice','ijuice'] # ['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice','ijuice']
seed_int = 54321
step = 0.01
train_fraction = 0.7
distance_type = ['euclidean'] # ['euclidean','L1','L_inf','L1_L0','L1_L0_L_inf','prob']
lagranges = [1]    # np.linspace(start=0, stop=1, num=11)
num_instances = 20 # data.test_df.shape[0]
prepare_for_mace = False

"""
Instructions to run tests:
    1. Set prepare_for_mace = True
    2. Set num_instances = 30
    3. Set the datasets to the datasets desired for running the tests. Method can be any
    4. Run tester.py: this stores the undesired class test instances indices only
    5. In maceTest.py set the datasets equal to the datasets in tester.py
    6. In maceTest.py set only_indices = True
    7. Run maceTest.py: this stores the matching undesired indices of both MACE and normal frameworks
    8. Run maceTest.py with only_indices = False: this runs the algorithm and prints the times and CF for all matching undesired instances
    9. Run tester.py with mace among the methods.
    10. Run plotter
"""

if __name__ == '__main__':

    if prepare_for_mace:
        for data_str in datasets:
            data = load_dataset(data_str, train_fraction, seed_int, step)
            model = Model(data)
            data.undesired_test(model)
            idx_list = [data.undesired_transformed_test_df.index[ins] for ins in range(num_instances)]
            save_obj(idx_list, results_obj+data_str+'/', f'{data_str}_idx_list.pkl')
            print(f'Saved list of indices for {data_str} Dataset')

    else:
        for data_str in datasets:
            data = load_dataset(data_str, train_fraction, seed_int, step)
            model = Model(data)
            data.undesired_test(model)
            idx_list = [data.undesired_transformed_test_df.index[ins] for ins in range(num_instances)]
            mace_df = load_obj(f'{data_str}/{data_str}_mace_df.pkl')
            mace_df_idx = list(mace_df.index)
            # num_instances = num_instances if num_instances <= data.undesired_transformed_test_df.shape[0] else data.undesired_transformed_test_df.shape[0]
            num_instances = num_instances if num_instances <= len(mace_df_idx) else len(mace_df_idx)
            for method_str in methods:
                for type in distance_type:
                    for lagrange in lagranges:
                        eval = Evaluator(data, method_str, type, lagrange)
                        for ins in range(num_instances):
                            idx = mace_df_idx[ins]
                            ioi = IOI(idx, data, model, type)
                            cf_gen = Counterfactual(data, model, method_str, ioi, type, lagrange)
                            eval.add_specific_x_data(cf_gen)
                            print(f'Data {data_str.capitalize()} | Method {method_str.capitalize()} | Type {type.capitalize()} | lagrange {lagrange} | Instance {ins+1}')
                        save_obj(eval, results_obj, f'{data_str}_{method_str}_{type}_{lagrange}.pkl')