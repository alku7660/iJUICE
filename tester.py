import warnings
warnings.filterwarnings("ignore")
from data_constructor import load_dataset
from model_constructor import Model
from ioi_constructor import IOI
from evaluator_constructor import Evaluator
from counterfactual_constructor import Counterfactual
from address import save_obj, load_obj, results_obj

datasets = ['synthetic_athlete'] # ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','ionosphere','student','oulad','law','heart','synthetic_athlete','synthetic_disease']
methods = ['nn'] # ['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice','ijuice']
seed_int = 54321
step = 0.01
train_fraction = 0.7
distance_type = ['euclidean'] # ['euclidean','L1','L1_L0','L1_L0_inf']
lagranges = [0.5]    # [0, 0.25 0.50, 0.75, 1.0]
num_instances = 5 # data.test_df.shape[0]
prepare_for_mace = True

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
        mace_df = load_obj(f'{data_str}_mace_df.pkl')
        mace_df_idx = mace_df.index
        # num_instances = num_instances if num_instances <= data.undesired_transformed_test_df.shape[0] else data.undesired_transformed_test_df.shape[0]
        num_instances = num_instances if num_instances <= len(mace_df_idx) else len(mace_df_idx)
        for method_str in methods:
            for type in distance_type:
                for lagrange in lagranges:
                    eval = Evaluator(data, method_str, type, lagrange)
                    for ins in range(num_instances):
                        idx = data.undesired_transformed_test_df.index[ins]
                        ioi = IOI(idx, data, model, type)
                        cf_gen = Counterfactual(data, model, method_str, ioi, type, lagrange)
                        eval.add_specific_x_data(cf_gen)
                        print(f'Data {data_str.capitalize()} | Method {method_str.capitalize()} | Type {type.capitalize()} | lagrange {lagrange} | Instance {ins+1}')
                    save_obj(eval, results_obj, f'{data_str}_{method_str}_{type}_{lagrange}.pkl')