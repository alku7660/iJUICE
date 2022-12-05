import warnings
warnings.filterwarnings("ignore")
from data_constructor import load_dataset
from model_constructor import Model
from ioi_constructor import IOI
from evaluator_constructor import Evaluator
from counterfactual_constructor import Counterfactual
from address import save_obj, results_obj

# datasets = ['adult','kdd_census','german','dutch','bank','credit','compass','compass','diabetes','ionosphere','student','oulad','law','synthetic_athlete','synthetic_disease']
datasets = ['synthetic_athlete']
methods = ['ijuice']
seed_int = 54321
step = 0.01
train_fraction = 0.7
distance_type = ['euclidean'] # ['euclidean','L1','L1_L0','L1_L0_inf']
lagranges = [0.0]    # [0, 0.25 0.50, 0.75, 1.0]
num_instances = 1 # data.test_df.shape[0]

for data_str in datasets:
    data = load_dataset(data_str, train_fraction, seed_int, step)
    model = Model(data)
    data.undesired_test(model)
    num_instances = num_instances if num_instances <= data.undesired_transformed_test_df.shape[0] else data.undesired_transformed_test_df.shape[0]
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