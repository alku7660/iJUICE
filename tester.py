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
continuous_split = ['train']    # ['2','5','10','20','50','100','train']
num_instances = 5 # data.test_df.shape[0]
justification_train_perc = 0.1

for data_str in datasets:
    data = load_dataset(data_str, train_fraction, seed_int, step)
    model = Model(data)
    data.undesired_test(model)
    num_instances = num_instances if num_instances <= data.undesired_transformed_test_df.shape[0] else data.undesired_transformed_test_df.shape[0]
    for method_str in methods:
        for type in distance_type:
            for split in continuous_split:
                eval = Evaluator(data, method_str, type, split, justification_train_perc)
                for ins in range(num_instances):
                    idx = data.undesired_transformed_test_df.index[ins]
                    ioi = IOI(idx, data, model, type)
                    cf_gen = Counterfactual(data, model, method_str, ioi, type, split)
                    eval.add_specific_x_data(cf_gen)
                    print(f'Data {data_str.capitalize()} | Method {method_str.capitalize()} | Type {type.capitalize()} | Split {split.capitalize()} | Instance {ins+1}')
                save_obj(eval, results_obj, f'{data_str}_{method_str}_{type}_{split}.pkl')