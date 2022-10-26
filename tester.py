from data_constructor import load_dataset
from model_constructor import Model
from ioi_constructor import IOI
from evaluator_constructor import Evaluator
from counterfactual_constructor import Counterfactual
from address import save_obj, results_obj

# datasets = ['adult','bank','compass','credit','dutch','diabetes','german','ionosphere','kdd_census','law','oulad','student','synthetic_athlete','synthetic_disease']
datasets = ['synthetic_athlete']
methods = ['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice','ijuice']
seed_int = 54321
step = 0.01
train_fraction = 0.7
distance_type = ['euclidean'] # ['euclidean','L1','mixed_L1','L1_inf']
continuous_split = ['100']    # ['2','5','10','20','50','100','train']
num_instances = 1 # data.test_df.shape[0]

for data_str in datasets:
    data = load_dataset(data_str, train_fraction, seed_int, step)
    model = Model(data)
    for method_str in methods:
        for type in distance_type:
            for split in continuous_split:
                eval = Evaluator(data, method_str, type, split)
                for ins in range(num_instances):
                    idx = data.test_df.index[ins]
                    ioi = IOI(idx, data, model, type)
                    cf_gen = Counterfactual(data, model, method_str, ioi, type, split)
                    eval.add_specific_x_data(data, model, ioi, cf_gen)
                save_obj(eval, results_obj, f'{data_str}_{method_str}_{type}_{split}.pkl')
            # print(f'Optimizer solution status: {ijuice.optimizer.status}')
            # print(f'Solution:')
            # nodes = [ijuice.nn_cf]
            # nodes.extend(list(ijuice.get_nodes(model)))
            # for i,j in ijuice.A:
            #     if ijuice.y[i,j].x > 0:
            #         print(f'y{i,j}: {ijuice.y[i,j].x}')
            # for i in ijuice.C.keys():
            #     if ijuice.x[i].x > 0:
            #         print(f'x({i}): {ijuice.x[i].x}')
            #         print(f'Node {i}: {nodes[i - 1]}')
            #         print(f'Original IOI: {ioi.normal_x}. Euclidean Distance: {np.round(np.sqrt(np.sum((nodes[i - 1] - ioi.normal_x)**2)),3)}')
            #         print(f'NN CF: {ijuice.nn_cf}')




