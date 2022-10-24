from data_constructor import load_dataset
from model_constructor import Model
from ioi_constructor import IOI
from evaluator_constructor import Evaluator
from ijuice import Ijuice
from address import save_obj, results_obj
import numpy as np

# datasets = ['adult','bank','compass','credit','dutch','diabetes','german','ionosphere','kdd_census','law','oulad','student','synthetic_athlete','synthetic_disease']
datasets = ['synthetic_athlete']
methods = ['nn','mo','ft','rt','gs','face','dice','mace','cchvae','jce','ijce']
seed_int = 54321
step = 0.01
train_fraction = 0.7
distance_type = 'euclidean' # ['L1', 'L1_inf']
num_instances = 1 # data.test_df.shape[0]

for data_str in datasets:
    data = load_dataset(data_str, train_fraction, seed_int, step)
    model = Model(data)
    for method_str in methods:
        eval = Evaluator(data, method_str)
        for ins in range(num_instances):
            idx = data.test_df.index[ins]
            ioi = IOI(idx, data, model)
            ijuice = Ijuice(data, model, ioi, distance_type)
            eval.add_specific_x_data()
            print(f'Optimizer solution status: {ijuice.optimizer.status}')
            print(f'Solution:')
            nodes = [ijuice.nn_cf]
            nodes.extend(list(ijuice.get_nodes(model)))
            for i,j in ijuice.A:
                if ijuice.y[i,j].x > 0:
                    print(f'y{i,j}: {ijuice.y[i,j].x}')
                    # print(f'Node {i}: {nodes[i - 1]}')
                    # print(f'Node {j}: {nodes[j - 1]}')
            for i in ijuice.C.keys():
                if ijuice.x[i].x > 0:
                    print(f'x({i}): {ijuice.x[i].x}')
                    print(f'Node {i}: {nodes[i - 1]}')
                    print(f'Original IOI: {ioi.normal_x}. Euclidean Distance: {np.round(np.sqrt(np.sum((nodes[i - 1] - ioi.normal_x)**2)),3)}')
                    print(f'NN CF: {ijuice.nn_cf}')




