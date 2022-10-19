from data_constructor import load_dataset
from model_constructor import Model
from ioi_constructor import IOI
from ijuice import Ijuice
from address import save_obj, results_obj
import numpy as np

# datasets = ['adult','bank','compass','credit','dutch','diabetes','german','ionosphere','kdd_census','law','oulad','student','synthetic_athlete','synthetic_disease']
datasets = ['synthetic_athlete']
seed_int = 54321
step = 0.01
train_fraction = 0.7

for data_str in datasets:
    data = load_dataset(data_str, train_fraction, seed_int, step)
    model = Model(data)
    num_instances = 1 # data.test_df.shape[0]
    for ins in range(num_instances):
        idx = data.test_df.index[ins]
        ioi = IOI(idx, data, model)
        ijuice_obj = Ijuice(data, model, ioi)
        
        print(f'Optimizer solution status: {ijuice_obj.optimizer.status}')
        print(f'Solution:')
        nodes = [ijuice_obj.nn_cf]
        nodes.extend(list(ijuice_obj.get_nodes(model)))
        for i,j in ijuice_obj.A:
            if ijuice_obj.y[i,j].x > 0:
                print(f'y{i,j}: {ijuice_obj.y[i,j].x}')
                # print(f'Node {i}: {nodes[i]}')
                # print(f'Node {j}: {nodes[j]}')
        for i in ijuice_obj.C.keys():
            if ijuice_obj.x[i].x > 0:
                print(f'x({i}): {ijuice_obj.x[i].x}')
                print(f'Node {i}: {nodes[i]}')
                print(f'Original IOI: {ioi.normal_x}. Euclidean Distance: {np.sqrt(np.sum((nodes[i] - ioi.normal_x)**2))}')




