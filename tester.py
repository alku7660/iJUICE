from data_constructor import load_dataset
from model_constructor import Model
from ioi_constructor import IOI
from ijuice import Ijuice
from address import save_obj, results_obj

# datasets = ['adult','bank','compass','credit','dutch','diabetes','german','ionosphere','kdd_census','law','oulad','student','synthetic_athlete','synthetic_disease']
datasets = ['ionosphere','synthetic_athlete','synthetic_disease']
seed_int = 54321
step = 0.01
train_fraction = 0.7

for data_str in datasets:
    data = load_dataset(data_str, seed_int, train_fraction, step)
    model = Model(data)
    num_instances = 1 # data.test_df.shape[0]
    for idx in range(num_instances):
        ioi = IOI(idx, data, model)
        ijuice_obj = Ijuice(data, model, ioi)
        
        print(f'Optimizer solution status: {ijuice_obj.optimizer.status}')
        print(f'Solution:')
        nodes = list(ijuice_obj.get_nodes(data, model))
        for i,j in ijuice_obj.A:
            if ijuice_obj.y[i,j].x > 0:
                print(f'y({i,j}): {ijuice_obj.y[i,j].x}')
                print(f'Node {i}: {nodes[i]}')
                print(f'Node {j}: {nodes[j]}')
        for i in ijuice_obj.C.keys():
            if ijuice_obj.x[i].x > 0:
                print(f'x({i}): {ijuice_obj.x[i].x}')
                print(f'Node {i}: {nodes[i]}')




