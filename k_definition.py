import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from evaluator_constructor import distance_calculation
from main import train_fraction, seed_int, step
from address import results_k_definition, results_plots, dataset_dir, save_obj, load_obj
from sklearn.neural_network import MLPClassifier 
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LinearLocator
# import seaborn as sns
from data_constructor import load_dataset
from model_constructor import Model
from ioi_constructor import IOI
from counterfactual_constructor import Counterfactual
from evaluator_constructor import Evaluator
from main import seed_int
# color_cmap=sns.diverging_palette(30, 250, l=65, center="dark", as_cmap=True)

def get_idx_cf(data):
    """
    Gets the indices of the counterfactual instances
    """
    train_label, desired_label = data.train_target, int(1 - data.undesired_class)
    idx_counterfactuals = np.where(train_label == desired_label)[0]
    return idx_counterfactuals

def calculate_cf_train_distance_matrix(data, idx_counterfactuals, type):
    """
    Computes the distance matrix among the counterfactual training observations
    """
    train = data.transformed_train_np
    counterfactual_train = train[idx_counterfactuals,:]
    dist = np.zeros((len(counterfactual_train), len(counterfactual_train)))
    for idx_i in range(len(counterfactual_train)-1):
        i = counterfactual_train[idx_i]
        for idx_j in range (idx_i+1, len(counterfactual_train)):
            j = counterfactual_train[idx_j]
            dist[idx_i, idx_j] = distance_calculation(i, j, data, type)
            dist[idx_j, idx_i] = dist[idx_i, idx_j]
    return dist

def estimate_outliers(data, type, neighbors):
    """
    Estimates how many outliers are there in the counterfactual class
    """
    dist, idx_counterfactuals = load_obj(f'dist_matrix_{data.name}_{type}', results_k_definition), load_obj(f'idx_counterfactuals_{data.name}', results_k_definition)
    cl = LocalOutlierFactor(n_neighbors=neighbors, metric='precomputed')
    outliers_labels = cl.fit_predict(dist)
    original_data_idx_outliers = idx_counterfactuals[np.where(outliers_labels == -1)]
    outliers = data.transformed_train_np[original_data_idx_outliers]
    return outliers

def class_0_points():
    """
    Creates the points for class 0
    """
    square1nx, square1ny = 8, 8
    square1x_limit_bot, square1x_limit_top = 0, 0.7
    square1y_limit_bot, square1y_limit_top = 0, 0.7
    square1x_coord, square1y_coord = np.linspace(square1x_limit_bot, square1x_limit_top, square1nx), np.linspace(square1y_limit_bot, square1y_limit_top, square1ny)
    square1x, square1y = np.meshgrid(square1x_coord, square1y_coord)
    square1 = np.vstack([square1x.ravel(), square1y.ravel()]).T
    
    square2nx, square2ny = 2, 3
    square2x_limit_bot, square2x_limit_top = 0, 0.1
    square2y_limit_bot, square2y_limit_top = 0.8, 1.0
    square2x_coord, square2y_coord = np.linspace(square2x_limit_bot, square2x_limit_top, square2nx), np.linspace(square2y_limit_bot, square2y_limit_top, square2ny)
    square2x, square2y = np.meshgrid(square2x_coord, square2y_coord)
    square2 = np.vstack([square2x.ravel(), square2y.ravel()]).T

    square3 = np.array([[0.6,0.8], [0.7,0.8], [0.8,0.8], [0.9,0.8], [1.0,0.8]])

    square4nx, square4ny = 3, 4
    square4x_limit_bot, square4x_limit_top = 0.8, 0.1
    square4y_limit_bot, square4y_limit_top = 0.4, 0.7
    square4x_coord, square4y_coord = np.linspace(square4x_limit_bot, square4x_limit_top, square4nx), np.linspace(square4y_limit_bot, square4y_limit_top, square4ny)
    square4x, square4y = np.meshgrid(square4x_coord, square4y_coord)
    square4 = np.vstack([square4x.ravel(), square4y.ravel()]).T

    square5 = np.array([[0.4,0.8], [0.4,0.9], [0.4,1.0], [0.5,0.8], [0.5,0.9], [0.5,1.0]])

    all_squares = np.concatenate((square1, square2, square3, square4, square5), axis=0)
    return all_squares

def class_1_points(seed):
    """
    Creates the points for class 1
    """
    np.random.seed(seed)

    size_blob1 = 2
    size_blob2 = 5
    size_blob3 = 50

    blob1_x = np.random.normal(loc=0.8, scale=0.02, size=(size_blob1, 1))
    blob1_y = np.random.normal(loc=0.9, scale=0.02, size=(size_blob1, 1))
    blob1 = np.concatenate((blob1_x, blob1_y), axis=1)

    blob2_x = np.random.normal(loc=0.3, scale=0.02, size=(size_blob2, 1))
    blob2_y = np.random.normal(loc=0.9, scale=0.02, size=(size_blob2, 1))
    blob2 = np.concatenate((blob2_x, blob2_y), axis=1)
    
    blob3_x = np.random.normal(loc=0.8, scale=0.05, size=(size_blob3, 1))
    blob3_y = np.random.normal(loc=0.2, scale=0.05, size=(size_blob3, 1))
    blob3 = np.concatenate((blob3_x, blob3_y), axis=1)

    all_blobs = np.concatenate((blob1, blob2, blob3), axis=0)
    return all_blobs

def point_of_interest():
    """
    Returns the point of interest
    """
    point = np.array([0.65,0.85])
    return point

def training_set(seed):
    """
    Returns the training dataset
    """
    class0 = class_0_points()
    class1 = class_1_points(seed)
    target0, target1 = np.zeros(class0.shape[0]), np.ones(class1.shape[0])
    X = np.concatenate((class0, class1), axis=0)
    Y = np.concatenate((target0, target1), axis=0)
    return X, Y

def train_model(X, Y, seed):
    """
    Trains a simple Neural Network model to obtain a classifier distinguishing between the two classes 
    """
    np.random.seed(seed)
    f = MLPClassifier([100, 200, 500, 200, 100], activation='tanh', solver='adam')
    f.fit(X, Y)
    print(f'Accuracy of model: {f.score(X, Y)}')
    return f

def store_data_set(seed):
    """
    Stores the dataset created
    """
    X, Y = training_set(seed)
    ioi = point_of_interest().reshape(1, 2)
    ioi_label = np.array([0])
    X = np.concatenate((X, ioi), axis=0)
    Y = np.concatenate((Y, ioi_label), axis=0)
    df = pd.DataFrame(data=X, columns=['x','y'])
    df['label'] = Y
    df.to_csv(f'{dataset_dir}synthetic_2d/synthetic_2d.csv')
    print('Dataset created')

def plot_dataset(f, X, Y, ioi):
    """
    Plots the 2D dataset
    """
    fig_2d, ax_2d = plt.subplots(figsize=(2, 2))
    N = 100
    all_x = np.linspace(min(X[:,0]), max(X[:,0]), N)
    all_y = np.linspace(min(X[:,1]), max(X[:,1]), N)
    x_grid, y_grid = np.meshgrid(all_x, all_y)
    mesh = np.vstack((x_grid.ravel(), y_grid.ravel())).T
    all_y = f.predict(mesh)
    X_label1 = X[np.where(Y == 1)[0]]
    Y_label1 = Y[np.where(Y == 1)[0]]

    ax_2d.scatter(mesh[:,0], mesh[:,1], s=10, c=all_y, cmap=color_cmap)
    ax_2d.scatter(X_label1[:,0], X_label1[:,1], s=10, c='blue', linewidths=0.5, edgecolors='green')
    ax_2d.scatter(ioi[0], ioi[1], s=16, c='red', marker='x')
    ax_2d.set_xlim((min(X[:,0]), max(X[:,0])))
    ax_2d.set_ylim((min(X[:,1]), max(X[:,1])))
    ax_2d.axes.xaxis.set_visible(True)
    
    ax_2d.plot(0.9, 0.95, marker='${}$'.format(1), markersize=8, markeredgecolor='black', markeredgewidth=0.2, label=1, color='black')
    ax_2d.plot(0.2, 0.95, marker='${}$'.format(2), markersize=8, markeredgecolor='black', markeredgewidth=0.2, label=1, color='black')
    ax_2d.plot(0.925, 0.4, marker='${}$'.format(3), markersize=8, markeredgecolor='black', markeredgewidth=0.2, label=1, color='black')

    ax_2d.xaxis.set_ticks(ticks=np.arange(0, 1.1, 0.1), labels=['0','','','','','0.5','','','','','1'])
    ax_2d.yaxis.set_ticks(ticks=np.arange(0, 1.1, 0.1), labels=['0','','','','','0.5','','','','','1'])
    fig_2d.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.95, wspace=0.02, hspace=0.0)
    fig_2d.tight_layout()
    fig_2d.savefig(f'{results_plots}synthetic_2d.pdf')

def ijuice_varying_k(idx, data_str, distance, k_list):
    t = 100
    method_str = 'ijuice'
    lagrange = 0.01
    data = load_dataset(data_str, train_fraction, seed_int, step)
    model = Model(data)
    data.undesired_test(model)
    eval = Evaluator(data, method_str, distance, lagrange)
    ioi = IOI(idx, data, model, distance)
    # f = model.model
    # x = ioi.x[0]
    # X, Y = data.transformed_train_np, data.train_target
    # plot_dataset(f, X, Y, x)
    
    for k in k_list:
        cf_gen = Counterfactual(data, model, method_str, ioi, distance, lagrange, t=t, k=k)
        eval.add_specific_x_data(cf_gen)
        print(f'Data {data_str.capitalize()} | Method {method_str.capitalize()} | Type {distance.capitalize()} | lagrange {str(lagrange)} | K number {k} | Proximity (distance) {eval.proximity_dict[idx]}')
        save_obj(eval, results_k_definition, f'{data_str}_{method_str}_{distance}_{str(lagrange)}_k_{k}.pkl')

idx = 0 # 150 for synthetic_2d, 0 for the others
data_str = 'dutch' # 'synthetic_2d', 'dutch', 'diabetes', 'oulad', 'athlete'
distance = 'L1_L0' # 'euclidean', 'L1_L0'
range_k_values = range(1, 21) # 'range(1, 58)', 'range(1, 21)' 
ijuice_varying_k(idx, data_str, distance, range_k_values)

# store_data_set(seed_int)
# X, Y = training_set(seed_int)
# f = train_model(X, Y, seed_int)
# ioi = point_of_interest()
# plot_dataset(f, X, Y, ioi)

# datasets = ['adult','kdd_census','diabetes','oulad',] # ,'adult','kdd_census','diabetes','oulad','german','dutch','bank','credit','compass','diabetes','student','oulad','law','heart','synthetic_disease'
# distance_type = ['L1_L0','L1_L0_L_inf','prob']

# for data_str in datasets:
#     data = load_dataset(data_str, train_fraction, seed_int, step)
#     idx_counterfactuals = get_idx_cf(data)
#     save_obj(idx_counterfactuals, results_k_definition, f'idx_counterfactuals_{data_str}')
#     for type in distance_type:
#         dist = calculate_cf_train_distance_matrix(data, idx_counterfactuals, type)
#         save_obj(dist, results_k_definition, f'dist_matrix_{data_str}_{type}')
#         print(f'Dataset: {data_str}, Distance: {type}, Total instances: {data.train_target[np.where(data.train_target == 1 - data.undesired_class)[0]].shape[0]}')


# for data_str in datasets:
#     data = load_dataset(data_str, train_fraction, seed_int, step)
#     total_cf_instances = data.train_target[np.where(data.train_target == 1 - data.undesired_class)[0]].shape[0]
#     range_neighbors = range(int(total_cf_instances*0.05), int(total_cf_instances*0.2))
#     pandas_results_type = pd.DataFrame(index=distance_type, columns=range_neighbors)
#     for type in distance_type:
#         for neighbors in range_neighbors:
#             outliers = estimate_outliers(data, type, neighbors)
#             print(f'Dataset: {data_str}, Distance: {type}, Neihgbors: {neighbors}, Total instances: {total_cf_instances}, Number of outliers: {len(outliers)}')
#             suggested_k = len(outliers) + 1
#             pandas_results_type.loc[type, neighbors] = suggested_k
#     pandas_results_type.to_csv(results_k_definition+'k_definition.csv')





