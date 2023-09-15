"""
Plotter for analysis of all CFs
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from data_constructor import load_dataset
# from autorank import autorank, plot_stats
from address import results_plots, load_obj, results_k_definition
from model_constructor import Model
from evaluator_constructor import distance_calculation, verify_feasibility
from itertools import product
from scipy.stats import norm

datasets = ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law','heart','synthetic_athlete','synthetic_disease']
distances = ['L1_L0','L1_L0_L_inf','prob']
methods = ['nn','mo','ft','rt','gs','face','dice','cchvae','juice','ijuice'] # 'mace'
lagranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
colors_list = ['red', 'blue', 'green', 'purple', 'lightgreen', 'tab:brown', 'orange']
mean_prop = dict(marker='D', markeredgecolor='firebrick', markerfacecolor='firebrick', markersize=2)
general_distance = 'euclidean'
general_lagrange = 1
step = 0.01
train_fraction = 0.7
seed_int = 54321

def dataset_name(name):
    """
    Method to output dataset names to be printed
    """
    if name == 'adult':
        name = 'Adult'
    elif name == 'kdd_census':
        name = 'Census'
    elif name == 'german':
        name = 'German'
    elif name == 'dutch':
        name = 'Dutch'
    elif name == 'bank':
        name = 'Bank'
    elif name == 'credit':
        name = 'Credit'
    elif name == 'compass':
        name = 'Compas'
    elif name == 'diabetes':
        name = 'Diabetes'
    elif name == 'student':
        name = 'Student'
    elif name == 'oulad':
        name = 'Oulad'
    elif name == 'law':
        name = 'Law'
    elif name == 'heart':
        name = 'Heart'
    elif name == 'synthetic_athlete':
        name = 'Athlete'
    elif name == 'synthetic_disease':
        name = 'Disease'
    return name

def method_name(method):
    """
    Method that changes the names of the methods
    """
    if method == 'nn':
        method = 'NN'
    elif method == 'mo':
        method = 'MO'
    elif method == 'ft':
        method = 'FT'
    elif method == 'rt':
        method = 'RT'
    elif method == 'gs':
        method = 'GS'
    elif method == 'face':
        method = 'FACE'
    elif method == 'dice':
        method = 'DICE'
    elif method == 'mace':
        method = 'MACE'
    elif method == 'cchvae':
        method = 'CCHVAE'
    elif method == 'juice':
        method = 'JUICE'
    elif method == 'ijuice':
        method = 'iJUICE'
    return method

def distance_name(distance):
    """
    Method that changes the names of the methods
    """
    if distance == 'euclidean':
        distance = 'L2'
    elif distance == 'L1':
        distance = 'L1'
    elif distance == 'L_inf':
        distance = 'L$\infty$'
    elif distance == 'L1_L0':
        distance = 'L1 & L0'
    elif distance == 'L1_L0_L_inf':
        distance = 'L1, L0 & L$\infty$'
    elif distance == 'prob':
        distance = 'Max. Percentile Shift'
    return distance

def feasibility_justification_time_sparsity_plots(metric_name):
    """
    Obtains a 5x3 feasibility, justification, and time plots for all datasets
    """
    fig, ax = plt.subplots(nrows=7, ncols=2, sharex=False, sharey=True, figsize=(7,10))
    ax = ax.flatten()
    for i in range(len(datasets)):
        dataset = dataset_name(datasets[i])
        all_metric_measures = []
        for k in range(len(methods)):
            if methods[k] == 'ijuice':
                dist = 'L1_L0'
            else:
                dist = 'euclidean'
            eval = load_obj(f'{datasets[i]}_{methods[k]}_{dist}_1.pkl')
            if metric_name == 'feasibility':
                metric_measures = list(eval.feasibility_dict.values())
                new_metric_measures = []
                for n in metric_measures:
                    if n:
                        value = 1
                    else:
                        value = 0
                    new_metric_measures.extend([value])
                metric_measures = new_metric_measures
            elif metric_name == 'justification':
                metric_measures = list(eval.justifier_ratio.values())
            elif metric_name == 'time':
                metric_measures = list(eval.time_dict.values())
                if isinstance(metric_measures[0], np.ndarray):
                    metric_measures = [list(i)[0] for i in metric_measures]
            elif metric_name == 'sparsity':
                metric_measures = [] 
                keys_index_instances = list(eval.x_dict.keys())
                for idx in keys_index_instances:
                    x = eval.x_dict[idx]
                    x_cf = eval.x_cf_dict[idx]
                    sparsity = calculate_sparsity(x, x_cf)
                    metric_measures.append(sparsity)
            all_metric_measures.append(metric_measures)
        ax[i].boxplot(all_metric_measures, showmeans=True, meanprops=mean_prop, showfliers=False)
        ax[i].set_xticklabels([method_name(i) for i in methods], rotation=25)
        ax[i].set_ylabel(dataset)
        ax[i].grid(axis='y', linestyle='--', alpha=0.4)
        if metric_name == 'time':
            ax[i].set_yscale('log')
    plt.suptitle(f'{metric_name.capitalize()} (s)' if metric_name == 'time' else metric_name.capitalize())
    fig.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.96, wspace=0.15, hspace=0.4)
    fig.savefig(f'{results_plots}{metric_name}_plot.pdf')

def calculate_sparsity(x, cf):
    """
    Calculates sparsity of the CF with respect to the ioi
    """
    return len(np.where(x != cf)[0])

def sparsity_plots(metric_name):
    """
    Obtains a 7x2 sparsity plots for all datasets
    """
    fig, ax = plt.subplots(nrows=7, ncols=2, sharex=False, sharey=True, figsize=(7,10))
    ax = ax.flatten()
    for i in range(len(datasets)):
        dataset = dataset_name(datasets[i])
        all_metric_measures = []
        for k in range(len(methods)):
            if methods[k] == 'ijuice':
                dist = 'L1_L0'
            else:
                dist = 'euclidean'
            eval = load_obj(f'{datasets[i]}_{methods[k]}_{dist}_1.pkl')
            if metric_name == 'sparsity':
                new_metric_measures = [] 
                keys_index_instances = list(eval.x_dict.keys())
                for idx in keys_index_instances:
                    x = eval.x_dict[idx]
                    x_cf = eval.x_cf_dict[idx]
                    metric_measures = calculate_sparsity(x, x_cf)
                    new_metric_measures.append(metric_measures)
                
                
                for n in metric_measures:
                    if n:
                        value = 1
                    else:
                        value = 0
                    new_metric_measures.extend([value])
                metric_measures = new_metric_measures
            elif metric_name == 'justification':
                metric_measures = list(eval.justifier_ratio.values())
            elif metric_name == 'time':
                metric_measures = list(eval.time_dict.values())
                if isinstance(metric_measures[0], np.ndarray):
                    metric_measures = [list(i)[0] for i in metric_measures]
            all_metric_measures.append(metric_measures)
        ax[i].boxplot(all_metric_measures, showmeans=True, meanprops=mean_prop, showfliers=False)
        ax[i].set_xticklabels([method_name(i) for i in methods], rotation=25)
        ax[i].set_ylabel(dataset)
        ax[i].grid(axis='y', linestyle='--', alpha=0.4)
        if metric_name == 'time':
            ax[i].set_yscale('log')
    plt.suptitle(f'{metric_name.capitalize()} (s)' if metric_name == 'time' else metric_name.capitalize())
    fig.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.96, wspace=0.15, hspace=0.4)
    fig.savefig(f'{results_plots}{metric_name}_plot.pdf')

def ablation_lagrange_plot():
    """
    Obtains an ablation plot where both the distances and the justifier ratio are plotted for iJUICE
    """
    fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(7,4.5))
    start = 0
    end = 1.1
    for i in range(len(distances)):
        for j in range(len(datasets)):
            dataset = dataset_name(datasets[j])
            justifier_ratio_mean_list = []
            distance_mean_list = []
            for lagrange in lagranges:
                eval = load_obj(f'{datasets[j]}/{datasets[j]}_ijuice_{distances[i]}_{lagrange}.pkl')
                justifier_ratio_mean = np.mean(list(eval.justifier_ratio.values()))
                print(f'Dataset: {dataset.upper()}, # of instances: {len(list(eval.justifier_ratio.values()))}')
                distance_measures = [eval.proximity_dict[idx][distances[i]] for idx in eval.proximity_dict.keys()]
                distance_mean = np.mean(distance_measures)
                justifier_ratio_mean_list.append(justifier_ratio_mean)
                distance_mean_list.append(distance_mean)
            ax[i,j].plot(lagranges, justifier_ratio_mean_list, color='#5E81AC', label='Justification')
            ax[i,j].grid(axis='both', linestyle='--', alpha=0.4)
            ax[i,j].yaxis.set_ticks(np.arange(start, end, 0.2))
            ax[i,j].yaxis.set_tick_params(labelcolor='#5E81AC')
            ax[i,j].xaxis.set_ticks(ticks=np.arange(start, end, 0.1), labels=['0','','','','','0.5','','','','','1'])
            secax = ax[i,j].twinx()
            secax.plot(lagranges, distance_mean_list, color='#BF616A', label='Distance')
            secax.yaxis.set_tick_params(labelcolor='#BF616A')
            secax.yaxis.set_ticks(np.arange(min(distance_mean_list),max(distance_mean_list),(max(distance_mean_list)-min(distance_mean_list))*0.2))
            ax[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            secax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for j in range(len(datasets)):
        dataset = dataset_name(datasets[j])
        ax[0,j].set_title(dataset)
    for i in range(len(distances)):
        ax[i,-1].set_ylabel(distance_name(distances[i]), labelpad=30, color='#BF616A')
        ax[i,-1].yaxis.set_label_position("right")
    fig.supxlabel('$\lambda$ Weight Parameter')
    fig.supylabel('Average Justification Ratio', color='#5E81AC')
    fig.suptitle(f'Average Distance and Average Justification Ratio vs. $\lambda$')
    fig.text(0.965, 0.5, 'Average Distance', color='#BF616A', va='center', rotation='vertical')
    fig.subplots_adjust(left=0.09, bottom=0.1, right=0.875, top=0.9, wspace=0.475, hspace=0.2)
    fig.savefig(f'{results_plots}lagrange_ablation_plot.pdf')

def plot_k_definition(data_str, distance, range_k):
    """
    Plots the results of the definition of K for the synthetic 2d dataset
    """
    method_str = 'ijuice'
    lagrange = 0.01
    proximity = []
    justifier_ratio = []
    for k in range_k:
        eval = load_obj(f'{data_str}_{method_str}_{distance}_{str(lagrange)}_k_{k}.pkl', results_k_definition)
        proximities = [eval.proximity_dict[idx][distance] for idx in eval.proximity_dict.keys()]
        mean_proximity = np.mean(proximities)
        justifier_ratio_mean = np.mean(list(eval.justifier_ratio.values()))
        proximity.append(mean_proximity)
        justifier_ratio.append(justifier_ratio_mean)
    fig, ax = plt.subplots()
    ax.plot(range_k, justifier_ratio)
    secax = ax.twinx()
    secax.plot(range_k, proximity, color='#BF616A')
    ax.yaxis.set_tick_params(labelcolor='#5E81AC')
    secax.yaxis.set_tick_params(labelcolor='#BF616A')
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    fig.suptitle(f'Distance and Justification Ratio vs. $k$')
    fig.supxlabel(f'$k$ Parameter')
    fig.supylabel(f'Justification Ratio', color='#5E81AC')
    fig.text(0.965, 0.5, f'Average Distance ({distance_name(distance)})', color='#BF616A', va='center', rotation='vertical')
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95)
    # fig.tight_layout()
    fig.savefig(f'{results_k_definition}k_definition_{data_str}.pdf')

def print_instances_ijuice(dataset, distance, lagrange):

    eval = load_obj(f'{dataset}_ijuice_{distance}_{lagrange}.pkl')
    for idx in eval.x_dict.keys():    
        ioi = eval.normal_x_dict[idx]
        ioi_original = eval.x_dict[idx]
        cf = eval.x_cf_dict[idx]
        justifiers_original = eval.justifiers_dict[idx] 
        print('IOI:')
        print(ioi_original)
        print('CF:')
        print(cf)
        print('Justifiers:')
        print(justifiers_original)

def print_instances(dataset, method, distance, lagrange):
    if method == 'ijuice':
        print_instances_ijuice(dataset, distance, lagrange)
    else:
        eval = load_obj(f'{dataset}_{method}_{distance}_{lagrange}.pkl')
        indices = eval.x_dict.keys()
        for idx in indices:
            ioi_original = eval.x_dict[idx]
            cf = eval.x_cf_dict[idx]
            print('==========================================')
            print('IOI index:')
            print(idx)
            print('IOI:')
            print(ioi_original)
            print('CF:')
            print(cf)
            print('==========================================')


# proximity_plots()
# feasibility_justification_time_plots('feasibility')
# feasibility_justification_time_plots('justification')
# feasibility_justification_time_plots('time')
# ablation_lagrange_plot()
data_str='synthetic_athlete'
distance='L1_L0'
range_k_values = range(1, 23)
plot_k_definition(data_str, distance, range_k_values)
# print_instances('adult','ijuice','L1_L0', 0.5)





