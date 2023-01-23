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
import pickle
# from autorank import autorank, plot_stats
from address import results_plots, load_obj
# from tester import datasets, methods, distance_type, lagranges 

datasets = ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','heart','synthetic_athlete','synthetic_disease']
methods = ['nn','mo','ft','rt','gs','face','dice','mace','juice','ijuice']
distances = ['euclidean','L1','L_inf','L1_L0','L1_L0_L_inf','prob']
mean_prop = dict(marker='D',markeredgecolor='firebrick',markerfacecolor='firebrick', markersize=2)
lagranges = np.linspace(start=0, stop=1, num=11)
general_distance = 'euclidean'
general_lagrange = 1

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

def proximity_plots():
    """
    Obtains a 3x2 proximity plots for all datasets
    """
    for i in range(len(datasets)):
        fig, ax = plt.subplots(nrows=3, ncols=2, sharex=False, figsize=(8,11))
        dataset = dataset_name(datasets[i])
        ax = ax.flatten()
        for j in range(len(distances)):
            distance = distance_name(distances[j])
            all_distance_measures = []
            for k in range(len(methods)):
                method = method_name(methods[k])
                eval = load_obj(f'{datasets[i]}_{methods[k]}_{general_distance}_{general_lagrange}.pkl')
                distance_measures = [eval.proximity_dict[idx][distances[j]] for idx in eval.proximity_dict.keys()]
                all_distance_measures.append(distance_measures)
            ax[j].boxplot(all_distance_measures, showmeans=True, meanprops=mean_prop)
            ax[j].set_title(distance)
            ax[j].set_xticklabels([method_name(n) for n in methods])
            plt.xticks(rotation=45)
        fig.suptitle(dataset)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
        fig.savefig(f'{results_plots}{dataset}_proximity_plot.pdf')

def feasibility_justification_time_plots(metric_name):
    """
    Obtains a 5x3 feasibility, justification, and time plots for all datasets
    """
    fig, ax = plt.subplots(nrows=5, ncols=3, sharex=False, sharey=True, figsize=(8,11))
    ax = ax.flatten()
    for i in range(len(datasets)):
        dataset = dataset_name(datasets[i])
        all_metric_measures = []
        for k in range(len(methods)):
            method = method_name(methods[k])
            eval = load_obj(f'{datasets[i]}_{methods[k]}_{general_distance}_{general_lagrange}.pkl')
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
            all_metric_measures.append(metric_measures)
        ax[i].boxplot(all_metric_measures, showmeans=True, meanprops=mean_prop)
        ax[i].set_xticklabels([method_name(i) for i in methods])
        ax[i].set_ylabel(dataset)
        if metric_name == 'time':
            ax[i].set_yscale('log')
    plt.suptitle(metric_name.capitalize())
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.35, hspace=0.2)
    fig.savefig(f'{results_plots}{metric_name}_plot.pdf')
    
def ablation_lagrange_plot():
    """
    Obtains an ablation plot where both the distances and the justifier ratio are plotted for iJUICE
    """
    fig, ax = plt.subplots(nrows=5, ncols=3, sharex=True, figsize=(8,11))
    ax = ax.flatten()
    dist = 'L1_L0'
    for i in range(len(datasets)):
        dataset = dataset_name(datasets[i])
        justifier_ratio_mean_list = []
        justifier_ratio_low_list = []
        justifier_ratio_high_list = []
        distance_mean_list = []
        distance_low_list = []
        distance_high_list = []
        for lagrange in lagranges:
            eval = load_obj(f'{dataset}_ijuice_{dist}_{lagrange}.pkl')
            justifier_ratio_mean, justifier_ratio_std = np.mean(eval.justifier_ratio.values()), np.std(eval.justifier_ratio.values())
            distance_measures = [eval.proximity_dict[idx][dist] for idx in eval.proximity_dict.keys()]
            distance_mean, distance_std = np.mean(distance_measures), np.std(distance_measures)
            justifier_ratio_mean_list.append(justifier_ratio_mean)
            justifier_ratio_low_list.append(justifier_ratio_mean - justifier_ratio_std)
            justifier_ratio_high_list.append(justifier_ratio_mean + justifier_ratio_std)
            distance_mean_list.append(distance_mean)
            distance_low_list.append(distance_mean - distance_std)
            distance_high_list.append(distance_mean + distance_std)
        ax[i].plot(lagranges, justifier_ratio_mean_list, color='blue')
        ax[i].fill_between(lagranges, justifier_ratio_low_list, justifier_ratio_high_list, color='blue', alpha=0.2)
        ax[i].set_xticklabels([method_name(i) for i in methods])
        ax[i].set_ylabel('Justification Ratio')
        ax[i].set_title(dataset)
        secax = ax[i].secondary_yaxis()
        secax.plot(lagranges, distance_mean_list, color='red')
        secax.fill_between(lagranges, distance_low_list, distance_high_list, color='red', alpha=0.2)
        secax.set_ylabel(dist.capitalize())
    fig.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.95, wspace=0.01, hspace=0.05)
    fig.savefig(f'{results_plots}_lagrange_ablation_plot.pdf')

proximity_plots()
feasibility_justification_time_plots('feasibility')
feasibility_justification_time_plots('justification')
feasibility_justification_time_plots('time')



