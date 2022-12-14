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
plt.rcParams.update({'font.size': 12})
import matplotlib.patches as mpatches
import pickle
# from autorank import autorank, plot_stats
from address import results_plots, load_obj
from tester import datasets, methods 

distances = ['euclidean','L1','L1_L0','L1_L0_inf','prob']
mean_prop = dict(marker='D',markeredgecolor='firebrick',markerfacecolor='firebrick', markersize=2)
lagranges = np.linspace(start=0, stop=1, num=11)
general_distance = 'euclidean'
general_lagrange = 0.5

def dataset_name(name):
    """
    Method to output dataset names to be printed
    """
    if name == 'synthetic_simple':
        name = 'Simple'
    elif name == 'synthetic_severe_disease':
        name = 'Disease'
    elif name == 'synthetic_athlete':
        name = 'Athlete'
    else:
        name = name.capitalize()
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

def proximity_plots():
    """
    Obtains a 3x2 proximity plots for all datasets
    """
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(8,11))
    ax = ax.flatten()
    for i in range(len(datasets)):
        dataset = dataset_name(datasets[i])
        for j in range(len(distances)):
            distance = distances[j].capitalize()
            all_distance_measures = []
            for k in range(len(methods)):
                method = method_name(methods[k])
                eval = load_obj(f'{dataset}_{method}_{general_distance}_{general_lagrange}.pkl')
                distance_measures = [eval.proximity_dict[idx][distance] for idx in eval.proximity_dict.keys()]
                all_distance_measures.append(distance_measures)
            ax[i,j].boxplot(all_distance_measures, showmeans=True, meanprops=mean_prop)
            ax[i,j].set_xticklabels([method_name(i) for i in methods])
            ax[i,j].set_title(dataset)
            ax[i,j].set_ylabel(distance)
    fig.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.95, wspace=0.01, hspace=0.05)
    fig.savefig(results_plots+'proximity_plots.pdf')

def feasibility_justification_time_plots(metric_name):
    """
    Obtains a 5x3 feasibility, justification, and time plots for all datasets
    """
    fig, ax = plt.subplots(nrows=5, ncols=3, sharex=True, figsize=(8,11))
    ax = ax.flatten()
    for i in range(len(datasets)):
        dataset = dataset_name(datasets[i])
        all_metric_measures = []
        for k in range(len(methods)):
            method = method_name(methods[k])
            eval = load_obj(f'{dataset}_{method}_{general_distance}_{general_lagrange}.pkl')
            if metric_name == 'feasibility':
                metric_measures = eval.feasibility_dict.values()
            elif metric_name == 'justification':
                metric_measures = eval.justifier_ratio.values()
            elif metric_name == 'time':
                metric_measures = eval.time_dict.values()
            all_metric_measures.append(metric_measures)
        ax[i].boxplot(all_metric_measures, showmeans=True, meanprops=mean_prop)
        ax[i].set_xticklabels([method_name(i) for i in methods])
        ax[i].set_ylabel(dataset)
    plt.suptitle(metric_name.capitalize())
    fig.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.95, wspace=0.01, hspace=0.05)
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




