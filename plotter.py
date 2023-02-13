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
import pickle
# from autorank import autorank, plot_stats
from address import results_plots, load_obj
# from tester import datasets, methods, distance_type, lagranges 

# datasets = ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law','heart','synthetic_athlete','synthetic_disease']
# methods = ['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice','ijuice']
# general_distance = 'L1_L0'
# general_lagrange = 1
datasets = ['adult','kdd_census','credit','synthetic_disease']
distances = ['L1_L0','L1_L0_L_inf','prob']
methods = ['ijuice']
lagranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
colors_list = ['red', 'blue', 'green', 'purple', 'lightgreen', 'tab:brown', 'orange']
mean_prop = dict(marker='D', markeredgecolor='firebrick', markerfacecolor='firebrick', markersize=2)

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
            ax[j].boxplot(all_distance_measures, showmeans=True, meanprops=mean_prop, showfliers=False)
            ax[j].set_title(distance)
            ax[j].set_xticklabels([method_name(n) for n in methods], rotation=45)
            ax[j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.suptitle(dataset)
        fig.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.3)
        fig.savefig(f'{results_plots}{dataset}_proximity_plot.pdf')

def feasibility_justification_time_plots(metric_name):
    """
    Obtains a 5x3 feasibility, justification, and time plots for all datasets
    """
    fig, ax = plt.subplots(nrows=7, ncols=2, sharex=False, sharey=True, figsize=(7,10))
    ax = ax.flatten()
    for i in range(len(datasets)):
        dataset = dataset_name(datasets[i])
        all_metric_measures = []
        for k in range(len(methods)):
            method = method_name(methods[k])
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
                # new_metric_measures = []
                # for n in metric_measures:
                #     if n > 0.01:
                #         val = 1
                #     else:
                #         val = 0
                #     new_metric_measures.append(val)
                # metric_measures = new_metric_measures
            elif metric_name == 'time':
                metric_measures = list(eval.time_dict.values())
                if isinstance(metric_measures[0], np.ndarray):
                    metric_measures = [list(i)[0] for i in metric_measures]
                if methods[k] in ['nn','mo','ft','rt']:
                    metric_measures = [0.5*i for i in metric_measures]
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

def scatter_proximity_var(var):
    """
    Scatter plot between distance and feasibility (evidences trade-off between feasibility and distance)
    """
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(1.1,1.6))
    for i in range(len(datasets)):
        dataset = dataset_name(datasets[i])
        legend_elements = create_legend_distance()
        method_mean_var = []
        for k in range(len(methods)):
            method = method_name(methods[k])
            eval = load_obj(f'{datasets[i]}_{methods[k]}_{general_distance}_{general_lagrange}.pkl')
            if var == 'feasibility':
                metric_measures = list(eval.feasibility_dict.values())
            elif var == 'justification':
                metric_measures = list(eval.justifier_ratio.values())
            new_metric_measures = []
            for n in metric_measures:
                if n:
                    value = 1
                else:
                    value = 0
                new_metric_measures.extend([value])
            metric_measures = new_metric_measures
            method_mean_var.append(np.mean(metric_measures))
        for j in range(len(distances)):
            method_mean_distance = []
            distance = distance_name(distances[j])
            for k in range(len(methods)):
                method = method_name(methods[k])
                eval = load_obj(f'{datasets[i]}_{methods[k]}_{general_distance}_{general_lagrange}.pkl')
                distance_measures = [eval.proximity_dict[idx][distances[j]] for idx in eval.proximity_dict.keys()]
                method_mean_distance.append(np.mean(distance_measures))
            ax.scatter(x=method_mean_distance, y=method_mean_var, color=colors_list[j])
        ax.legend(handles=legend_elements)
        fig.savefig(f'{results_plots}_scatter_proximity_{var}_plot.pdf')

def ablation_lagrange_plot():
    """
    Obtains an ablation plot where both the distances and the justifier ratio are plotted for iJUICE
    """
    fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(7,4.5))
    start = 0
    end = 1.1
    for i in range(len(distances)):
        dist = distance_name(distances[i])
        for j in range(len(datasets)):
            dataset = dataset_name(datasets[j])
            justifier_ratio_mean_list = []
            distance_mean_list = []
            for lagrange in lagranges:
                eval = load_obj(f'{datasets[j]}_ijuice_{distances[i]}_{lagrange}.pkl')
                justifier_ratio_mean, justifier_ratio_std = np.mean(list(eval.justifier_ratio.values())), np.std(list(eval.justifier_ratio.values()))
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

def count_instances():
    fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(7,4.5))
    start = 0
    end = 1.1
    for i in range(len(distances)):
        for j in range(len(datasets)):
            dataset = dataset_name(datasets[j])
            justifier_ratio_mean_list = []
            distance_mean_list = []
            eval = load_obj(f'{datasets[j]}_ijuice_{distances[i]}_1.pkl')
            justifier_ratio_mean, justifier_ratio_std = np.mean(list(eval.justifier_ratio.values())), np.std(list(eval.justifier_ratio.values()))
            print(f'Dataset: {dataset.upper()}, Distance: {distances[i]}, # of instances: {len(list(eval.justifier_ratio.values()))}')

# proximity_plots()
# feasibility_justification_time_plots('feasibility')
# feasibility_justification_time_plots('justification')
# feasibility_justification_time_plots('time')
# scatter_proximity_var('feasibility')
# scatter_proximity_var('justification')
ablation_lagrange_plot()
# count_instances()


