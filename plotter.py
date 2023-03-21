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
from address import results_plots, load_obj
from model_constructor import Model
from evaluator_constructor import distance_calculation, verify_feasibility
from itertools import product
from scipy.stats import norm

datasets = ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law','heart','synthetic_athlete','synthetic_disease']
distances = ['L1_L0','L1_L0_L_inf','prob']
methods = ['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice','ijuice']
lagranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
colors_list = ['red', 'blue', 'green', 'purple', 'lightgreen', 'tab:brown', 'orange']
mean_prop = dict(marker='D', markeredgecolor='firebrick', markerfacecolor='firebrick', markersize=2)
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
            eval_extra = load_obj(f'{datasets[i]}_{methods[k]}_{dist}_1_extra.pkl')
            if metric_name == 'feasibility':
                metric_measures = list(eval.feasibility_dict.values()) + list(eval_extra.feasibility_dict.values())
                new_metric_measures = []
                for n in metric_measures:
                    if n:
                        value = 1
                    else:
                        value = 0
                    new_metric_measures.extend([value])
                metric_measures = new_metric_measures
            elif metric_name == 'justification':
                metric_measures = list(eval.justifier_ratio.values()) + list(eval_extra.justifier_ratio.values())
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
                eval_extra = load_obj(f'{datasets[j]}_ijuice_{distances[i]}_{lagrange}_extra.pkl')
                justifier_ratio_mean = np.mean(list(eval.justifier_ratio.values())+list(eval_extra.justifier_ratio.values()))
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

def print_instances(dataset, distance):

    def find_potential_justifiers(data, ioi, ioi_label):
        """
        Finds the set of training observations belonging to, and predicted as, the counterfactual class
        """
        train_np = data.transformed_train_np
        train_target = data.train_target

        potential_justifiers = train_np[train_target != ioi_label]
        sort_potential_justifiers = []
        for i in range(potential_justifiers.shape[0]):
            dist = distance_calculation(potential_justifiers[i], ioi, data, type=distance)
            sort_potential_justifiers.append((potential_justifiers[i], dist))
        sort_potential_justifiers.sort(key=lambda x: x[1])
        sort_potential_justifiers = [i[0] for i in sort_potential_justifiers]
        if len(sort_potential_justifiers) > 100:
            sort_potential_justifiers = sort_potential_justifiers[:100]
        return sort_potential_justifiers

    def nn_list(data, ioi, potential_justifiers):
        """
        Method that gets the list of training observations labeled as cf-label with respect to the cf, ordered based on graph nodes size
        """
        permutations_potential_justifiers = []
        for i in range(len(potential_justifiers)):
            possible_feat_values_justifier_i = get_feat_possible_values(data, ioi, points=[potential_justifiers[i]])[0]
            len_permutations = len(list(product(*possible_feat_values_justifier_i)))
            permutations_potential_justifiers.append((potential_justifiers[i], len_permutations))
        permutations_potential_justifiers.sort(key=lambda x: x[1])
        permutations_potential_justifiers = [i[0] for i in permutations_potential_justifiers]
        if len(permutations_potential_justifiers) > 10:
            permutations_potential_justifiers = permutations_potential_justifiers[:10]
        return permutations_potential_justifiers
    
    def get_feat_possible_values(data, ioi, points):
        """
        Method that obtains the features possible values
        """
        pot_justifier_feat_possible_values = {}
        normal_x = ioi
        for k in range(len(points)):
            potential_justifier_k = points[k]
            v = normal_x - potential_justifier_k
            nonzero_index = list(np.nonzero(v)[0])
            feat_checked = []
            feat_possible_values = []
            for i in range(len(normal_x)):
                if i not in feat_checked:
                    feat_i = data.processed_features[i]
                    if feat_i in data.bin_enc_cols:
                        if i in nonzero_index:
                            value = [potential_justifier_k[i], normal_x[i]]
                        else:
                            value = [potential_justifier_k[i]]
                        feat_checked.extend([i])
                    elif feat_i in data.cat_enc_cols:
                        idx_cat_i = data.idx_cat_cols_dict[feat_i[:-4]]
                        nn_cat_idx = list(potential_justifier_k[idx_cat_i])
                        if any(item in idx_cat_i for item in nonzero_index):
                            ioi_cat_idx = list(normal_x[idx_cat_i])
                            value = [nn_cat_idx, ioi_cat_idx]
                        else:
                            value = [nn_cat_idx]
                        feat_checked.extend(idx_cat_i)
                    elif feat_i in data.ordinal:
                        if i in nonzero_index:
                            values_i = list(data.processed_feat_dist[feat_i].keys())
                            max_val_i, min_val_i = max(normal_x[i], potential_justifier_k[i]), min(normal_x[i], potential_justifier_k[i])
                            value = [j for j in values_i if j <= max_val_i and j >= min_val_i]
                        else:
                            value = [potential_justifier_k[i]]
                        feat_checked.extend([i])
                    elif feat_i in data.continuous:
                        if i in nonzero_index:
                            max_val_i, min_val_i = max(normal_x[i], potential_justifier_k[i]), min(normal_x[i], potential_justifier_k[i])
                            value = continuous_feat_values(i, min_val_i, max_val_i, data)
                        else:
                            value = [potential_justifier_k[i]]
                        feat_checked.extend([i])
                    feat_possible_values.append(value)
            pot_justifier_feat_possible_values[k] = feat_possible_values
        return pot_justifier_feat_possible_values

    def continuous_feat_values(i, min_val, max_val, data):
        """
        Method that defines how to discretize the continuous features
        """
        sorted_feat_i = list(np.sort(data.transformed_train_np[:,i][(data.transformed_train_np[:,i] >= min_val) & (data.transformed_train_np[:,i] <= max_val)]))
        value = list(np.unique(sorted_feat_i))
        if len(value) <= 100:
            if min_val not in value:
                value = [min_val] + value
            if max_val not in value:
                value = value + [max_val]
            return value
        else:
            mean_val, std_val = np.mean(data.transformed_train_np[:,i]), np.std(data.transformed_train_np[:,i])
            percentiles_range = list(np.linspace(0, 1, 101))
            value = []
            for perc in percentiles_range:
                value.append(norm.ppf(perc, loc=mean_val, scale=std_val))
            value = [val for val in value if val >= min_val and val <= max_val]
            if min_val not in value:
                value = [min_val] + value
            if max_val not in value:
                value = value + [max_val]
        return value

    data = load_dataset(dataset, train_fraction, seed_int, step)
    eval = load_obj(f'{dataset}_ijuice_{distance}_1.pkl')
    model = Model(data)
    max_justification_idx = max(eval.justifier_ratio, key=eval.justifier_ratio.get)
    ioi = eval.normal_x_dict[max_justification_idx]
    ioi_original = eval.x_dict[max_justification_idx]
    cf = eval.x_cf_dict[max_justification_idx]
    justifiers = eval.justifiers_dict[max_justification_idx]
    potential_justifiers = find_potential_justifiers(data, ioi, ioi_label=0)
    nn_potential_justifiers = nn_list(data, ioi, potential_justifiers)
    justifiers_original = []
    for i in justifiers:
        justifier_original = data.inverse(nn_potential_justifiers[i-1])
        justifiers_original.extend(justifier_original)
    justifiers_original = pd.DataFrame(data=justifiers_original, columns=data.features)
    print('IOI:')
    print(ioi_original)
    print('CF:')
    print(cf)
    print('Justifiers:')
    print(justifiers_original)

# proximity_plots()
# feasibility_justification_time_plots('feasibility')
# feasibility_justification_time_plots('justification')
# feasibility_justification_time_plots('time')
# scatter_proximity_var('feasibility')
# scatter_proximity_var('justification')
# ablation_lagrange_plot()
# count_instances()
print_instances('adult','prob')


