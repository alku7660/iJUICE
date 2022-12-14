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
from autorank import autorank, plot_stats
from address import results_plots, load_obj
from tester import datasets, methods 

distances = ['euclidean','L1','L1_L0','L1_L0_inf','prob']
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
    elif method == 'face_knn':
        method = 'FACE'
    elif method == 'dice':
        method = 'DICE'
    elif method == 'mace':
        method = 'MACE'
    elif method == 'cchvae':
        method = 'CCHVAE'
    elif method == 'jce_prox':
        method = 'JUICEP'
    elif method == 'jce_spar':
        method = 'JUICES'
    return method

def proximity_plots():
    """
    Method to obtain 3x2 proximity plots per dataset
    """
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True)
    ax = ax.flatten()
    for i in range(len(datasets)):
        dataset = dataset_name(datasets[i])
        for j in range(len(distances)):
            distance = distances[j].capitalize()
            for k in range(len(methods)):
                method = method_name(methods[k])




