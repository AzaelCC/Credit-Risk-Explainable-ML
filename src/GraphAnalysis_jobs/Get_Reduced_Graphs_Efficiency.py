seed = 0

import os
import sys

utils_path = os.path.abspath("./utilities/")
sys.path.append(utils_path)

from load_data import load_fullECAI
from evaluation import *
from evaluation import _my_scorer

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, Normalizer
from sklearn.metrics import euclidean_distances
import joblib
from joblib import Parallel, delayed
import shap

from igraph import Graph
import igraph as ig


def nodal_eff(g):
    """
    Finds local nodal efficiency
    
    Source: https://stackoverflow.com/questions/56609704/how-do-i-calculate-the-global-efficiency-of-graph-in-igraph-python
    
    Parameters
    --------
    g : igraph.Graph
        Weighted Graph
    
    Returns
    --------
    ne : float
        Nodal efficiency
    
    Examples
    --------
    # Local efficiency
    >>> eff = nodal_eff(g)
    # Global efficiency
    >>> global_eff = mean(eff)
    """

    has_conections = bool(g.es.attributes())
    if has_conections:
        weights = g.es["weight"][:]
        sp = (1.0 / np.array(g.shortest_paths_dijkstra(weights=weights)))
        np.fill_diagonal(sp,0)
        N=sp.shape[0]
        ne= (1.0/(N-1)) * np.apply_along_axis(sum,0,sp)
        return ne
    else: 
        return -1 # Negative to ensure compatibilty

def mean_eff_from_distances(normalized_dist, cutoff, model_folder, model_name):
    """
    Creates a reduced graph from and adjaceny matrix (normalized_dist) where edges are cut at the given cutoff. 
    It also saves the graph to a file.
    """
    adj_matrix = normalized_dist.copy()
    too_far = adj_matrix > cutoff
    adj_matrix = adj_matrix.astype(object) 
    adj_matrix[too_far] = None # Remove edges between too far vertices
    reduced_G = Graph.Weighted_Adjacency(adj_matrix, mode='undirected', loops=False)
    edges = len(list(reduced_G.es))
    eff = nodal_eff(reduced_G)
    
    # Save individual reduced graph
    results = (cutoff, np.mean(eff), edges, reduced_G, adj_matrix)
    cutoff_int = int(cutoff * 100)
    joblib.dump(results, '{}/reduced_graphs/{}/reduced_{}.pkl'.format(model_folder, model_name, cutoff_int))
    
    
    return cutoff, np.mean(eff), edges, reduced_G, adj_matrix

    
def reduce_dimension_efficiency(percent, model_folder, model_name):
    """
    Plots global (mean) efficienty of network by cutting edges that are too far
    
    Parameters
    ----------
    percent: float 0-1
        Percent of minimum global efficiency from max global efficiency
    """
    #cutoffs = np.array(range(1,11))/10
    cutoffs_100 = np.array(range(1,101, 3))/100
    cutoffs_10 = np.array(range(1,11))/10
    cutoffs = np.append(cutoffs_100[0:17], cutoffs_10[4:11])
    
    mean_effs = []
    graphs = []
    
    results = Parallel(n_jobs=-1)(delayed(mean_eff_from_distances)(normalized_dist, co, model_folder, model_name) for co in cutoffs)
    cutoffs, effs, edges, graphs, adj_matrices = list(zip(*results))
    
    optimal_eff = np.max(effs) * percent
    for i, co in enumerate(cutoffs):
        if effs[i] > optimal_eff:
            optimal_cutoff = cutoffs[i]
            optimal_eff = effs[i]
            optimal_adj = adj_matrices[i]
            optimal_G = graphs[i]
            optimal_edges = edges[i]
            break
    
    print('New reduced Graph has {:.2%} edges of the fully connected'.format(optimal_edges/edges[-1]))
    optimals = (optimal_cutoff, optimal_eff, optimal_adj, optimal_G)
    
    return effs, edges, optimals, results


dataset = load_fullECAI()
# Prep data
X = dataset.drop('status', axis=1)
y = dataset.loc[:, 'status']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)

models_paths = ['./results/XGBoost/GridSearchCV_22-03-22_03-27-03/models/best/best_xgb.pkl', 
                './results/Basic1/models/RandomForestClassifier.pkl', 
                './results/Basic1/models/LogisticRegression.pkl',
                './results/Basic1/models/LinearDiscriminantAnalysis.pkl']

for model in models_paths:
    model_folder = '/'.join(model.split('/')[:-1])
    model_name = model.split('/')[-1].split('.')[0]
    
    model_results = joblib.load(model)
    shap_df = model_results['shap_df']
    shap_dist = euclidean_distances(shap_df)
    n = shap_df.shape[0]
    
    mm = MinMaxScaler()
    mm.fit(shap_dist.flatten().reshape(-1, 1))
    normalized_dist = mm.transform(shap_dist.flatten().reshape(-1, 1)).reshape(n, n)
    
    # Create appropieate folder
    path = '{}/reduced_graphs/{}/'.format(model_folder, model_name)
    if not os.path.exists(path):
        os.makedirs(path)


    effs, edges, optimal, results = reduce_dimension_efficiency(0.8, model_folder, model_name)
    optimal_cutoff, optimal_eff, optimal_adj, optimal_G = optimal
    
    # Dump data
    #joblib.dump(results, '{}/{}_reduced_graphs_efficiency.pkl'.format(model_folder, model_name))
    #joblib.dump(optimal, '{}/{}_optimal_graph_efficiency.pkl'.format(model_folder, model_name))
    