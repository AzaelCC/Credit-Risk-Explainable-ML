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

    
    weights = g.es["weight"][:]
    sp = (1.0 / np.array(g.shortest_paths_dijkstra(weights=weights)))
    np.fill_diagonal(sp,0)
    N=sp.shape[0]
    ne= (1.0/(N-1)) * np.apply_along_axis(sum,0,sp)

    return ne

def mean_eff_from_distances(normalized_dist, cutoff):
    adj_matrix = normalized_dist.copy()
    too_far = adj_matrix > cutoff
    adj_matrix = adj_matrix.astype(object) 
    adj_matrix[too_far] = None # Remove edges between too far vertices
    reduced_G = Graph.Weighted_Adjacency(adj_matrix, mode='undirected', loops=False)
    edges = len(list(reduced_G.es))
    eff = nodal_eff(reduced_G)

    return cutoff, np.mean(eff), edges, reduced_G, adj_matrix

    
def reduce_dimension_efficiency(percent=0.8):
    """
    Plots global (mean) efficienty of network by cutting edges that are too far
    """
    # Estimated time with full graph: 2-3hrs
    cutoffs = np.array(range(1,11))/10
    mean_effs = []
    graphs = []
    
    results = Parallel(n_jobs=-1)(delayed(mean_eff_from_distances)(normalized_dist, co) for co in cutoffs)
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
                './results/Basic/models/RandomForestClassifier.pkl', 
                './results/Basic/models/LogisticRegression.pkl',
                './results/Basic/models/LinearDiscriminantAnalysis.pkl',
                './results/Basic/models/XGBClassifier.pkl']

for model in models_paths:
    model_results = joblib.load(model)
    shap_df = model_results['shap']
    shap_dist = euclidean_distances(shap_df1)
    normalized_dist = MinMaxScaler().fit_transform(shap_dist)

    effs, edges, optimal, results = reduce_dimension_efficiency(percent=0.8)
    optimal_cutoff, optimal_eff, optimal_adj, optimal_G = optimal
    
    # Dump data
    model_folder = '/'.join(model.split('/')[:-1])
    model_name = model.split('/')[-1].split('.')[0]
    joblib.dump(results, '{}/{}_reduced_graphs_efficiency.pkl'.format(model_folder, model_name))
    joblib.dump(optimal, '{}/{}_optimal_graph_efficiency.pkl'.format(model_folder, model_name))
    