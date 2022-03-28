### RANDOM SEED ###
seed = 0

### IMPORT UTILITIES MODULES ###
import sys
import os
import joblib
from datetime import datetime
from time import time

utils_path = os.path.abspath("./utilities/")
sys.path.append(utils_path)
from load_data import load_fullECAI
from evaluation import *

### Read GRID commandline options
import argparse
parser = argparse.ArgumentParser()

# CV parameters
parser.add_argument('-n', '--n_jobs', type=int, required=True)
parser.add_argument('-s', '--n_splits', type=int, required=True)
parser.add_argument('-r', '--n_repeats', type=int, required=True)

# XGBoost parameters
parser.add_argument('--scale_pos_weight', nargs='+', type=float)
parser.add_argument('--max_depth', nargs='+', type=int)
parser.add_argument('--gamma', nargs='+', type=float)
parser.add_argument('--learning_rate', nargs='+', type=float)
parser.add_argument('--n_estimators', nargs='+', type=int)

args = parser.parse_args()
param_grid = vars(args)
param_grid = {k: v for k, v in param_grid.items() if v is not None} # Remove nonsupplied values

n_jobs = param_grid.pop('n_jobs')
n_splits = param_grid.pop('n_splits')
n_repeats = param_grid.pop('n_repeats')

### Logging ###
start = datetime.now()
start_str = start.strftime("%d-%m-%y_%H-%M-%S")

if not os.path.exists('results/XGBoost/GridSearchCV_{}'.format(start_str)):
    os.makedirs('results/XGBoost/GridSearchCV_{}'.format(start_str))

sys.stdout = open("results/XGBoost/GridSearchCV_{}/GridSearchCV_{}.txt".format(start_str,start_str),"w")   

print('START: {}'.format(start_str))
print('seed={}\n'.format(seed))
print('Grid={}\n'.format(param_grid))

### REQUIRED LIBRARIES ###
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from xgboost import XGBClassifier

import shap

def estimate_time(n_splits, n_repeats, param_grid):
    avg_secs_model = 10
    n_cores = 12
    ## Number of models to fit
    models = 1
    for listElem in list(param_grid.values()):
         models = models * len(listElem)
    
    seconds = models * n_splits * n_repeats * avg_secs_model / n_cores

    print('Tiempo estimado: {} hrs.\n'.format(seconds/60/60))

    
def log_metrics_and_explanations(clf, X_val, y_true_val, y_pred_val):
    cf = confusion_matrix(y_true_val, y_pred_val)
    f1 = f1_score(y_true_val, y_pred_val)
    
    explainer = shap.Explainer(clf)
    shap_test = explainer(X_val)
    shap_df = pd.DataFrame(shap_test.values, 
                           columns=shap_test.feature_names, 
                           index=X_val.index)
    to_save = {'params': clf.get_params(),
               'shap': shap_df,
               'val_idx': X_val.index.values, 
               'y_pred': y_pred_val,
               'confusion_matrix': cf, 
               'f1': f1}
    
    ### Save to file
    timestring = "".join(str(time()).split("."))
    model_name = 'xgb_' + timestring + '.pkl'
    
    if not os.path.exists('results/XGBoost/GridSearchCV_{}/models'.format(start_str)):
        os.makedirs('results/XGBoost/GridSearchCV_{}/models'.format(start_str))
    model_savepath = 'results/XGBoost/GridSearchCV_{}/models/{}'.format(start_str, model_name)
    joblib.dump(to_save, model_savepath)
    
    return f1
    
    
def _my_scorer(clf, X_val, y_true_val):
    # do all the work and return some of the metrics
    y_pred_val = clf.predict(X_val)
    f1 = log_metrics_and_explanations(clf, X_val, y_true_val, y_pred_val)
    
    return f1

### LOAD DATA ###
dataset = load_fullECAI()
# Prep data
X = dataset.drop('status', axis=1)
y = dataset.loc[:, 'status']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)

### MODEL ###
model = XGBClassifier(use_label_encoder=False, tree_method='gpu_hist', gpu_id=0)

### GRID SEARCH ###
# Define Grid
# scale_pos_weight = [0.8, 1, 5, 8, 10] # Incluir menores a 1. 8 sería el recomendado para el desbalanceo en la base ECAI
# max_depth = range(3,10)
# gamma = [0, .5, 1]
# learning_rate = [.01, 0.5, 0.1, 0.2, 0.5] # Reducir el rango
# n_estimators = [100, 200, 500]

# Define evaluation procedure
print('n_splits={}\n_repeats={}\n'.format(n_splits, n_repeats))
cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

# Define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=n_jobs, cv=cv, scoring=_my_scorer, verbose=10)
estimate_time(n_splits, n_repeats, param_grid)

grid.fit(X_train, y_train)

### SAVE ###
joblib.dump(grid, 'results/XGBoost/GridSearchCV_{}/GridSearchCV_{}.pkl'.format(start_str, start_str))

end = datetime.now()
print('Elapsed time: {}'.format(end - start))

end_str = end.strftime("%d-%m-%y_%H-%M-%S")
print('END: {}'.format(end_str))
sys.stdout.close()