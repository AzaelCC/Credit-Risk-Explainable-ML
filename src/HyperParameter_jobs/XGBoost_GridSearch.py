### RANDOM SEED ###
seed = 0

### IMPORT UTILITIES MODULES ###
import sys
import os
import joblib
from datetime import datetime

utils_path = os.path.abspath("./utilities/")
sys.path.append(utils_path)
from load_data import load_fullECAI
from evaluation import *

### Logging ###

if not os.path.exists('results/XGBoost'):
    os.makedirs('results/XGBoost')

sys.stdout=open("results/XGBoost/external_file.txt","w")   
print('seed={}\n'.format(seed))

### REQUIRED LIBRARIES ###
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from xgboost import XGBClassifier

def estimate_time(n_splits, n_repeats, param_grid):
    avg_secs_model = 10
    n_cores = 12
    ## Number of models to fit
    models = 1
    for listElem in list(param_grid.values()):
         models = models * len(listElem)
    
    seconds = models * n_splits * n_repeats * avg_secs_model / n_cores

    print('Tiempo estimado: {} hrs.\n'.format(seconds/60/60))

### LOAD DATA ###
dataset = load_fullECAI()
# Prep data
X = dataset.drop('status', axis=1)
y = dataset.loc[:, 'status']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)

### MODEL ###
model = XGBClassifier(use_label_encoder=False)

### GRID SEARCH ###
# Define Grid
scale_pos_weight = [0.8, 1, 5, 8, 10, 15] # Incluir menores a 1. 8 sería el recomendado para el desbalanceo en la base ECAI
max_depth = range(2,10)
gamma = [0, .5, 1]
learning_rate = [.01, 0.1, 0.2, 0.5, 1] # Reducir el rango
n_estimators = [100, 200, 500]

param_grid = dict(scale_pos_weight=scale_pos_weight, 
                  max_depth=max_depth, 
                  gamma=gamma, 
                  learning_rate=learning_rate, 
                  n_estimators=n_estimators)

print('Grid={}\n'.format(param_grid))


# Define evaluation procedure
n_splits = 5
n_repeats = 1
print('n_splits={}\nn_repeats={}\n'.format(n_splits, n_repeats))
cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

# Define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=12, cv=cv, scoring=make_scorer(f1_score), verbose=10)
estimate_time(n_splits, n_repeats, param_grid)

now = datetime.now()
print('Inicio: {}'.format(now.strftime("%d-%m-%y_%H-%M-%S")))

grid.fit(X_train, y_train)

### SAVE 
now = datetime.now()
current_time = now.strftime("%d-%m-%y_%H-%M-%S")
sys.stdout.close()

joblib.dump(grid, 'results/XGBoost/GridSearchCV1_{}.pkl'.format(current_time))

now = datetime.now()
print('Fin: {}'.format(now.strftime("%d-%m-%y_%H-%M-%S")))