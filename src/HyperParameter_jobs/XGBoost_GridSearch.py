### RANDOM SEED ###
seed = 0



### IMPORT UTILITIES MODULES ###
import sys
import os

utils_path = os.path.abspath("./utilities/")
sys.path.append(utils_path)
from load_data import load_fullECAI
from evaluation import *

### REQUIRED LIBRARIES ###
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from xgboost import XGBClassifier

def estimate_time(n_splits, n_repeats, param_grid):
    n_cores = 16
    ## Number of models to fit
    models = 1
    for listElem in list(param_grid.values()):
         models = models * len(listElem)
    
    seconds = models * n_splits * n_repeats * 10 / n_cores

    print('Tiempo estimado: {} hrs.'.format(segs/60/60))

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
scale_pos_weight = [1, 10, 25, 50, 75, 99] # Incluir menores a 1
max_depth = range(4,11)
gamma = [0, .5, 1]
learning_rate = [.0001, .001, .01, .1, 1] # Reducir el rango
n_estimators = [100, 200, 500]

param_grid = dict(scale_pos_weight=scale_pos_weight, 
                  max_depth=max_depth, 
                  gamma=gamma, 
                  learning_rate=learning_rate, 
                  n_estimators=n_estimators)

# Define evaluation procedure
n_splits = 5
n_repeats = 3
cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

# Define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc', verbose=10)
estimate_time(n_splits, n_repeats, param_grid)
grid.fit(X_train, y_train)

### SAVE 
import joblib
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%d-%m-%y_%H:%M:%S")

if not os.path.exists('results/XGBoost'):
    os.makedirs('figures/XGBoost')
joblib.dump(grid, 'GridSearchCV1_{}.pkl'.format(current_time))