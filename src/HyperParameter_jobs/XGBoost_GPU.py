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

import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from xgboost import XGBClassifier

### LOAD DATA ###
dataset = load_fullECAI()
# Prep data
X = dataset.drop('status', axis=1)
y = dataset.loc[:, 'status']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)

### MODEL ###
model = XGBClassifier(use_label_encoder=False, tree_method='gpu_hist', gpu_id=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f1_score(y_pred, y_test))