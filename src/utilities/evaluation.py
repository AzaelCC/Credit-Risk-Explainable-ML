# spot check machine learning algorithms on the german credit dataset
from sklearn.metrics import fbeta_score, roc_auc_score, make_scorer, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from numpy import min, max, mean

# calculate f2-measure
def f2_measure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
 
# evaluate a model
def evaluate_model(X, y, model, metric=f2_measure, splits=5, reps=3, seed=None):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=reps, random_state=seed)
    # define the model evaluation metric
    scorer = make_scorer(metric)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring=scorer, cv=cv, n_jobs=-1)
    
    print('min:', min(scores))
    print('max:', max(scores))
    print('mean:', mean(scores))
    
    return scores