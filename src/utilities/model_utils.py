import pandas as pd
import joblib

def save_model(model_savepath, params, shap_test, shap_df, X, y_pred, y_proba, cf, f1):
    
    to_save = {'params': params,
               'shap_test': shap_test,
               'shap_df': shap_df,
               'val_idx': X.index.values, 
               'y_pred': y_pred,
               'y_proba': y_proba,
               'confusion_matrix': cf, 
               'f1': f1}
    
    joblib.dump(to_save, model_savepath)