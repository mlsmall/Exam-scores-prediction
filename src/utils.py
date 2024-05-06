import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import dill

import numpy as np
import pandas as p

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from exception import CustomException

def save_object(file_path, obj):
    """This function saves an object as a pickle file"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file: # serialize (pickle) the object
            dill.dump(obj, file) # when we dump an object, it will be saved in a pickle file in the specified file_path
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    This function trains and evaluates the model using the R2 score.
    It returns a dictionary with the model name and the R2 score for the train and test datasets.
    """
    try:  
        report = {}
    
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = list(params.values())[i]
             
            gs = GridSearchCV(model, param, cv=3) # cv = cross validation
            gs.fit(X_train, y_train) # grid search using the model parameters
            
            model.set_params(**gs.best_params_) # sets the best parameters to the model
            model.fit(X_train, y_train) # Train the model
             
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)         
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score, train_model_score
            
        return(report)
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    """This function loads a pickle file"""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)