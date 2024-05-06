import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import dill

import numpy as np
import pandas as p

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from exception import CustomException

def save_object(file_path, obj):
    """This function saves an object in a pickle file"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file: # serialize (pickle) the object
            dill.dump(obj, file) # when we dump an object, it will be saved in a pickle file in the specified file_path
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """This trains and evaluates the model"""
    try:
        report = {}
    
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train) # Train the model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)         
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score, train_model_score
            
        return(report)
    
    except Exception as e:
        raise CustomException(e, sys)