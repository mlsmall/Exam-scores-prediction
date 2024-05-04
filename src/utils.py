import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import dill

import numpy as np
import pandas as p

from sklearn.model_selection import train_test_split

from exception import CustomException

def save_object(obj, file_path):
    """This function saves an object in a pickle file"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_object:
            dill.dump(obj, file_object) # when we dump an object, it will be saved in a pickle file in the specified path
    
    except Exception as e:
        raise CustomException(e, sys)