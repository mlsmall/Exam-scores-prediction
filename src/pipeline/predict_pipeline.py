import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from exception import CustomException
from utils import load_object # to load pickle files

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts","trained_model.pkl") # -> 'artifacts\\trained_model.pkl'
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl') # -> 'artifacts\\preprocessor.pkl'
            model = load_object(model_path) 
            preprocessor = load_object(preprocessor_path) 
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
    
        except CustomException as e:
            raise CustomException(e, sys)
        
        
class CustomData:
    """This class maps the values from the HTML document to the corresponding backend values"""
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, lunch: str, test_preparation_course: str,
                 reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_dataframe(self):
        """This method has the mapping variable"""
        custom_data_input_dict = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score]
            }
        
        return pd.DataFrame(custom_data_input_dict)
        
        
        