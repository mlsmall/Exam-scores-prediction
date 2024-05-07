import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer, ModelTrainerConfig


@dataclass # to define your data classes
class DataIngestionConfig: # inputs given to the data_ingestion.py component
    train_data_path: str=os.path.join("artifacts", "train.csv") # all the output will be stored in this folder 
    test_data_path: str=os.path.join("artifacts", "test.csv")
    raw_data_path: str=os.path.join("artifacts", "data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # The three paths from DataIngestionConfig class will be stored in this object
        
    def initiatite_data_ingestion(self):
        """ Function to read code from your database(mongoDB, SQL, etc) and create train and test csv files """
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv('notebooks/data/students.csv') # Or mongoDB or SQL
            logging.info('Read the datatset as dataframe') # logging is important to know in which line the exception happened     
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=7)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )          
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiatite_data_ingestion()
    
    data_transformation_obj = DataTransformation()
    train_arr, test_arr = data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_training(train_arr, test_arr))