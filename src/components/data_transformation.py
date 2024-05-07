# Feature Engineering and Data Cleaning

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from exception import CustomException
from logger import logging

from utils import save_object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = os.path.join('artifacts', 'preprocessor.pkl')
        
    def get_data_transformer_object(self):
        """
        This function is responsible for transforming the data
        """
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course']
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), # handling missing values
                ('scaler', StandardScaler()) # standard scaling
            ])
            
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # handling missing values
                ('one_hot_encoder', OneHotEncoder()), # one hot encoding
                ('scaler', StandardScaler(with_mean=False)) # standard scaling
            ])
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns enconding completed")
            
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, numerical_columns),
                    ("categorical_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
    
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Created the train and test dataframes")
            
            logging.info("Creating the preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = 'math_score'
            numerical_columns = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying the preprocessor object on the train and test dataframes.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saving the preprocessor object")
            save_object(file_path=self.data_transformation_config, obj=preprocessing_obj)
            
            return train_arr, test_arr
        
        except Exception as e:
            raise CustomException(e, sys)