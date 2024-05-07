import os
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from exception import CustomException
from logger import logging
from utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "trained_model.pkl")
    
class ModelTrainer:
    """
    This class is responsible for training and evaluating machine learning models.
    """
    def __init__(self):
        # Configuration object containing the path to save the trained model.
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Creating the training and testing inputs for model")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], # X_training data # 2D Numpy array [rows, columns]
                train_array[:, -1], # y_training data
                test_array[:, :-1], # X_test data
                test_array[:, -1] # y_test data
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(), # Symmetric trees, more efficient for categories
                "CatBoost": CatBoostRegressor(), # Asymetric trees
                "Linear Regression": LinearRegression(),
            }
            
            params = {
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # "splitter": ["best", "random"],
                    # "max_features": ["sqrt", "log2"]
                },
                "Gradient Boosting": {
                    # "loss": ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.6, 0.7, 0.8, 0.9],
                    "n_estimators":  [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost": {
                    # "loss": ["linear", "square", "exponential"],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "n_estimators":  [8, 16, 32, 64, 128, 256]
                }, 
                "XGBoost": {
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "n_estimators":  [8, 16, 32, 64, 128, 256]
                },
                "CatBoost": {
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "depth": [6, 8, 10],
                    'iterations': [30, 50, 100]
                },
                "Linear Regression": {}
             }
            
            logging.info("Training and evaluating models")
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            #print(model_report)
        
            # To get the best model score and name from the model report dictionary
            best_model_name = max(model_report, key=model_report.get) # key=model_report.get will return the key with the highest value
            best_model_score = model_report[best_model_name][0] # model_report returns [test_model_score, train_model_score]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException(f"The best model is {best_model_name} with a score of {best_model_score}. No good model found")
            logging.info(f"The best model is {best_model_name} with a score of {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Saving the trained model")
            
            prediction = best_model.predict(X_test)
            r2_square = r2_score(y_test, prediction) # r2_square is the coefficient of determination using the best model
            
            return r2_square
            
        except CustomException as e:
            raise CustomException(e, sys)