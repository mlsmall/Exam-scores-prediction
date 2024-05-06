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
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbors": KNeighborsRegressor() 
            }
            
            logging.info("Training and evaluating models")
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)
        
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