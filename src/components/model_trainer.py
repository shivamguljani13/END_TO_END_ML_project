import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utlis import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    # Path where the trained model will be saved
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    # Path to save the trained model
    # Path where the model artifacts will be saved

class ModelTrainer: 
    """
     This class is responsible for training the machine learning model 
    Model Trainer class to train and evaluate machine learning models.
    This class handles the training of various regression models, evaluates their performance,"""
    def __init__(self):
        # Initialize the model trainer configuration
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        
        """ This method initiates the model training process.
        Args:
            train_array (numpy.ndarray): The training data array.
            test_array (numpy.ndarray): The testing data array.
        Returns:
            float: The R-squared score of the best model on the test data.
        """
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],#this will take all rows and all columns except last column as X_train_value
                train_array[:,-1], #this will make the last column as y_train_value
                test_array[:,:-1],#this will take all rows and all columns except last column as X_test_value
                test_array[:,-1]#this will make the last column as y_test_value
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                #"Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression()
            }

            model_report:dict=  evaluate_models( 
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test,
                                models=models)
            for model_name in models:
                print(f"Model: {model_name} \n {model_report[model_name]} \n")
            logging.info(f"Model Report: {model_report}")
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)