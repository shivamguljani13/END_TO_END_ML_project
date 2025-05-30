# This module contains the DataTransformation class which is responsible for transforming the data
# and creating a preprocessor object that can be used to preprocess the data     

import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler 
from dataclasses import dataclass
from src.utlis import save_object


@dataclass 
class DataTransformationConfig:
# This class defines the configuration for data transformation
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    # This will store the path where the preprocessor will be saved
    #preprocessed_data_path: str = os.path.join('artifacts', 'preprocessed_data.csv')
    #transformed_data_path: str = os.path.join('artifacts', 'transformed_data.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        # This will create an instance of the DataTransformationConfig
        # class and store it in the data_transformation_config attribute  
    def get_data_transformer_object(self):
           
            '''
            this  function is responsible for data transformation baed on the numerical and categorical columns
            It will create a preprocessor object that will be used to preprocess the data
            
            '''
            
            try:
                numerical_columns = ["writing_score", "reading_score"]
                categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
                # These are the numerical and categorical columns in the dataset
                # We will use these columns to create the preprocessor object
                #create a pipeline for numerical columns
                num_pipeline = Pipeline(
                    steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
                )
                #this will run on the numerical columns of the  training dataset
                #create a pipeline for categorical columns
               
               
                cat_pipeline = Pipeline(
                    steps=[
                   ("imputer",SimpleImputer(strategy="most_frequent")),
                   ("one_hot_encoder",OneHotEncoder()),
                   ("scaler",StandardScaler(with_mean=False))
                ]
                )
                
                
                logging.info(f"Numerical columns {numerical_columns} are defined")
                logging.info(f"and categorical columns {categorical_columns} are defined")
                
                
                preprcoessor = ColumnTransformer(
                    [
                        ('num_pipeline', num_pipeline, numerical_columns),# this will apply the numerical pipeline to the numerical columns
                        ('cat_pipeline', cat_pipeline, categorical_columns)# this will apply the categorical pipeline to the categorical columns
                    ]
                )
                # This will create a preprocessor object that will apply the numerical and categorical pipelines to the respective columns
                logging.info("Preprocessor object created")
                return preprcoessor
            
            
            
            except Exception as e:
                raise CustomException(e, sys)
            # this will raise the custom exception if there is any error in the code    
     
    def initiate_data_transformation(self, train_path, test_path):
            logging.info("Entered the data transformation method or component")
            try:
                train_df = pd.read_csv(train_path)# reading the train data from the csv file
                test_df = pd.read_csv(test_path)# reading the test data from the csv file
                logging.info("Read the train and test dataframes")
                
                logging.info("Obtaining preprocessing object")
                # this will call the get_data_transformer_object function to create a preprocessor object
                
                preprocessing_obj = self.get_data_transformer_object()
                # this will create a preprocessor object that will be used to preprocess the data
                
                target_column_name = "math_score"# this is the target column in the dataset
                # this will be used to separate the target feature from the input features
                numerical_columns = ["writing_score", "reading_score"]# these are the numerical columns in the dataset
                
                input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)# dropping the target column from the train dataframe
                 # this will create a dataframe with only the input features
                target_feature_train_df = train_df[target_column_name]
                
                input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = test_df[target_column_name]
                
                logging.info(
                    f"applying preprocessing object on training and testing datasets: {input_feature_train_df.shape}, {input_feature_test_df.shape}"
                )
                
                
                # transform the training and testing data
                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)
                # this will apply the preprocessor object on the input features of the train and test dataframes
                
                train_arr = np.c_[
                    input_feature_train_arr,
                    np.array(target_feature_train_df)]# this will combine the input features and target feature into a single array
                # convert the input features and target feature to numpy arrays
                test_arr = np.c_[
                    input_feature_test_arr,
                    np.array(target_feature_test_df)]# this will combine the input features and target feature into a single array
                # convert the input features and target feature to numpy arrays
                
                
                logging.info("saved preprocessor object")
                # save the preprocessor object to the specified path    
                # convert the target feature to numpy array
                save_object(
                    
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj
               
                )# this will save the preprocessor object to the specified path
                
                
                logging.info("Data transformation completed")
                
                return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                )
            except Exception as e:
                raise CustomException(e, sys)