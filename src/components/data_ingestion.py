import os 
import sys 
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


#any input that is required is given through this data ingestion config class
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')
    # we can add more parameters if required in future  
# it will tell components where to save the data and where to read the data from
# this is the config class for data ingestioncomponent

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        # this will create an instance of the config class and store it in the ingestion_config attribute
        # this will be used to access the parameters of the config class
        # so that we can use it in the data ingestion component
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/stud.csv')
            # reading the data from the csv file stored in the notebook/data folder locally on to the system
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            # this will create the directory if it does not exist
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            # this will save the data in the raw data path
            logging.info('Train test split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            # this will split the data into train and test set
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            # this will save the train and test set in the respective paths
            logging.info('Ingestion of the data is completed')
            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)
            # this will raise the custom exception if there is any error in the code    
            
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
    # this will create an instance of the data ingestion class and call the initiate_data_ingestion method to start the data ingestion process
    # this will be used to test the data ingestion component            