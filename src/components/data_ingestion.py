import os
import sys
from src.exceptions import Custom_exception
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer




@dataclass
##Using dataclass allows to directly define class variable (Only use for variables)
class DataIngestionConfig:
    train_data_path : str = os.path.join("./artifact", "train.csv")
    test_data_path : str = os.path.join("./artifact", "test.csv")
    raw_data_path : str= os.path.join("./artifact", "raw.csv")



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion Method!!")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the CSV File!!")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info("Train-Test Split Initiated!!")

            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            train_set.to_csv(self.ingestion_config.train_data_path)
            test_set.to_csv(self.ingestion_config.test_data_path)
            logging.info("Train Test Data Ingestion Completed!!")

            return (self.ingestion_config.train_data_path, 
                    self.ingestion_config.test_data_path)
        except Exception as e:
            raise Custom_exception(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformer = DataTransformation()
    data_transformer.apply_sentence_transformer(train_data, test_data)
    pca_train_data, pca_test_data = data_transformer.apply_pca()
    
    modeltrainer = ModelTrainer()
    print("R-Squared of the Model: ", modeltrainer.initiateModelTrainer(pca_train_data, pca_test_data))

            










