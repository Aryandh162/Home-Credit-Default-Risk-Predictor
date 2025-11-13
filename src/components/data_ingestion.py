import os 
import pandas as pd
import sys
from src.constant import application_train_path, application_test_path, artifact_folder 
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    artifact_folder: str = "artifacts"
    train_file_name: str = "application_train.csv"
    
    
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            os.makedirs(self.config.artifact_folder, exist_ok=True)
            logging.info(f"Created artifact folder at: {self.config.artifact_folder}")
            dst_path = os.path.join(self.config.artifact_folder, self.config.train_file_name)
            logging.info(f"Copying training data to: {dst_path}")  
            df = pd.read_csv(application_train_path)
            df.to_csv(dst_path, index=False)
            logging.info(f"data saveed to {dst_path}")
            logging.info('data ingested successfully')
        except Exception as e:
            logging.error(f"error occurred during data ingestion: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
    logging.info("Data Ingestion completed successfully.")
            