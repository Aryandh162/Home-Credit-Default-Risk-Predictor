import sys
from typing import Dict, Tuple
import os
import pandas as pd
import pickle
import yaml



from src.constant import *
from src.exception import CustomException
from src.logger import logging



class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            logging.error(f"Error reading YAML file: {filename}")
            raise CustomException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))

            return schema_config

        except Exception as e:
            logging.error("Error reading schema configuration file.")
            raise CustomException(e, sys) from e

    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        logging.info(f"saveing object to {file_path}")


        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)


            logging.info(f"Object saved to {file_path} successfully")


        except Exception as e:
            logging.error(f"Error saving object to {file_path}")
            raise CustomException(e, sys) from e


    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info(f"Loading object from {file_path}")

        try:
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)
                
            logging.info(f"Object loaded from {file_path} successfully")
            return obj

        except Exception as e:
            raise CustomException(e, sys) from e
   
    @staticmethod    
    def load_object(file_path):
        try:
            with open(file_path,'rb') as file_obj:
                return pickle.load(file_obj)
        except Exception as e:
            logging.info('Exception Occured in load_object function utils')
            raise CustomException(e,sys)
   