import sys
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    artifact_dir = os.path.join("artifacts")
    ingested_train_path = os.path.join(artifact_dir, "application_train.csv")
    transformed_train_file_path = os.path.join(artifact_dir, 'train.npy')
    transformed_test_file_path = os.path.join(artifact_dir, 'test.npy')
    transformed_train_csv_path = os.path.join(artifact_dir, 'transformed_train.csv')
    transformed_test_csv_path = os.path.join(artifact_dir, 'transformed_test.csv')
    transformed_object_file_path = os.path.join(artifact_dir, 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.utils = MainUtils()

    def initiate_data_transformation(self):
        logging.info("Starting data transformation process")
        try:
            df = pd.read_csv(self.config.ingested_train_path)
            logging.info(f"data read successfully from {self.config.ingested_train_path}")
            # Drop SK_ID_CURR
            if 'SK_ID_CURR' in df.columns:
                df = df.drop(columns=['SK_ID_CURR'])
                logging.info("Dropped SK_ID_CURR column")
            X = df.drop(columns=['TARGET'])
            y = df['TARGET']
            logging.info("Split features and target variable")

            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            logging.info("Completed train-test split")
            logging.info(f"X_Train shape: {X_train.shape}, X_Test shape: {X_test.shape}")

            # Identify columns
            categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object'
                                and col not in ['SK_ID_CURR']]
            numeric_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
            logging.info(f"categorical columns: {categorical_cols}")
            logging.info(f"numeric columns: {numeric_cols}")

            # Numeric pipeline
            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            X_train_num = numeric_pipeline.fit_transform(X_train[numeric_cols])
            X_test_num = numeric_pipeline.transform(X_test[numeric_cols])

            # Categorical encoding
            X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
            X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
            logging.info("Applied one-hot encoding to categorical columns")
            # Align test set to train set
            X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)
            logging.info("Aligned categorical features between train and test sets")
            # Combine
            X_train_processed = np.hstack([X_train_num, X_train_cat.values])
            X_test_processed = np.hstack([X_test_num, X_test_cat.values])
            logging.info("Combined numeric and categorical features")

            # Save as .npy for ML pipeline (TARGET is not last column, so save separately)
            np.save(self.config.transformed_train_file_path, {'X': X_train_processed, 'y': y_train.values})
            np.save(self.config.transformed_test_file_path, {'X': X_test_processed, 'y': y_test.values})
            logging.info(f"Saved processed train to {self.config.transformed_train_file_path}")
            # Prepare column names for CSV
            all_feature_names = numeric_cols + list(X_train_cat.columns)
            # Save as .csv for inspection with column names and TARGET as last column
            train_df_out = pd.DataFrame(X_train_processed, columns=all_feature_names)
            train_df_out['TARGET'] = y_train.values
            train_df_out.to_csv(self.config.transformed_train_csv_path, index=False)

            test_df_out = pd.DataFrame(X_test_processed, columns=all_feature_names)
            test_df_out['TARGET'] = y_test.values
            test_df_out.to_csv(self.config.transformed_test_csv_path, index=False)

            # Save preprocessor
            preprocessor = {
                'numeric_cols': numeric_cols,
                'categorical_cols': categorical_cols,
                'numeric_pipeline': numeric_pipeline,
                'categorical_columns': X_train_cat.columns.tolist()
            }
            self.utils.save_object(self.config.transformed_object_file_path, preprocessor)

            logging.info(f"Saved preprocessor object to {self.config.transformed_object_file_path}")
            return (
                self.config.transformed_train_file_path,
                self.config.transformed_test_file_path,
                self.config.transformed_object_file_path
            )
        except Exception as e:
            logging.error(f"error occcured during data transformation: {str(e)}")
            raise CustomException(e, sys)