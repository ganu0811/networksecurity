import os, sys
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from network_security.constants.training_pipeline import TARGET_COLUMN
from network_security.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from network_security.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from network_security.entity.config_entity import DataTransformationConfig
from network_security.utils.main_utils.utils import save_object, save_numpy_array_data


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        
        try:
            self.data_validation_artifact:DataValidationArtifact= data_validation_artifact
            self.data_transformation_config : DataTransformationConfig= data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) 
    
    @staticmethod
    
    def read_data(file_path)-> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            df.drop(columns=["_id"], axis =1, inplace = True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    
    ## KNN Imputer
    def get_data_transformer_object(cls)-> Pipeline:
        """
        It initializes the KNN Imputer object with the parameters specified in the training_pipeline.py file 
        and returns a pipeline object with the KNN Imputer object as a first step
        
        Args:
            cls: DataTransformation class object
        
        Returns:
            A Pipeline Object
        
        """
        
        logging.info("Entered the get_data_transformer_object method of DataTransformation class")
        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)   ## ** consider the values in key-value pairs
            logging.info(f"Initialized the KNN Imputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor:Pipeline=Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)        
    
    
    
    def initiate_data_transformation(self)-> DataTransformationArtifact:
        logging.info(f"Data Tranformation initiated")
        
        try:
            logging.info("Starting the data transformation process")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            # print(train_df.dtypes)
            
            ## Training DataFrame
            
            input_feature_train_df = train_df.drop(columns = [TARGET_COLUMN], axis =1)
            print(input_feature_train_df.dtypes)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0) ## Replacing -1 in the target column in dataset with 0, One hot encoding
            
        
            ## Test Dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis =1)
            target_featue_test_df = test_df[TARGET_COLUMN]
            target_featue_test_df = target_featue_test_df.replace(-1, 0) ## Replacing -1 in the target column in dataset with 0, One hot encoding
            
            preprocessor= self.get_data_transformer_object() ## Calling a function inside a class
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            # print(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df) 
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
            
            # Combining the transformed input feature and target(output) feature arrays
            # np.c_ is used to combine the arrays
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_featue_test_df)]
            
            # save the numpy array data
            
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array = train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array = test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            
            ##preparing artifacts
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            )
            
            return data_transformation_artifact
            
            
        except Exception as e:
            raise NetworkSecurityException(e, sys) 

