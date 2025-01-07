from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.entity.config_entity import DataIngestionConfig, DataValidationConfig
from network_security.entity.config_entity import TrainingPipelineConfig
import sys

if __name__=="__main__":
    try:
       
        trainingpipelineconfig= TrainingPipelineConfig()
        dataingestionconfig= DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        
        logging.info("Initiate the Data ingestion")
        
        dataingestionartifact= data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion is completed")
        print(dataingestionartifact)
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, data_validation_config)
        logging.info("Initaite the data validation")
        data_validation_artifact= data_validation.initiate_data_validation()
        logging.info('Data Validation Completed')
        print(data_validation_artifact)
        
        
        
    
    
    except Exception as e:
        raise NetworkSecurityException(e,sys)