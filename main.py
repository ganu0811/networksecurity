from network_security.components.data_ingestion import DataIngestion
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.entity.config_entity import DataIngestionConfig
from network_security.entity.config_entity import TrainingPipelineConfig
import sys

if __name__=="__main__":
    try:
       
        trainingpipelineconfig= TrainingPipelineConfig()
        dataingestionconfig= DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        
        logging.info("Initiate the Data ingestion")
        
        dataingestionartifact= data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
    
    
    except Exception as e:
        raise NetworkSecurityException(e,sys)