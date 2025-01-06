import os
import sys
import json

from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

print(MONGO_DB_URL)

import certifi
ca=certifi.where()

"""
Certifi is a python package that provides set root certificates. Commonly used by python libraries that need to make a proper and secure http connection.
certifi.where() returns the path to the certificate file.
"""

import pandas as pd

import numpy as np
import pymongo
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

class NetworkDataExtract():   # ETL pipeline responsible for extracting data
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def csv_to_json_converter(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop = True, inplace = True)
            records= list(json.loads(data.T.to_json()).values())   #This converts every features which are in the row to list of records in the form of key-value pair
            # records = json.load(data.to_json(orient='records'))   #This converts every row to list of records in the form of key-value pair
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    
    def insert_data_to_mongodb(self, records, database, collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            
            return(len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e,sys)

if __name__=="__main__":
    
    FILE_PATH= "Network_Data\phisingData.csv"
    DATABASE = "Ganesh"
    Collection = "NetworkData"
    networkobj = NetworkDataExtract()
    records= networkobj.csv_to_json_converter(file_path=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_to_mongodb(records, DATABASE, Collection)
    print(no_of_records)

    
