## Basic imports
import os
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mlflow


## Load and split dataset
IrisData = load_iris(as_frame=True)
try:
    IrisDataset = pd.read_csv('iris.csv')
except:
    IrisDataset = IrisData.frame
    IrisDataset.to_csv('iris.csv',index=False)

TrainDataset , IrisDataset = train_test_split(IrisDataset,test_size=0.2,random_state=8013)
TestDataset , ValidDataset = train_test_split(IrisDataset,test_size=1/2,random_state=8013)

Features = load_iris().feature_names
Target = ['target']


## Dump processed datasets
Path('./Datasets').mkdir(parents=True,exist_ok=True)
TrainDataset[Features].to_csv('./Datasets/train_X.csv',index=False)
TrainDataset[Target].to_csv('./Datasets/train_y.csv',index=False)
TestDataset[Features].to_csv('./Datasets/test_X.csv',index=False)
TestDataset[Target].to_csv('./Datasets/test_y.csv',index=False)
ValidDataset[Features].to_csv('./Datasets/valid_X.csv',index=False)
ValidDataset[Target].to_csv('./Datasets/valid_y.csv',index=False)


## Init datasets paths 
DatasetPath = Path('./Datasets')
Train_X_DatasetPath = DatasetPath/'train_X.csv'
Test_X_DatasetPath = DatasetPath/'test_X.csv'
Valid_X_DatasetPath = DatasetPath/'valid_X.csv'
Train_y_DatasetPath = DatasetPath/'train_y.csv'
Test_y_DatasetPath = DatasetPath/'test_y.csv'
Valid_y_DatasetPath = DatasetPath/'valid_y.csv'


## Setup MLFlow client
MLFLOW_SERVER_URI = os.environ.get('MLFLOW_SERVER_URI','http://localhost:5000/')
mlflow.set_tracking_uri(MLFLOW_SERVER_URI) # This commando set both tracking server and model registry (default is equal to tracking) URIs


## Load datasets to a MLFlow experiment as artifacts
try:
    ExperimentID = mlflow.create_experiment('Iris Example')
except:
    ExperimentID = mlflow.get_experiment_by_name('Iris Example').experiment_id

with mlflow.start_run(run_name='load-datasets',experiment_id=ExperimentID):
    mlflow.log_artifact(Train_X_DatasetPath.absolute())
    mlflow.log_artifact(Train_y_DatasetPath.absolute())
    mlflow.log_artifact(Test_X_DatasetPath.absolute())
    mlflow.log_artifact(Test_y_DatasetPath.absolute())
    mlflow.log_artifact(Valid_X_DatasetPath.absolute())
    mlflow.log_artifact(Valid_y_DatasetPath.absolute())