## Basic imports
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mlflow


## Load and split dataset
IrisData = load_iris(as_frame=True)
IrisDataset = IrisData.frame

TrainDataset , IrisDataset = train_test_split(IrisDataset,test_size=0.2,random_state=8013)
TestDataset , EvalDataset = train_test_split(IrisDataset,test_size=1/2,random_state=8013)


## Init MLFlow datasets 
TrainMLDataset = mlflow.data.from_pandas(
    df = TrainDataset,
    source = 'https://archive.ics.uci.edu/dataset/53/iris',
    targets = 'target',
    name = 'iris-dataset'
)

TestMLDataset = mlflow.data.from_pandas(
    df = TestDataset,
    source = 'https://archive.ics.uci.edu/dataset/53/iris',
    targets = 'target',
    name = 'iris-dataset'
)

ValidMLDataset = mlflow.data.from_pandas(
    df = EvalDataset,
    source = 'https://archive.ics.uci.edu/dataset/53/iris',
    targets = 'target',
    name = 'iris-dataset'
)


## Setup MLFlow client
MLFLOW_SERVER_URI = os.environ.get('MLFLOW_SERVER_URI','http://localhost:5000/')
mlflow.set_tracking_uri(MLFLOW_SERVER_URI) # This commando set both tracking server and model registry (default is equal to tracking) URIs


## Load datasets to a MLFlow experiment with theirs specific context (type of dataset)
ExperimentID = mlflow.create_experiment('Iris Example')
with mlflow.start_run(run_name='load-datasets',experiment_id=ExperimentID):
    mlflow.log_input(TrainMLDataset,context='training')
    mlflow.log_input(TestMLDataset,context='testing')
    mlflow.log_input(ValidMLDataset,context='validation')