## Basic imports
import os
from functools import partial
import pandas as pd
import mlflow


## Setup MLFlow client
MLFLOW_SERVER_URI = os.environ.get('MLFLOW_SERVER_URI','http://localhost:5000/')
mlflow.set_tracking_uri(MLFLOW_SERVER_URI)

try:
    ExperimentID = mlflow.create_experiment('Iris Example')
except:
    ExperimentID = mlflow.get_experiment_by_name('Iris Example').experiment_id


## Load registered models
### This function loads the pre-trained model into a Scikit-Learn pipeline object  
DecisionTreeModel = mlflow.sklearn.load_model('models:/IrisClassTree/1')
LogisticRegressionModel = mlflow.sklearn.load_model('models:/IrisClassLogistic/1')


## Run the models in a local inference server
# mlflow models serve -m runs:/<run_id>/model -p 5000