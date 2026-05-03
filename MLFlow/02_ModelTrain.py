## Basic imports
import os
from functools import partial
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , precision_score , f1_score
import mlflow


## Setup MLFlow client
MLFLOW_SERVER_URI = os.environ.get('MLFLOW_SERVER_URI','http://localhost:5000/')
mlflow.set_tracking_uri(MLFLOW_SERVER_URI)

try:
    ExperimentID = mlflow.create_experiment('Iris Example')
except:
    ExperimentID = mlflow.get_experiment_by_name('Iris Example').experiment_id


## Download preprocessed datasets as artifacts
ResultSearch = mlflow.search_runs(
    experiment_ids = [ExperimentID],
    filter_string = "tags.mlflow.runName = 'load-datasets'"
)
RunID_LoadDatasets = ResultSearch['run_id'][0]

mlflow.artifacts.download_artifacts(
    run_id = RunID_LoadDatasets,
    dst_path = './MLDatasets',
)


## Load preprocessed datasets
Train_X_DatasetPath = './MLDatasets/train_X.csv'
Test_X_DatasetPath = './MLDatasets/test_X.csv'
Train_y_DatasetPath = './MLDatasets/train_y.csv'
Test_y_DatasetPath = './MLDatasets/test_y.csv'

TrainDataset_X = pd.read_csv(Train_X_DatasetPath)
TrainDataset_y = pd.read_csv(Train_y_DatasetPath)
TestDataset_X = pd.read_csv(Test_X_DatasetPath)
TestDataset_y = pd.read_csv(Test_y_DatasetPath)


## Init MLFlow datasets to log
TrainDataset = pd.concat([TrainDataset_X,TrainDataset_y],axis=1)
TestDataset = pd.concat([TestDataset_X,TestDataset_y],axis=1)

TrainDataset = mlflow.data.from_pandas(
    TrainDataset,
    source = 'UCI Iris Dataset',
    targets = 'target',
    name = 'iris-train-dataset',
)

TestDataset = mlflow.data.from_pandas(
    TestDataset,
    source = 'UCI Iris Dataset',
    targets = 'target',
    name = 'iris-test-dataset',
)


## Init ML Models

### Init Decision Tree ML Model
ModelParams_Tree = dict(
    criterion = 'gini',
    max_depth = 3,
    random_state = 8013,
)
Classifier_Tree = DecisionTreeClassifier(**ModelParams_Tree)

### Init ML Model
ModelParams_Logistic = dict(
    penalty = 'elasticnet',
    solver = 'saga',
    l1_ratio = 0.5,
    random_state = 8013,
)
Classifier_Logistic = LogisticRegression(**ModelParams_Logistic,)


## Logs experiment results
Metrics = {
    'accuracy': accuracy_score,
    'precision': partial(precision_score,average='macro'),
    'f1': partial(f1_score,average='macro'),
}

with mlflow.start_run(run_name='train-models',experiment_id=ExperimentID) as MainRun:
    ### Log datasets for this experiment
    mlflow.log_input(
        TrainDataset,
        context = 'training',tags={'run_artifact_source': RunID_LoadDatasets, 'features_artifact': 'train_X.csv', 'target_artifact': 'train_y.csv'}) # Log train dataset reference
    mlflow.log_input(
        TestDataset,
        context = 'testing',tags={'run_artifact_source': RunID_LoadDatasets, 'features_artifact': 'test_X.csv', 'target_artifact': 'test_y.csv'})

    ### Decision Tree
    with mlflow.start_run(run_name='decision-tree-model',experiment_id=ExperimentID,nested=True):
        mlflow.log_param('model','Decision Tree')
        Classifier_Tree.fit(TrainDataset_X,TrainDataset_y)
        tree_logged_model = mlflow.sklearn.log_model(
            Classifier_Tree,
            name = 'iris-classifier-tree',
            params = ModelParams_Tree
        ) # Log model as artifact

        tree_predictions_y = Classifier_Tree.predict(TestDataset_X)
        for metric_name , metric in Metrics.items():
            score = metric(TestDataset_y,tree_predictions_y)
            mlflow.log_metric(
                metric_name,score,
                model_id = tree_logged_model.model_id,
                dataset = TestDataset,
            )

    ### Logistic Regression
    with mlflow.start_run(run_name='logistic-model',experiment_id=ExperimentID,nested=True):
        mlflow.log_param('model','Logistic Regression')
        Classifier_Logistic.fit(TrainDataset_X,TrainDataset_y)
        logistic_logged_model = mlflow.sklearn.log_model(
            Classifier_Logistic,
            name = 'iris-classifier-logistic',
            params = ModelParams_Logistic,
        ) # Log model as artifact

        logistic_predictions_y = Classifier_Logistic.predict(TestDataset_X)
        for metric_name , metric in Metrics.items():
            score = metric(TestDataset_y,logistic_predictions_y)
            mlflow.log_metric(
                metric_name,score,
                model_id = logistic_logged_model.model_id,
                dataset = TestDataset,
            )