# Basics of MLFlow

[MLFlow](https://mlflow.org/) is a AI engineer, MLOps and LLMOps platform which enables to evaluate, monitor and deploy production-quality models and applications.  

## Setup with Docker Compose
An official ready-to-use Docker Compose file can be found [here](https://github.com/mlflow/mlflow/tree/master/docker-compose). This guide will use it to interact and work with MLFlow.

```bash
cp .env.example .env
docker compose up -d
```

This Docker Compose file provides a setup to run MLFlow locally with PostgreSQL for backend store and RustFS for artifact store.

## Installation
```bash
pip install -r requirements.txt
```

## Basic Scripts
Using the [Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris), a basic example of how to use MLFlow with Scikit-Learn was developed. Every script in this folder shows a core feature of MLFlow (log artifacts, log and load models). These scripts can be executed with:
```
python 01_LoadData.py
python 02_ModelTrain.py
python 03_ServerModel.py
```

## Serve Models in a Local Inference Server (Without Docker)
After execute the previous scripts, a local server can be deployed to do inference (predictions, classifications):
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000/
mlflow models serve --no-conda -m models:/<model_name>/<model_version> -p 5001
```

The documentation of this local sever can be found in http://localhost:5001/docs. The inferences can be performed with the main endpoint (POST /invocations) as shows in 04_ExampleRequest.py.
