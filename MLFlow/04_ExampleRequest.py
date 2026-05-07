## Basic imports
import requests
import json
import pandas as pd


## Load data
InferenceData = pd.read_csv('./MLDatasets/valid_X.csv')


## Request body
Endpoint = 'http://localhost:5001/invocations'
Headers = {'Content-Type': 'application/json'}
DataToPredict = InferenceData.to_dict(orient='split')


## Response 
PredictedResponse = requests.post(
    Endpoint,
    headers = Headers,
    json = {'dataframe_split': DataToPredict},
)

print(json.loads(PredictedResponse.content))