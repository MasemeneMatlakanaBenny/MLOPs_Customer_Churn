import mlflow.sklearn
from src.configurations_mlflow import set_mlflow_exp,set_mlflow_host,mlflow_client,load_dt_model_name


## set the mlflow host and mlflow experience within the workflow first:
set_mlflow_host()
set_mlflow_exp()

## get the model's parameters:

model_name=load_dt_model_name()
model_version="1"

## get the client:
client=mlflow_client()


## stage the model in mlflow:
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="staging"
)
