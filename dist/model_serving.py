import mlflow.sklearn
from src.configurations_mlflow import set_mlflow_exp,set_mlflow_host,load_blue_model_name,load_green_model_name


## set the mlflow host and experiment within the workflow:
set_mlflow_host()
set_mlflow_exp()

## get the model names:
blue_model_name=load_blue_model_name()
green_model_name=load_green_model_name()


stage="production"


blue_model_uri=f"models:/{blue_model_name}/{stage}"
green_model_uri=f"models:/{green_model_name}/{stage}"

blue_model=mlflow.pyfunc.load(model_uri=blue_model_uri)
green_model=mlflow.pyfunc.load(model_uri=green_model_uri)
