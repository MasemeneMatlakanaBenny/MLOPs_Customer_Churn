from configurations_mlflow import set_mlflow_exp,set_mlflow_host,load_log_model_name
from configurations_mlflow import test_model_versioning,mlflow_client


## set the mlflow experiment and mlflow host within the workflow:
set_mlflow_exp()
set_mlflow_host()

model_name=load_log_model_name()
model_stage="staging"

## client:
client=mlflow_client()

test_stage_phase=test_model_versioning(name=model_name,stage=model_stage,client_var=client)

print(test_stage_phase)
