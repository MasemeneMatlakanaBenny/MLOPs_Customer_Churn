import joblib

try:
    import mlflow
    from mlflow import MlflowClient

except ImportError:
    raise ImportError("modules not found")

## create variables first:
host="http://127.0.0.1:5000"

exp_name="Customer_Churn_Modeling"

exp_description="Customer Churn with LOGISTIC REGRESSION & DECISION TREE"

tags={
    "project_name":"Customer churn",
    "team":"Machine Learning and Data Science",
    "team lead":"Masemene Matlakana Benny",
    "date":"October/November 2025",
    "mlflow.note.content":exp_description
}
log_model_name="logistic_churn_model"

dt_model_name="tree_churn_model"

## load the mlflow tracking uri:
def load_host():
    return host

##function for setting mlflow within the python file workflow:
def set_mlflow_host():
    return mlflow.set_tracking_uri(uri=host)

## function for creating the mlflowclient:
def mlflow_client():
    return MlflowClient(tracking_uri=host)

## function for creating the mlflow experiment:
def set_mlflow_exp(exp_name=exp_name):
    return mlflow.set_experiment(experiment_name=exp_name)

## this the function that will load the models name:
def load_blue_model_name():
    return log_model_name


## this is the function for loading the logistic regression model:
def load_green_model_name():
    return dt_model_name

## function for testing model registry
def test_model_registry(name, version,client_var=mlflow_client()):
    from mlflow.exceptions import RestException
    """
    Check if a specific model version exists in MLflow Model Registry
    and print the result.

    Args:
        name (str): Model name in the registry.
        version (str or int): Version number of the model.
    """
    client = client_var
    try:
        client.get_model_version(name=name, version=version)
        print(f"Model '{name}' version {version} exists")
    except RestException:
        print(f" Model '{name}' version {version} not found")
        

## a function for testing model versioning or stages per model
def test_model_versioning(name, stage,client_var=mlflow_client()):

    from mlflow.exceptions import RestException
    """
    Check if a specific model version exists in MLflow Model Registry
    and print the result.

    Args:
        name (str): Model name in the registry.
        version (str or int): Version number of the model.
    """
    client = mlflow_client()
    try:
        client.get_model_version(name=name, stage=stage)
        print(f"Model '{name}' at {stage} exists")
    except RestException:
        print(f"Model '{name}' at {stage} not found")






