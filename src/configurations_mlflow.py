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
def load_dt_model(name=dt_model_name,model_path:str="models/dt_model.pkl"):
    model_name=name
    model_file=joblib.load(model_path)

    return model_name,model_file


## this is the function for loading the logistic regression model:
def load_dt_model(name=log_model_name,model_path:str="models/log_model.pkl"):
    model_name=name
    model_file=joblib.load(model_path)

    return model_name,model_file







