# mlflow_config.py
import mlflow

def init_mlflow():
    """
    Set up MLflow experiment and tracking location.
    Modify the URI if you want to use a remote MLflow server.
    """
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("penguin_dvc_classifier")