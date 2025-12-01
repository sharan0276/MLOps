import joblib
import logging
import mlflow
from src.mlflow_config import init_mlflow

logger = logging.getLogger(__name__)

# Initialise MLflows
init_mlflow()


def predict_data(X):
    """
    Predict the class labels for the input dataa and log inference details to  MLflow.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    try:
        # using nested = True because
        with mlflow.start_run(run_name="inference_run", nested = True):

            # Log inout shape
            mlflow.log_param("input_shape", X.shape)

            logger.debug(f"Loading model for prediction.....")
            model = joblib.load("model/penguin_model.pkl")
            logger.debug(f"Input Shape : {X.shape}")
            y_pred = model.predict(X)

            # Log output
            mlflow.log_metric("num_predictions", len(y_pred))
            logger.debug(f"prediction result : {y_pred}")
            return y_pred

    except Exception as e:
        logger.error(f"Error in perdict_data function: {str(e)}")
        raise
