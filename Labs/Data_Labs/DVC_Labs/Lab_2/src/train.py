from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder
from data import load_data, split_data
import logging
import mlflow
import mlflow.sklearn
from mlflow_config import init_mlflow


#Configuring logging for training
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def fit_model(X_train, y_train):
    """
    Train a RandomForest model and log training details using MLflow.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    try:
        logger.info("Training Random Forest Classifier...")
        logger.info(f"Training data shape: {X_train.shape}")

        rf_classifier = RandomForestClassifier(max_depth=3, random_state=12)
        rf_classifier.fit(X_train, y_train)

        logger.info("Model training completed succesfully!")
        logger.info("Saving model to 'model/penguin_model.pkl'")

        joblib.dump(rf_classifier, "../model/penguin_model.pkl")
        logger.info("Saved Model Successfully")

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise


if __name__ == "__main__":
    logger.info("Starting Penguin Model Training Pipeline")

    # Initialize ML flow
    init_mlflow()

    try:
        with mlflow.start_run(run_name =  "training_run"):
            X, y = load_data()

            logger.info("Encoding target labels.")
            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            logger.info(f"Classes: {list(le.classes_)}")

            # Logging class name parameters
            mlflow.log_param("classes", list(le.classes_))


            X_train, X_test, y_train, y_test = split_data(X, y_enc)


            # Log dataset sizes
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])


            # Train model and log training params
            model = fit_model(X_train, y_train)

            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("max_depth", 3)

            #log model artifact to MLflow
            mlflow.sklearn.log_model(model, "rf_penguin_model")

            # Save artifact containing class names
            artifact = {"classes": list(le.classes_)}
            joblib.dump(artifact, "../model/penguin_artifact.joblib")
            mlflow.log_artifact("../model/penguin_artifact.joblib")


            #fit_model(X_train, y_train)

            logger.info("Saving classes in artifcat")
            artifact = {
                "classes": list(le.classes_)
            }
            joblib.dump(artifact, "../model/penguin_artifact.joblib")
            logger.info('Artifact saved successfully!')

            logger.info("Training Pipeline Completed Successfully!")

    except Exception as e:
        logger.exception(f"Training pipeline Failure : {str(e)}")
        raise