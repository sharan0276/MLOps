from fastapi import FastAPI, status, HTTPException, Request
from pydantic import BaseModel
from src.predict import predict_data
import numpy as np
import logging
import joblib
import time
import mlflow
from src.mlflow_config import init_mlflow

# Configure logging
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler('logs/api.log'), # log to file api.log
        logging.StreamHandler()  # Log to Console
    ]
)

logger  = logging.getLogger(__name__)

# Initialise MLflow
init_mlflow()

app = FastAPI()

class PenguinData(BaseModel):
    bill_length: float
    bill_depth: float
    flipper_length: float
    body_mass: float

class PenguinResponse(BaseModel):
    response:str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    #Log incoming requests
    logger.info(f"Incoming request : {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            f"Request completed: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time: .3f}s"
        )
        return response

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed : {request.method} {request.url.path} - Error: {str(e)} - Time: {process_time:.3f}s"
        )
        raise    # raise the error after logging


@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI Penguin Prediction API Starting Up!")
    logger.info("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Fast API Application Shutting Down!")


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=PenguinResponse)
async def predict_penguin(penguin_features: PenguinData):
    start_time = time.time()

    try:
        with mlflow.start_run(run_name="api_request"):
            mlflow.log_param("bill_length", penguin_features.bill_length)
            mlflow.log_param("bill_depth", penguin_features.bill_depth)
            mlflow.log_param("flipper_length", penguin_features.flipper_length)
            mlflow.log_param("body_mass", penguin_features.body_mass)

            art = joblib.load("model/penguin_artifact.joblib")
            classes = art["classes"]

            features = [[penguin_features.bill_length, penguin_features.bill_depth,
                    penguin_features.flipper_length, penguin_features.body_mass]]

            prediction = predict_data(np.array(features, dtype=float))

            predicted_class = classes[int(prediction[0])]

            mlflow.log_metric("prediction_label_index", int(prediction[0]))
            mlflow.log_param("prediction_label", predicted_class)

            total_time = time.time() - start_time
            mlflow.log_metric('latency_seconds', total_time)

            return PenguinResponse(response=classes[int(prediction[0])])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    
