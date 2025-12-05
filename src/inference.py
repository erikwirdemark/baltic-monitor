import joblib
import pandas as pd
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn

# --- 1. Define Data Schemas using Pydantic ---
# Pydantic is used for request/response validation and automatic JSON handling.

# Define the structure for a single grid point input (from the frontend)
class GridPointInput(BaseModel):
    # Unique identifier for the grid cell
    grid_index: int
    # Features used by your ML model
    phosphorus: float
    nitrogen: float
    salinity: float
    temperature: float

# Define the expected structure for the entire request payload (a list of grid points)
class InferenceRequest(BaseModel):
    data: List[GridPointInput]

# Define the structure for a single grid point output (to the frontend)
class GridPointOutput(BaseModel):
    grid_index: int
    cyanobacteria_density: float # The predicted output (e.g., cells/mL or a risk score)

# Define the expected structure for the entire response payload (a list of results)
class InferenceResponse(BaseModel):
    predictions: List[GridPointOutput]

# --- 2. Initialize FastAPI App and Load Model ---

app = FastAPI(title="HABs Prediction Inference Service")

# Use a global variable to store the loaded model
model = None

# Define the path to the model artifact.
# In a Vertex AI deployment, this might be an environment variable pointing to GCS.
MODEL_PATH = os.getenv("MODEL_PATH", "habs_predictor.joblib")

@app.on_event("startup")
def load_model():
    """Load the ML model when the server starts up."""
    global model
    try:
        # Load the saved model file
        model = joblib.load(MODEL_PATH)
        print(f"INFO: Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        # Crucial for deployment: If the model fails to load, the server must fail.
        print(f"ERROR: Failed to load model from {MODEL_PATH}. Reason: {e}")
        raise HTTPException(status_code=500, detail="Model loading error on startup.")

@app.get("/health")
def health_check():
    """Endpoint for Kubernetes/Vertex AI health checks."""
    if model is not None:
        return {"status": "ok", "model_loaded": True}
    return {"status": "error", "model_loaded": False}

# --- 3. The Core Inference Endpoint ---

@app.post("/predict", response_model=InferenceResponse)
def predict_habs(request_data: InferenceRequest):
    """
    Accepts a list of grid point data, runs inference, and returns predicted densities.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

    # 1. Prepare Data for Model
    # Convert the list of Pydantic objects into a list of lists/array for the model
    input_data_list = []
    grid_indices = []

    for item in request_data.data:
        # Order of features MUST match the order used during model training!
        features = [
            item.phosphorus,
            item.nitrogen,
            item.salinity,
            item.temperature
        ]
        input_data_list.append(features)
        grid_indices.append(item.grid_index)

    # Convert to NumPy array for efficient prediction
    X = np.array(input_data_list)
    
    # 2. Run Inference
    # NOTE: This is where you would integrate the CyFi processing *if* your model
    # uses satellite band data directly. For this example, we assume the model
    # takes processed environmental variables.
    
    # Run the prediction
    # Assumes the model returns a 1D array of predicted density values
    predictions = model.predict(X)

    # 3. Format Output
    results = []
    for index, density in zip(grid_indices, predictions):
        results.append(GridPointOutput(
            grid_index=index,
            cyanobacteria_density=float(density) # Ensure prediction is converted to a JSON-compatible float
        ))

    return InferenceResponse(predictions=results)

# --- 4. Local Execution (Optional, for testing) ---

# To run locally, save your trained model as 'habs_predictor.joblib' and run:
# uvicorn api_server:app --reload --host 0.0.0.0 --port 8080
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)