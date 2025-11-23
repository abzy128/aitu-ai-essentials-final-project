import os
from datetime import datetime
from typing import Optional

import joblib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# Define the MLP model architecture (must match the notebook)
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.3):
        super(MLPRegressor, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


# Initialize FastAPI app
app = FastAPI(
    title="Power Consumption Prediction API",
    description="API for predicting power consumption in Tetouan, Morocco using MLP model",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables for model and scalers
model = None
scaler_X = None
scaler_y = None
config = None


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    temperature: float = Field(..., description="Weather temperature in °C", example=20.0)
    humidity: float = Field(..., description="Weather humidity in %", ge=0, le=100, example=60.0)
    wind_speed: float = Field(..., description="Wind speed", ge=0, example=0.1)
    general_diffuse_flows: float = Field(..., description="General diffuse flows", ge=0, example=0.05)
    diffuse_flows: float = Field(..., description="Diffuse flows", ge=0, example=0.1)
    datetime: str = Field(
        ...,
        description="Date and time in ISO format (e.g., 2017-01-01T12:00:00)",
        example="2017-01-01T12:00:00"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 20.0,
                "humidity": 60.0,
                "wind_speed": 0.1,
                "general_diffuse_flows": 0.05,
                "diffuse_flows": 0.1,
                "datetime": "2017-01-01T12:00:00"
            }
        }


class PredictionResponse(BaseModel):
    predicted_power_consumption: float = Field(..., description="Predicted total power consumption")
    input_datetime: str = Field(..., description="Input datetime")
    input_features: dict = Field(..., description="Input features used for prediction")
    model_info: dict = Field(..., description="Model information")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[dict] = None


def load_mlp_model(model_dir="../output/mlp"):
    """Load the trained MLP model and preprocessing objects."""
    global model, scaler_X, scaler_y, config
    
    try:
        # Load configuration
        config_path = os.path.join(model_dir, "model_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Load the complete model
        model_path = os.path.join(model_dir, "mlp_model_complete.pth")
        model = torch.load(model_path, weights_only=False)
        model.eval()
        
        # Load scalers
        scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.pkl"))
        scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))
        
        print("✓ Model loaded successfully!")
        print(f"  - Input size: {config['input_size']}")
        print(f"  - Hidden layers: {config['hidden_sizes']}")
        print(f"  - Test R² Score: {config['test_metrics']['R2']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False


def predict_power_consumption(
    temperature: float,
    humidity: float,
    wind_speed: float,
    general_diffuse_flows: float,
    diffuse_flows: float,
    datetime_obj: datetime,
) -> float:
    """
    Make a prediction for power consumption.
    
    Parameters:
    - temperature: Weather temperature
    - humidity: Weather humidity
    - wind_speed: Wind speed
    - general_diffuse_flows: General diffuse flows
    - diffuse_flows: Diffuse flows
    - datetime_obj: datetime object for the prediction time
    
    Returns:
    - Predicted total power consumption
    """
    # Extract time features
    hour = datetime_obj.hour
    day_of_week = datetime_obj.weekday()
    month = datetime_obj.month
    day_of_year = datetime_obj.timetuple().tm_yday
    
    # Create cyclical features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Create feature vector (must match the training feature order)
    features = np.array(
        [
            [
                temperature,
                humidity,
                wind_speed,
                general_diffuse_flows,
                diffuse_flows,
                hour_sin,
                hour_cos,
                day_of_week_sin,
                day_of_week_cos,
                month_sin,
                month_cos,
                day_of_year,
            ]
        ]
    )
    
    # Scale features
    features_scaled = scaler_X.transform(features)
    
    # Make prediction
    with torch.no_grad():
        # Get the device the model is on
        device = next(model.parameters()).device
        
        # Create tensor and move to the same device as the model
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        
        # Get prediction and move back to CPU for numpy conversion
        prediction_scaled = model(features_tensor).cpu().numpy()
    
    # Inverse transform to get original scale
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
    
    return float(prediction)


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("\n" + "=" * 60)
    print("Starting Power Consumption Prediction API")
    print("=" * 60)
    load_mlp_model()
    print("=" * 60 + "\n")


@app.get("/")
async def root():
    """Serve the web interface"""
    return FileResponse("index.html", media_type="text/html")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "model_info": {
            "input_size": config["input_size"],
            "hidden_sizes": config["hidden_sizes"],
            "test_metrics": config["test_metrics"],
        } if config else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict power consumption based on weather conditions and datetime.
    
    Returns the predicted total power consumption for all three zones.
    """
    if model is None or scaler_X is None or scaler_y is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Parse datetime
        datetime_obj = pd.to_datetime(request.datetime)
        
        # Make prediction
        prediction = predict_power_consumption(
            temperature=request.temperature,
            humidity=request.humidity,
            wind_speed=request.wind_speed,
            general_diffuse_flows=request.general_diffuse_flows,
            diffuse_flows=request.diffuse_flows,
            datetime_obj=datetime_obj,
        )
        
        return PredictionResponse(
            predicted_power_consumption=prediction,
            input_datetime=request.datetime,
            input_features={
                "temperature": request.temperature,
                "humidity": request.humidity,
                "wind_speed": request.wind_speed,
                "general_diffuse_flows": request.general_diffuse_flows,
                "diffuse_flows": request.diffuse_flows,
            },
            model_info={
                "model_type": "MLP (Multilayer Perceptron)",
                "test_r2_score": config["test_metrics"]["R2"],
                "test_mae": config["test_metrics"]["MAE"],
            }
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid datetime format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/model-info")
async def get_model_info():
    """Get detailed model information"""
    if config is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_architecture": {
            "type": "Multilayer Perceptron (MLP)",
            "input_size": config["input_size"],
            "hidden_layers": config["hidden_sizes"],
            "dropout_rate": config["dropout_rate"],
            "total_layers": len(config["hidden_sizes"]) + 1,
        },
        "features": config["feature_columns"],
        "performance_metrics": config["test_metrics"],
        "target": "Total Power Consumption (sum of 3 zones)",
    }


# For testing locally
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("Running Power Consumption Prediction API")
    print("=" * 60)
    print("Loading model...")
    
    # Load model before starting server
    if load_mlp_model():
        print("\nStarting server...")
        print("API Documentation: http://localhost:8000/docs")
        print("=" * 60 + "\n")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Failed to load model. Exiting.")
