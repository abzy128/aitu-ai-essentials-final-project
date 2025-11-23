# Power Consumption Prediction API

FastAPI web application for predicting power consumption in Tetouan, Morocco using a trained MLP (Multilayer Perceptron) model.

## Features

- 🚀 Fast and async API built with FastAPI
- 🤖 MLP model for power consumption forecasting
- 📊 RESTful endpoints with automatic documentation
- ✅ Input validation using Pydantic
- 🔄 CORS enabled for frontend integration

## Installation

1. Install dependencies (from project root):
```bash
uv sync
```

## Running the API

### Option 1: Using Python directly

```bash
cd web
python main.py
```

### Option 2: Using Uvicorn

```bash
cd web
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Web Interface**: http://localhost:8000 (main page)
- **Interactive Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc

## API Endpoints

### 1. Web Interface

**GET** `/`

Serves the interactive web interface for making predictions.

### 2. Health Check

**GET** `/health`

Check if the API and model are loaded successfully.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "test_r2_score": 0.5253,
    "test_mae": 7113.61,
    "test_mape": 10.11
  }
}
```

### 3. Make Prediction

**POST** `/predict`

Predict power consumption based on weather conditions and datetime.

**Request Body:**
```json
{
  "temperature": 20.0,
  "humidity": 60.0,
  "wind_speed": 0.1,
  "general_diffuse_flows": 0.05,
  "diffuse_flows": 0.1,
  "datetime": "2017-01-01T12:00:00"
}
```

**Response:**
```json
{
  "predicted_power_consumption": 65432.12,
  "input_datetime": "2017-01-01T12:00:00",
  "input_features": {
    "temperature": 20.0,
    "humidity": 60.0,
    "wind_speed": 0.1,
    "general_diffuse_flows": 0.05,
    "diffuse_flows": 0.1
  },
  "model_info": {
    "model_type": "MLP (Multilayer Perceptron)",
    "test_r2_score": 0.5253,
    "test_mae": 7113.61
  }
}
```

### 4. Get Model Information

**GET** `/model-info`

Get detailed information about the model architecture and performance.

**Response:**
```json
{
  "model_architecture": {
    "type": "Multilayer Perceptron (MLP)",
    "input_size": 12,
    "hidden_layers": [256, 128, 64, 32],
    "dropout_rate": 0.3,
    "total_layers": 5
  },
  "features": [
    "Temperature",
    "Humidity",
    "WindSpeed",
    "GeneralDiffuseFlows",
    "DiffuseFlows",
    "Hour_sin",
    "Hour_cos",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "Month_sin",
    "Month_cos",
    "DayOfYear"
  ],
  "performance_metrics": {
    "MSE": 96276624.0,
    "RMSE": 9812.07,
    "MAE": 7113.61,
    "R2": 0.5253,
    "MAPE": 10.11
  },
  "target": "Total Power Consumption (sum of 3 zones)"
}
```

## Testing the API

### Using the Web Interface (Easiest)

Simply navigate to http://localhost:8000 in your browser to access the interactive web interface.

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 20.0,
    "humidity": 60.0,
    "wind_speed": 0.1,
    "general_diffuse_flows": 0.05,
    "diffuse_flows": 0.1,
    "datetime": "2017-01-01T12:00:00"
  }'
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make a prediction
data = {
    "temperature": 20.0,
    "humidity": 60.0,
    "wind_speed": 0.1,
    "general_diffuse_flows": 0.05,
    "diffuse_flows": 0.1,
    "datetime": "2017-01-01T12:00:00"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

### Using the Interactive Docs

1. Navigate to http://localhost:8000/docs
2. Click on an endpoint (e.g., POST /predict)
3. Click "Try it out"
4. Modify the request body if needed
5. Click "Execute"
6. See the response below

## Model Information

- **Architecture**: Multilayer Perceptron (MLP) with 4 hidden layers
- **Hidden Layers**: [256, 128, 64, 32] neurons
- **Input Features**: 12 features (weather data + cyclical time features)
- **Output**: Total power consumption (sum of 3 zones)
- **Performance**: R² Score of 0.53 on test set

## Input Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `temperature` | float | Weather temperature in °C | 20.0 |
| `humidity` | float | Weather humidity (0-100%) | 60.0 |
| `wind_speed` | float | Wind speed (m/s) | 0.1 |
| `general_diffuse_flows` | float | General diffuse flows | 0.05 |
| `diffuse_flows` | float | Diffuse flows | 0.1 |
| `datetime` | string | ISO format datetime | "2017-01-01T12:00:00" |

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

## Project Structure

```
web/
├── main.py           # FastAPI application
└── README.md         # This file
```

## Notes

- The model files are loaded from `../output/mlp/` directory
- The API uses CORS middleware to allow cross-origin requests
- All predictions are made using PyTorch with no gradient computation
- Input features are automatically scaled using the saved scaler

