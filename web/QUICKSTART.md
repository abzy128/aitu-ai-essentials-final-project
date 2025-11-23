# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

From the project root directory:

```bash
uv sync
```

This will install all required packages including:
- FastAPI
- Uvicorn
- PyTorch
- NumPy, Pandas
- Joblib, Scikit-learn

### Step 2: Start the API Server

```bash
cd web
python main.py
```

You should see:
```
✓ Model loaded successfully!
  - Input size: 12
  - Hidden layers: [256, 128, 64, 32]
  - Test R² Score: 0.5253

Starting server...
API Documentation: http://localhost:8000/docs
```

### Step 3: Test the API

#### Option A: Use the Web Interface (Easiest!)

Simply open your browser and navigate to:
```
http://localhost:8000
```

You'll see a beautiful, interactive interface to make predictions!

#### Option B: Use the Interactive API Docs

Navigate to: http://localhost:8000/docs

#### Option C: Use the Test Script

In a new terminal:
```bash
cd web
python test_api.py
```

#### Option D: Use curl

```bash
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

## 📝 Example Response

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

## 🔧 Troubleshooting

### Error: "Can't get attribute 'MLPRegressor'"
**Solution**: This is already fixed! The MLPRegressor class is now defined in `main.py`.

### Error: "Cannot connect to API"
**Solution**: Make sure the server is running with `python main.py`

### Error: "Model not loaded"
**Solution**: Ensure the model files exist in `../output/mlp/` directory. Run the Jupyter notebook first to train and save the model.

### Error: Import errors
**Solution**: Run `uv sync` from the project root to install all dependencies.

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed API documentation
- Explore the interactive API docs at http://localhost:8000/docs
- Integrate the API with your frontend application
- Deploy to production (see deployment guide in README.md)

## 🎯 API Endpoints Quick Reference

- `GET /` - Web interface (main page)
- `GET /health` - Health check and status
- `POST /predict` - Make prediction (API)
- `GET /model-info` - Get model details
- `GET /docs` - Interactive API documentation

---

**Happy Predicting! ⚡**

