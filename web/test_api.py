"""
Test script for the Power Consumption Prediction API
Run this after starting the API server
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_prediction():
    """Test the prediction endpoint"""
    print("\n" + "=" * 60)
    print("Testing Prediction Endpoint")
    print("=" * 60)
    
    # Test data
    test_cases = [
        {
            "name": "Summer Day - High Temperature",
            "data": {
                "temperature": 30.0,
                "humidity": 65.0,
                "wind_speed": 0.15,
                "general_diffuse_flows": 0.08,
                "diffuse_flows": 0.12,
                "datetime": "2017-07-15T14:00:00"
            }
        },
        {
            "name": "Winter Night - Low Temperature",
            "data": {
                "temperature": 8.0,
                "humidity": 80.0,
                "wind_speed": 0.05,
                "general_diffuse_flows": 0.03,
                "diffuse_flows": 0.08,
                "datetime": "2017-01-20T02:00:00"
            }
        },
        {
            "name": "Spring Morning - Moderate Conditions",
            "data": {
                "temperature": 18.0,
                "humidity": 70.0,
                "wind_speed": 0.1,
                "general_diffuse_flows": 0.06,
                "diffuse_flows": 0.1,
                "datetime": "2017-04-10T08:00:00"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 60)
        print(f"Input: {json.dumps(test_case['data'], indent=2)}")
        
        response = requests.post(f"{BASE_URL}/predict", json=test_case['data'])
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Prediction successful!")
            print(f"  Predicted Power Consumption: {result['predicted_power_consumption']:.2f}")
            print(f"  Model R² Score: {result['model_info']['test_r2_score']:.4f}")
            print(f"  Model MAE: {result['model_info']['test_mae']:.2f}")
        else:
            print(f"✗ Error: {response.json()}")


def test_model_info():
    """Test the model info endpoint"""
    print("\n" + "=" * 60)
    print("Testing Model Info Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        info = response.json()
        print("\nModel Architecture:")
        print(f"  Type: {info['model_architecture']['type']}")
        print(f"  Hidden Layers: {info['model_architecture']['hidden_layers']}")
        print(f"  Total Layers: {info['model_architecture']['total_layers']}")
        
        print("\nPerformance Metrics:")
        for metric, value in info['performance_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nTarget: {info['target']}")
    else:
        print(f"Error: {response.json()}")


def test_error_handling():
    """Test error handling with invalid input"""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    # Test with invalid datetime
    print("\nTest: Invalid datetime format")
    invalid_data = {
        "temperature": 20.0,
        "humidity": 60.0,
        "wind_speed": 0.1,
        "general_diffuse_flows": 0.05,
        "diffuse_flows": 0.1,
        "datetime": "not-a-date"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"✓ Error handled correctly: {response.json()['detail']}")
    
    # Test with missing fields
    print("\nTest: Missing required field")
    incomplete_data = {
        "temperature": 20.0,
        "humidity": 60.0,
        # Missing other required fields
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=incomplete_data)
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"✓ Error handled correctly: Validation error")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Power Consumption Prediction API - Test Suite")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test health check
        if not test_health_check():
            print("\n✗ Health check failed. Is the server running?")
            print("Start the server with: python main.py")
            return
        
        # Run other tests
        test_prediction()
        test_model_info()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed!")
        print("=" * 60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to API server.")
        print("Please start the server first with: python main.py")
    except Exception as e:
        print(f"\n✗ Error running tests: {str(e)}")


if __name__ == "__main__":
    main()

