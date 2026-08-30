"""
Test API

"""

# Import packages
import requests

# Set URL

url = "https://8edatdilna.execute-api.us-east-2.amazonaws.com/prod/predict"

# Set payload
payload = {
    "age": 54,
    "sex": 1,
    "cp": 4,
    "trestbps": 140,
    "chol": 239,
    "thalach": 160,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 2
}

# Set response
response = requests.post(url, json=payload)

assert response.status_code == 200

# Get and print results
result = response.json()

assert result["prediction"] in [0,1]
assert 0 <= result["probability"] <= 1.0

print(result)

