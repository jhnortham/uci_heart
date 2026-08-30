import json
import joblib
import pandas as pd

MODEL_PATH = "/var/task/heart_disease_pipeline.joblib"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "thalach",
    "exang",
    "oldpeak",
    "slope"
]

def lambda_handler(event, context):
    body = event.get("body", event)

    if isinstance(body, str):
        body = json.loads(body)

    input_data = pd.DataFrame([body], columns=FEATURES)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "prediction": int(prediction),
            "probability": float(probability)
        })
    }
