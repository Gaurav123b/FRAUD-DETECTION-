import joblib
import pandas as pd

model = joblib.load('fraud_detection_pipeline.pkl')

input_data = pd.DataFrame([{
    "type": "PAYMENT",
    "amount": 1000.0,
    "oldbalanceOrg": 10000.0,
    "newbalanceOrig": 9000.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 1000.0
}])

pred = model.predict(input_data)
print('Prediction:', pred)
