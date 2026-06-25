from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model
model = joblib.load("models/fraud_model.pkl")

@app.route("/")
def home():
    return "Credit Card Fraud Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)

    result = "Fraud" if prediction[0] == 1 else "Legitimate"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)