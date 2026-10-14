
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained Linear SVM model and scaler
artifact = joblib.load("banknote_svm.pkl")

model = artifact["model"]
scaler = artifact["scaler"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get feature values from HTML form
        variance = float(request.form["variance"])
        skewness = float(request.form["skewness"])
        curtosis = float(request.form["curtosis"])
        entropy = float(request.form["entropy"])

        # Prepare input for prediction
        features = np.array([[
            variance,
            skewness,
            curtosis,
            entropy
        ]])

        # Apply the same scaling used during training
        features_scaled = scaler.transform(features)

        # Predict class
        result = int(model.predict(features_scaled)[0])

        # UCI dataset: class 0 = genuine, class 1 = counterfeit
        if result == 0:
            prediction = "Genuine Banknote"
        else:
            prediction = "Counterfeit Banknote"

        return render_template(
            "index.html",
            prediction=prediction
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            prediction="Please enter valid numerical values."
        )


if __name__ == "__main__":
    app.run(debug=True)
