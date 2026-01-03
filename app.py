from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert input data to DataFrame (must match dataset columns)
    input_df = pd.DataFrame([{
        "longitude": float(data["longitude"]),
        "latitude": float(data["latitude"]),
        "housing_median_age": float(data["housing_median_age"]),
        "total_rooms": float(data["total_rooms"]),
        "total_bedrooms": float(data["total_bedrooms"]),
        "population": float(data["population"]),
        "households": float(data["households"]),
        "median_income": float(data["median_income"]),
        "ocean_proximity": data["ocean_proximity"]
    }])

    # Transform and predict
    transformed_data = pipeline.transform(input_df)
    prediction = model.predict(transformed_data)[0]

    return jsonify({
        "prediction": round(prediction, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
