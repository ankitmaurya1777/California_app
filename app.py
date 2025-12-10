from flask import Flask, render_template, request
import os
import joblib
import pandas as pd

app = Flask(__name__)

# -----------------------------
#  MODEL & PIPELINE LOAD KARO
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "pipeline.pkl")

model = joblib.load(MODEL_PATH)       # RandomForestRegressor
pipeline = joblib.load(PIPELINE_PATH) # preprocessing pipeline (imputer + scaler + onehot)


# -----------------------------
#  ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ---- form se values lo ----
        longitude = float(request.form["longitude"])
        latitude = float(request.form["latitude"])
        housing_median_age = float(request.form["housing_median_age"])
        total_rooms = float(request.form["total_rooms"])
        total_bedrooms = float(request.form["total_bedrooms"])
        population = float(request.form["population"])
        households = float(request.form["households"])
        median_income = float(request.form["median_income"])
        ocean_proximity = request.form["ocean_proximity"]  # string

        # ---- DataFrame banao exactly training jaisa ----
        input_data = pd.DataFrame({
            "longitude": [longitude],
            "latitude": [latitude],
            "housing_median_age": [housing_median_age],
            "total_rooms": [total_rooms],
            "total_bedrooms": [total_bedrooms],
            "population": [population],
            "households": [households],
            "median_income": [median_income],
            "ocean_proximity": [ocean_proximity],
        })

        # ---- pehle pipeline se transform, phir model se predict ----
        prepared = pipeline.transform(input_data)
        prediction = model.predict(prepared)[0]

        # y ko tune scale nahi kiya hai, isliye direct value hi house price hai
        predicted_price = float(prediction)

        msg = f"Predicted median house value: ${predicted_price:,.2f}"

        return render_template("index.html", prediction_text=msg)

    except Exception as e:
        # agar kuch galat input ho gaya to error dikha denge
        return render_template("index.html", prediction_text=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
