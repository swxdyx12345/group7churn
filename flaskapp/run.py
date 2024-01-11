from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        tenure = request.form["tenure"]
        MonthlyCharges = request.form["MonthlyCharges"]
        num_InternetServices = request.form["Number_of_InternetServices"]
        X = np.array([[float(tenure), float(MonthlyCharges), float(num_InternetServices)]])
        pred = model.predict_proba(X)[0][1]
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
