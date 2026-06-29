from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, "best_model.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])

    gender = request.form["gender"]
    gender = 1 if gender == "Male" else 0

    overtime = request.form["overtime"]
    overtime = 1 if overtime == "Yes" else 0

    income = float(request.form["income"])
    data = np.array([[age, gender, overtime, income]])
    data = scaler.transform(data)
    result = model.predict(data)[0]

    if result == 1:
        msg = "이직 확률이 높습니다."
    else:
        msg = "잔류할 가능성이 높습니다."

    return render_template("index.html", result=msg)


# 서버 실행
if __name__ == "__main__":
    app.run(debug=True)