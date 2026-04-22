from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Dummy data for visualization (you can replace with real test data later)
y_test_sample = np.random.randint(100000, 500000, 50)
y_pred_sample = y_test_sample + np.random.randint(-50000, 50000, 50)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()

    input_data = [0] * len(columns)

    for key in data:
        try:
            if key in columns:
                input_data[list(columns).index(key)] = float(data[key])
            else:
                col_name = f"{key}_{data[key]}"
                if col_name in columns:
                    input_data[list(columns).index(col_name)] = 1
        except:
            pass

    prediction = model.predict([input_data])[0]

    # =========================
    # Generate Graphs
    # =========================

    if not os.path.exists("static"):
        os.makedirs("static")

    # 1. Actual vs Predicted
    plt.figure()
    plt.scatter(y_test_sample, y_pred_sample)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted")
    plt.savefig("static/graph1.png")
    plt.close()

    # 2. Error Distribution
    errors = y_test_sample - y_pred_sample
    plt.figure()
    plt.hist(errors, bins=20)
    plt.title("Error Distribution")
    plt.savefig("static/graph2.png")
    plt.close()

    return render_template(
        "index.html",
        prediction=f"Estimated Price: ₹ {round(prediction, 2)}",
        graph1="static/graph1.png",
        graph2="static/graph2.png"
    )

if __name__ == "__main__":
    app.run(debug=True)