# 🏠 House Price Prediction System

## 📌 Project Overview

The House Price Prediction System is a machine learning–based web application that predicts house prices based on key features such as area, number of bedrooms, and bathrooms. The system uses a **Random Forest Regression** model trained on housing data and is deployed using a **Flask web framework** with an interactive user interface.

---

## 🚀 Features

* 📊 Predict house prices in real-time
* 🧠 Machine Learning model (Random Forest)
* 🌐 Flask-based web application
* 📈 Visualization graphs (Actual vs Predicted, Error Distribution)
* 🎯 User-friendly interface

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Backend:** Flask
* **Frontend:** HTML, CSS
* **Model Storage:** Pickle

---

## 📂 Project Structure

```
house_price_pred/
│
├── train_model.py        # Model training & evaluation
├── app.py                # Flask web app
├── model.pkl             # Trained model
├── columns.pkl           # Feature columns
│
├── static/               # Saved graphs
│   ├── graph1.png
│   └── graph2.png
│
├── templates/
│   └── index.html        # UI page
│
└── dataset.csv           # Dataset
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Train the model

```
python train_model.py
```

### 4️⃣ Run the application

```
python app.py
```

### 5️⃣ Open in browser

```
http://127.0.0.1:5000
```

---

## 📊 Model Evaluation

The model is evaluated using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **R² Score**

---

## 📈 Visualizations

* Actual vs Predicted Prices
* Error Distribution
* Feature Importance

---

## 💡 Future Improvements

* Add more features (location, amenities, etc.)
* Use advanced models like XGBoost
* Deploy on cloud (AWS / Render / Heroku)
* Add interactive charts (Plotly)

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Kushal SS**
GitHub: https://github.com/KushalSS2004
