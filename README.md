# House Price Prediction System

## Project Overview

The House Price Prediction System is a machine learning-based web application that predicts house prices using key property features such as area, number of bedrooms, and number of bathrooms. The application uses a **Random Forest Regression** model trained on housing data and is deployed using the **Flask** web framework with a simple and interactive web interface.

---

## Features

* Predict house prices in real time
* Random Forest Regression model for accurate predictions
* Flask-based web application
* Data visualization for model performance analysis
* User-friendly interface

---

## Technology Stack

| Category             | Technologies  |
| -------------------- | ------------- |
| Programming Language | Python        |
| Machine Learning     | Scikit-learn  |
| Data Processing      | Pandas, NumPy |
| Data Visualization   | Matplotlib    |
| Backend              | Flask         |
| Frontend             | HTML, CSS     |
| Model Serialization  | Pickle        |

---

## Project Structure

```text
house_price_pred/
│
├── train_model.py        # Model training and evaluation
├── app.py                # Flask application
├── model.pkl             # Trained machine learning model
├── columns.pkl           # Feature columns
│
├── static/
│   ├── graph1.png
│   └── graph2.png
│
├── templates/
│   └── index.html        # Web interface
│
└── dataset.csv           # Housing dataset
```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

### 4. Run the Application

```bash
python app.py
```

### 5. Access the Application

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

---

## Model Evaluation

The model performance is evaluated using the following metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## Visualizations

The application generates the following visualizations:

* Actual vs. Predicted House Prices
* Error Distribution
* Feature Importance Analysis

---

## Future Enhancements

* Incorporate additional features such as location, age of property, and nearby amenities
* Experiment with advanced algorithms such as XGBoost and LightGBM
* Deploy the application on cloud platforms such as AWS, Render, or Azure
* Add interactive visualizations using Plotly
* Improve the user interface with responsive design

---

## Contributing

Contributions are welcome. If you would like to improve this project, feel free to fork the repository, create a feature branch, and submit a pull request.

---

## License

This project is licensed under the MIT License.

---

## Author

**Kushal Shiddibhavi**

GitHub: https://github.com/KushalSS2004
