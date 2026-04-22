import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("D:\\house_price_pred\\House Price Prediction Dataset.csv")

# Drop unnecessary column
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

# =========================
# 2. Split Data
# =========================
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. Train Model
# =========================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 4. Predictions
# =========================
y_pred = model.predict(X_test)

# =========================
# 5. Evaluation
# =========================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# =========================
# 6. Graphs
# =========================

# 1. Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# 2. Error Distribution
errors = y_test - y_pred
plt.figure()
plt.hist(errors, bins=20)
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

# 3. Feature Importance
importance = model.feature_importances_
feat_importance = pd.Series(importance, index=X.columns)

plt.figure()
feat_importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()

# =========================
# 7. Save Model
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("\n✅ Model trained and saved successfully!")