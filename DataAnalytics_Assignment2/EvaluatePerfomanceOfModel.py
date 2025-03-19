import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
#123 eeeee

file_path = "D:\DataAnaProject\dataset.csv"  # Ensure the dataset is in the same directory
df = pd.read_csv(file_path)

# Selecting input features and target variable for regression
X_reg = df[["Age", "App Sessions", "Distance Travelled (km)"]]
y_reg = df["Calories Burned"]

# Splitting data into train (80%) and test (20%) sets
train_size = int(0.8 * len(df))
X_train, X_test = X_reg[:train_size], X_reg[train_size:]
y_train, y_test = y_reg[:train_size], y_reg[train_size:]

# Applying Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictions and Model Performance Evaluation
y_pred = regressor.predict(X_test)

# Compute Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display Metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

