import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = "D:\DataAnaProject\dataset.csv"  # Ensure the dataset is in the same directory
data = pd.read_csv(file_path)

# Set figure size
plt.figure(figsize=(15, 10))

# Histogram for Age distribution
plt.subplot(2, 2, 1)
plt.hist(data['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Histogram for App Sessions
plt.subplot(2, 2, 2)
plt.hist(data['App Sessions'], bins=20, color='salmon', edgecolor='black')
plt.title('App Sessions Distribution')
plt.xlabel('App Sessions')
plt.ylabel('Frequency')

# Histogram for Distance Travelled
plt.subplot(2, 2, 3)
plt.hist(data['Distance Travelled (km)'], bins=20, color='limegreen', edgecolor='black')
plt.title('Distance Travelled Distribution')
plt.xlabel('Distance Travelled (km)')
plt.ylabel('Frequency')

# Histogram for Calories Burned
plt.subplot(2, 2, 4)
plt.hist(data['Calories Burned'], bins=20, color='violet', edgecolor='black')
plt.title('Calories Burned Distribution')
plt.xlabel('Calories Burned')
plt.ylabel('Frequency')

# Adjust layout and display plots
plt.tight_layout()
plt.show()