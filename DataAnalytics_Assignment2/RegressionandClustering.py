# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset (Ensure dataset.csv is in the same directory)
file_path = "D:\DataAnaProject\dataset.csv"
df = pd.read_csv(file_path)

### ---------------------------------------------
###  📊  K-MEANS CLUSTERING ANALYSIS (USER SEGMENTATION)
### ---------------------------------------------

# Select relevant features for clustering (Age & App Sessions)
X_cluster = df[["Age", "App Sessions"]]

# Standardize the data for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Finding the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 6):  # Testing 1-5 clusters for efficiency
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Displaying the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), wcss, marker="o", linestyle="-")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal Clusters")
plt.show()

# Applying KMeans with 3 clusters (based on elbow method)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Displaying the K-Means Clustering map using Age & App Sessions
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Age", y="App Sessions", hue="Cluster", palette="viridis", alpha=0.7)
plt.title("User Segmentation Based on Age & App Sessions (K-Means Clustering)")
plt.xlabel("Age")
plt.ylabel("App Sessions")
plt.show()

### ---------------------------------------------
###  📈 REGRESSION ANALYSIS (PREDICTING CALORIES BURNED)
### ---------------------------------------------

# Select input features and target variable
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

# Visualizing the Regression Predictions vs Actual Values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color="red")  # Perfect prediction line
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Actual vs Predicted Calories Burned (Regression Model)")
plt.show()

# Displaying Model Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Regression Model Performance:")
print(f"   Mean Absolute Error (MAE): {mae:.2f}")
print(f"   Mean Squared Error (MSE): {mse:.2f}")
print(f"   R² Score: {r2:.2f}")