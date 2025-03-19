
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (Update the path if needed)
file_path = "D:\DataAnaProject\dataset.csv"  # Ensure this matches your dataset file
df = pd.read_csv(file_path)

# Set plot style
sns.set_style("whitegrid")

# 1️⃣ Correlation Heatmap - User Engagement & App Performance
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr(numeric_only=True)  # Avoids future warnings
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap - User Engagement & App Performance")
plt.show()

# 2️⃣ Distribution of App Sessions
plt.figure(figsize=(8, 5))
sns.histplot(df["App Sessions"], bins=30, kde=True, color="blue")
plt.title("Distribution of App Sessions")
plt.xlabel("Number of Sessions")
plt.ylabel("Frequency")
plt.show()

# 3️⃣ Distribution of Distance Travelled (km)
plt.figure(figsize=(8, 5))
sns.histplot(df["Distance Travelled (km)"], bins=30, kde=True, color="green")
plt.title("Distribution of Distance Travelled (km)")
plt.xlabel("Distance Travelled (km)")
plt.ylabel("Frequency")
plt.show()

# 4️⃣ Scatter Plot: App Sessions vs. Calories Burned (Color-coded by Activity Level)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="App Sessions", y="Calories Burned", hue="Activity Level", alpha=0.7, edgecolor="w")
plt.title("App Sessions vs Calories Burned")
plt.xlabel("Number of App Sessions")
plt.ylabel("Calories Burned")
plt.show()

# 5️⃣ Boxplot: App Sessions by Activity Level
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Activity Level", y="App Sessions", palette="Set2")
plt.title("App Sessions Distribution by Activity Level")
plt.xlabel("Activity Level")
plt.ylabel("Number of App Sessions")
plt.show()

# 6️⃣ Boxplot: Calories Burned by Location
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Location", y="Calories Burned", palette="Set3")
plt.title("Calories Burned Distribution by Location")
plt.xlabel("Location")
plt.ylabel("Calories Burned")
plt.show()