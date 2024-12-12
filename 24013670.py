import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the dataset with appropriate encoding
df = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')

# Preview the dataset
print("Dataset Head:")
print(df.head())

# Drop rows with missing values
df = df.dropna(subset=['CustomerID', 'Quantity', 'UnitPrice'])

# Remove rows with negative or zero Quantity or UnitPrice
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Add a new column for Total Spent
df['TotalSpent'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Print basic statistics to verify the dataset
print("\nDataset Statistics:")
print(df.describe())

# Select features for clustering
X = df[['Quantity', 'UnitPrice', 'TotalSpent']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal clusters using Elbow Method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-means clustering with optimal clusters (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Scatter plot for Clustering
plt.figure(figsize=(8, 6))
plt.scatter(df['Quantity'], df['TotalSpent'], c=df['Cluster'], cmap='viridis', alpha=0.5)
plt.title('Customer Segmentation')
plt.xlabel('Quantity')
plt.ylabel('Total Spent')
plt.colorbar(label='Cluster')
plt.show()

# Convert InvoiceDate to numerical format for regression
df['InvoiceTimestamp'] = df['InvoiceDate'].map(lambda x: x.timestamp())

# Features and target
X_reg = df[['InvoiceTimestamp']]
y = df['TotalSpent']

# Linear regression model
regressor = LinearRegression()
regressor.fit(X_reg, y)

# Add predictions to the dataset
df['FittedLine'] = regressor.predict(X_reg)

# Plot actual vs fitted values
plt.figure(figsize=(10, 6))
plt.scatter(df['InvoiceTimestamp'], df['TotalSpent'], color='blue', alpha=0.5, label='Actual')
plt.plot(df['InvoiceTimestamp'], df['FittedLine'], color='red', label='Fitted Line')
plt.title('Linear Fitting: Total Spent Over Time')
plt.xlabel('Invoice Timestamp')
plt.ylabel('Total Spent')
plt.legend()
plt.show()

# Correlation heatmap
correlation_matrix = df[['Quantity', 'UnitPrice', 'TotalSpent']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
