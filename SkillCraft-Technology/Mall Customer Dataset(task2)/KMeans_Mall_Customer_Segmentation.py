import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# Step 1: Create CSV if not exists
# -----------------------------
file_name = "Mall_Customers.csv"

if not os.path.exists(file_name):
    data = {
        'CustomerID': [1,2,3,4,5,6,7,8,9,10],
        'Annual Income (k$)': [15,16,17,18,45,46,47,48,60,62],
        'Spending Score (1-100)': [39,81,6,77,40,90,8,88,50,60]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    print("CSV file created successfully!")

# -----------------------------
# Step 2: Load Dataset
# -----------------------------
df = pd.read_csv(file_name)

# Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# -----------------------------
# Step 3: Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Step 4: Apply K-Means
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Print Results
print(df)

# -----------------------------
# Step 5: Visualization
# -----------------------------
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=200, c='red')

plt.title("Mall Customer Segmentation using K-Means")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.show()