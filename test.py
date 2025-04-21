import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Load datasets (replace with your actual file paths/names)
df_academic = pd.read_csv("c:\\Users\\arhod\\Desktop\\Diploma-vscode\\career_change_prediction_dataset.csv")
df_startups = pd.read_csv("c:\\Users\\arhod\\Desktop\\Diploma-vscode\\INC 5000 Companies 2019.csv")          # Dataset for startups


# Inspect the dataframes to determine a merge strategy
print(df_academic.head())
print(df_startups.head())


# Merge the datasets
# If there is a common key (e.g., 'id'):
# merged_df = pd.merge(df_academic, df_startups, on="id", how="inner")

# If the datasets are to be concatenated side-by-side (ensure they have the same number of rows or align correctly):
merged_df = pd.concat([df_academic, df_startups], axis=1)

# Save merged dataframe if needed
merged_df.to_csv("merged_dataset.csv", index=False)

# Preprocessing for clustering:
# Select numeric columns for clustering (or convert categorical to numeric)
numeric_cols = merged_df.select_dtypes(include=["int64", "float64"]).columns
data_for_clustering = merged_df[numeric_cols].dropna()  # Optionally handle missing values differently

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_clustering)

# Determine the number of clusters (e.g., using the elbow method, here we choose 3 as an example)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Add cluster results to the dataframe
merged_df.loc[data_for_clustering.index, 'cluster'] = clusters

# Optional: Visualize the clusters using the first two numeric features
plt.figure(figsize=(8, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap="viridis", alpha=0.6)
plt.title("K-means Clustering")
plt.xlabel("Scaled Feature 1")
plt.ylabel("Scaled Feature 2")
plt.show()

# Print the centroids in the scaled feature space
print("Cluster Centers:\n", kmeans.cluster_centers_)