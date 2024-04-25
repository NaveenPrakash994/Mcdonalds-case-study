#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram

# Load your full dataset
df = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with the actual file path


# Encode categorical variables as numerical
df_encoded = pd.get_dummies(df)

# Step 1: Extracting Segments using K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=1234)
kmeans.fit(df_encoded)
labels_kmeans = kmeans.labels_

# Step 2: Plotting Scree Plot
distortions = []
for i in range(2, 9):
    kmeans = KMeans(n_clusters=i, random_state=1234)
    kmeans.fit(df_encoded)
    distortions.append(kmeans.inertia_)

plt.plot(range(2, 9), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Scree Plot for K-Means Clustering')
plt.show()

# Step 3: Using Gaussian Mixture Models for Stability Analysis
gmm = GaussianMixture(n_components=4, n_init=10, random_state=1234)
gmm.fit(df_encoded)
labels_gmm = gmm.predict(df_encoded)

# Step 4: Plotting Global Stability
plt.boxplot(gmm.converged_, labels=range(2, 9))
plt.xlabel('Number of clusters')
plt.ylabel('Convergence')
plt.title('Global Stability of Gaussian Mixture Models')
plt.show()

# Step 5: Plotting Segment Evaluation
# Replace 'VisitFrequency', 'Like', and 'Gender' with the actual column names in your dataset
visit = df['VisitFrequency']  # Assuming 'VisitFrequency' is a numerical variable
like = df['Like']  # Assuming 'Like' is a numerical variable
female_percentage = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)  # Assuming 'Gender' is Female/Male

# Bubble sizes representing the percentage of female consumers
bubble_sizes = np.array(female_percentage) * 100

plt.figure(figsize=(8, 6))
plt.scatter(visit, like, s=bubble_sizes, c=labels_kmeans, cmap='viridis', alpha=0.7)
plt.xlabel('Mean Visit Frequency')
plt.ylabel('Mean Liking McDonald\'s')
plt.title('Segment Evaluation Plot')
plt.colorbar(label='Segment Number')
plt.grid(True)
plt.show()

# Step 6: Hierarchical Clustering on Attributes
MD_vclust = linkage(df_encoded.T, method='complete')
plt.figure(figsize=(10, 5))
dendrogram(MD_vclust, labels=labels_kmeans, orientation='top')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Segments')
plt.ylabel('Distance')
plt.show()

# Step 7: Visualizing Association between Segment Membership and McDonald's Preference
sns.heatmap(pd.crosstab(labels_kmeans, df['Like']), annot=True, cmap='coolwarm', cbar=False)
plt.xlabel('Like for McDonald\'s')
plt.ylabel('Segment number')
plt.title('Association between Segment Membership and McDonald\'s Preference')
plt.show()

# Step 8: Computing Mean Values of Descriptor Variables for Each Segment
visit_means = df.groupby(labels_kmeans)['VisitFrequency'].mean()





