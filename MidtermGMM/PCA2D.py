
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode

class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, data):
        # Centering the data
        self.mean = np.mean(data, axis=0)
        data_centered = data - self.mean
        
        # Calculating the covariance matrix
        covariance_matrix = np.cov(data_centered, rowvar=False)
        
        # Calculating eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sorting the eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Selecting the top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, data):
        data_centered = data - self.mean
        return np.dot(data_centered, self.components)

# Load the dataset
data = pd.read_csv('vowel_train.txt')
X = data.drop(['row.names', 'y'], axis=1).values
y = data['y'].values

# Fit and transform the data using PCA
pca = PCAFromScratch(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# Plotting the first 2 principal components
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=40)
plt.title('PCA of Vowel Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Vowel')
plt.show()
