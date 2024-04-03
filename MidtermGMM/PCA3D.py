
from gmm import GMMFromScratch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

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


# Load the test data
test_data = pd.read_csv('vowel_test.txt')
X_test = test_data.drop(['row.names', 'y'], axis=1).values

# Reduce the test data to 3 principal components
pca = PCAFromScratch(n_components=3)
pca.fit(X_test)  # You might want to fit to the training data instead, and then transform the test data
X_test_pca = pca.transform(X_test)

# Classify the test data using the GMM model
# Separate features and labels
train_data = pd.read_csv('vowel_train.txt')
X_train = train_data.iloc[:, 2:].values
y_train = train_data.iloc[:, 1].values
gmm_model = GMMFromScratch(k=11, iterations=100)
gmm_model.fit(X_train)  # Ensure the GMM model is trained on the training data
gmm_predictions = gmm_model.predict(X_test)

# Map the GMM predictions to colors
class_to_color_map = {
    1: "red",
    2: "blue",
    3: "green",
    4: "yellow",
    5: "orange",
    6: "purple",
    7: "pink",
    8: "brown",
    9: "gray",
    10: "olive",
    11: "cyan",
}
colors = [class_to_color_map[label + 1] for label in gmm_predictions]

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2], c=colors, s=50)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.set_title("3D Scatter Plot of Vowel Data with GMM Classification")

plt.show()
