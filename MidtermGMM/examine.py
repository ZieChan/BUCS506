
from gmm import GMMFromScratch
import numpy as np
import pandas as pd
from scipy.stats import mode

# Set the random seed for reproducibility
np.random.seed(42)

# Load the training and testing datasets
train_data = pd.read_csv('vowel_train.txt')
test_data = pd.read_csv('vowel_test.txt')

# Separate features and labels
X_train = train_data.iloc[:, 2:].values
y_train = train_data.iloc[:, 1].values
X_test = test_data.iloc[:, 2:].values
y_test = test_data.iloc[:, 1].values

# Initialize and fit the GMM from scratch
gmm_model = GMMFromScratch(k=11, iterations=100)
gmm_model.expectation_maximization(X_train)

predicted_clusters = gmm_model.predict(X_test)

cluster_to_label_mapping = {}
training_predictions = gmm_model.predict(X_train)

for cluster in np.unique(training_predictions):
    # Find the most common training label for samples in this cluster
    labels_in_cluster = y_train[training_predictions == cluster]
    if labels_in_cluster.size > 0:
        common_label_mode = mode(labels_in_cluster)
        most_common_label = common_label_mode.mode.item()  # safely extract the mode value
        cluster_to_label_mapping[cluster] = most_common_label
    else:
        # Handle the case where a cluster is empty
        cluster_to_label_mapping[cluster] = np.random.choice(y_train)

# Map the predicted clusters to the most common labels
predicted_labels = np.array([cluster_to_label_mapping.get(cluster, np.random.choice(y_train)) for cluster in predicted_clusters])

# Calculate the misclassification rate
if predicted_labels.shape[0] == y_test.shape[0]:
    misclassification_rate = np.mean(predicted_labels != y_test)
    print(f'Misclassification Rate: {misclassification_rate * 100:.2f}%')
else:
    print("Error: Predicted labels and test labels have mismatched lengths.")
