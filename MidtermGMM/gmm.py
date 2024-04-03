
import numpy as np
from numpy import array, argmax
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy.random import multivariate_normal as mvn_random
from scipy.stats import multivariate_normal
from numpy.random import normal, uniform
from scipy.stats import mode


class Component:
    def __init__(self, mixture_prop, mean, variance):
        self.mixture_prop = mixture_prop
        self.mean = mean
        self.variance = variance

class GMMFromScratch:
    def __init__(self, k, iterations):
        self.k = k
        self.iterations = iterations
        self.gmm_params = []

    def fit(self, dataset):
        self.expectation_maximization(dataset)

    def expectation_maximization(self, dataset):
        self.gmm_init(dataset)
        probs = None
        for _ in range(self.iterations):
            probs = self.compute_probs(dataset)
            self.compute_gmm(dataset, probs)
        return probs

    def gmm_init(self, dataset):
        kmeans = KMeans(self.k, init='k-means++').fit(np.array(dataset))
        for j in range(self.k):
            idx = np.where(kmeans.labels_ == j)[0]
            p_cj = len(idx) / len(dataset)
            mean_j = np.mean(np.array(dataset)[idx], axis=0)
            var_j = np.cov(np.array(dataset)[idx].T) + np.eye(dataset.shape[1]) * 1e-6
            self.gmm_params.append(Component(p_cj, mean_j, var_j))

    def compute_probs(self, dataset):
        probs = []
        for data_point in dataset:
            p_cj_xi = []
            for component in self.gmm_params:
                p_xi_cj = multivariate_normal.pdf(data_point, mean=component.mean, cov=component.variance, allow_singular=True)
                p_cj_xi.append(component.mixture_prop * p_xi_cj)
            p_cj_xi = np.array(p_cj_xi) / sum(p_cj_xi)
            probs.append(p_cj_xi)
        return np.array(probs)

    def compute_gmm(self, dataset, probs):
        n_samples = len(dataset)
        self.gmm_params = []
        for j in range(self.k):
            p_cj = sum(probs[i][j] for i in range(n_samples)) / n_samples
            mean_j = np.sum([probs[i][j] * dataset[i] for i in range(n_samples)], axis=0) / np.sum([probs[i][j] for i in range(n_samples)])
    
            # Accumulate the covariance matrix with regularization to avoid singular matrices
            var_j = np.sum([probs[i][j] * np.outer(dataset[i] - mean_j, dataset[i] - mean_j) for i in range(n_samples)], axis=0)
            var_j /= np.sum(probs[:, j])
            var_j += np.eye(dataset.shape[1]) * 1e-6  # Regularization term
    
            if np.any(np.isnan(var_j)) or np.any(np.isinf(var_j)):
                var_j = np.eye(dataset.shape[1]) * 1e-6  # Reset to prevent NaN or inf
    
            self.gmm_params.append(Component(p_cj, mean_j, var_j))
            
    def predict(self, dataset):
        probs = self.compute_probs(dataset)
        return np.argmax(probs, axis=1)

  

