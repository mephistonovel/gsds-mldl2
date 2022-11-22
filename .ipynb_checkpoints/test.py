import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky
from scipy.spatial.distance import pdist, squareform


def rbf(X, length_scale=1):
    dists = pdist(X / length_scale, metric="sqeuclidean")
    K = squareform(np.exp(-0.25 * dists))
    np.fill_diagonal(K, 1)
    return K

if __name__ == '__main__':
    X = np.arange(0, 10, 0.25).reshape(-1, 1)
    n, _ = X.shape
    K = rbf(X) + 1e-6 * np.eye(n)
    L = cholesky(K, lower=True)
    plt.plot(X, L @ np.random.randn(n, 5))
    plt.show()