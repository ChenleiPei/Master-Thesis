import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

def load_params():
    means = np.load('output/mu.npy')
    std_devs = np.load('output/sigma.npy')
    return means, std_devs

def initialize_gmm(means, std_devs):
    covariances = [np.diag(std_dev**2) for std_dev in std_devs]
    gmm = GaussianMixture(n_components=len(means), covariance_type='full', init_params='random')
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = np.full(len(means), 1/len(means))
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))
    return gmm

def visualize_gmm(gmm):
    X, _ = gmm.sample(300)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5, label='Generated Data')
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=100, marker='x', label='Centers')

    for mean, covar in zip(gmm.means_, gmm.covariances_):
        eigenvalues, eigenvectors = np.linalg.eigh(covar)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        vx, vy = eigenvectors[:, 0][0], eigenvectors[:, 0][1]
        theta = np.arctan2(vy, vx)
        for i in range(1, 4):  # 1-3标准差范围
            width = 2 * np.sqrt(eigenvalues[0]) * i
            height = 2 * np.sqrt(eigenvalues[1]) * i
            ell = Ellipse(xy=mean, width=width, height=height, angle=np.degrees(theta),
                          edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1.5, alpha=0.5)
            plt.gca().add_patch(ell)

    plt.legend()
    plt.title('Visualization of Gaussian Distributions')
    plt.xlabel('Z1 Coordinate')
    plt.ylabel('Z2 Coordinate')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    #means, std_devs = load_params()
    means = np.load('output/mu.npy')
    std_devs = np.load('output/sigma.npy')
    gmm = initialize_gmm(means, std_devs)
    visualize_gmm(gmm)
