import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# 加载数据
means = np.load('output/mu.npy')
std_devs = np.load('output/sigma.npy')

#print the shape of means and std_devs
print(means.shape)
print(std_devs.shape)

# 假设每个高斯分量的协方差是对角线的
covariances = [np.diag(std_dev**2) for std_dev in std_devs]

# 创建GMM实例
gmm = GaussianMixture(n_components=len(means), covariance_type='full', init_params='random')
gmm.means_ = means
gmm.covariances_ = covariances
gmm.weights_ = np.full(len(means), 1/len(means))  # 假设所有组件的初始权重相等

# 初始化精度矩阵（协方差的逆）
gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))

# 生成样本用于可视化
X, _ = gmm.sample(300)

# 绘制生成的数据点
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5, label='Generated Data')
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=100, marker='x', label='Centers')

# 可视化每个高斯分布的协方差范围
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
plt.title('Visualization of Prior Knowledge in the Latent Space using Dataset3')
plt.xlabel('Z1 Coordinate')
plt.ylabel('Z2 Coordinate')
plt.grid(True)
plt.show()


# 假设gmm是你已经训练好的模型
# 生成一系列点来计算其概率密度
x = np.linspace(-5, 5, 30)
y = np.linspace(-5, 5, 30)
X, Y = np.meshgrid(x, y)
XY = np.column_stack([X.ravel(), Y.ravel()])

# 计算这些点的对数似然
log_prob = gmm.score_samples(XY)
prob = np.exp(log_prob).reshape(X.shape)  # 取指数得到概率密度

# 可视化概率密度
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, prob, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Expectation of each location in the Latent Space using Dataset3 as Prior Knowledge')
plt.xlabel('Z1 Coordinate')
plt.ylabel('Z2 Coordinate')
plt.show()