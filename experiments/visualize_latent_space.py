import numpy as np
import matplotlib.pyplot as plt
import argparse

def visualize_latent_space_2d(npy_file):
    # 加载潜在空间数据
    latent_vectors = np.load(npy_file)
    print(f"Loaded latent vectors of shape: {latent_vectors.shape}")

    # 检查是否为二维数据
    if latent_vectors.shape[1] != 2:
        raise ValueError("Latent space data must be two-dimensional for direct visualization.")

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], alpha=0.6)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('2D Latent Space Visualization')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize 2D latent space from a .npy file')
    parser.add_argument('--npy_file', type=str, required=True, help='Path to the .npy file containing latent vectors')

    args = parser.parse_args()

    visualize_latent_space_2d(args.npy_file)
